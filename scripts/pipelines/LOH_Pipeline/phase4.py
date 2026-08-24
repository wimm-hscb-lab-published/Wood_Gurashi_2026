"""
Phase 4 – Non-LOH ↔ LOH Clone Correlation & Merge
====================================================

For each non-LOH clone, check whether it shares a base genotype name
with any LOH clone.  If it does, compute the per-cell mean allele-
frequency correlation between the two on the LOH arm's variants.
When correlation ≥ ``correlation_threshold``, merge the non-LOH cells
into the LOH clone (they are the same biological clone but were not
detected in Phase 1 due to weaker signal).

Call after Phase 3::

    from phase4 import run_phase4
    merged = run_phase4(state)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from .loh_config import LOHConfig, PipelineState


# ── ANSI helpers ───────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"


def _is_active_loh(label: object) -> bool:
    from .loh_config import is_loh_label
    return is_loh_label(label)


def _is_active_non_loh(label: object) -> bool:
    from .loh_config import is_loh_label, is_fp_label, is_sub_label, is_small_label
    return (
        (not is_loh_label(label))
        and (not is_fp_label(label))
        and (not is_sub_label(label))
        and (not is_small_label(label))
    )


def _get_base_name(label: str, loh_tag: str = "LOH") -> str:
    """
    Strip all LOH tokens from a clone label to recover the base genotype.

    E.g. 'TP53_E204Ter_Hem_BCORL1_T907Lfs18_Hom LOH-C0 chr5q'
      →  'TP53_E204Ter_Hem_BCORL1_T907Lfs18_Hom'
    """
    parts = str(label).split()
    base_parts = []
    skip_next = False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        if p.startswith(f"{loh_tag}-"):
            # Next token is the arm (e.g. 'chr5q'), skip it too
            skip_next = True
            continue
        base_parts.append(p)
    return " ".join(base_parts).strip()


def _zygosity_normalized_base_name(label: str, loh_tag: str = "LOH") -> str:
    """
    Return the base genotype with zygosity tags collapsed.

    Examples
    --------
    TP53_R248W_Het -> TP53_R248W_ZYG
    TP53_R248W_Hem -> TP53_R248W_ZYG

    This is used only for explicitly enabled Het -> Hem/Hom reassignment,
    so mutation identity must still match exactly.
    """
    base = _get_base_name(label, loh_tag=loh_tag)
    return re.sub(r"(?<=_)(Het|Hem|Hom)(?=_|$)", "ZYG", base, flags=re.IGNORECASE)


def _has_zygosity_tag(label: str, tags: tuple[str, ...], loh_tag: str = "LOH") -> bool:
    """Return True if the base genotype contains one of the requested zygosity tags."""
    base = _get_base_name(label, loh_tag=loh_tag)
    tag_pattern = "|".join([re.escape(t) for t in tags])
    return re.search(rf"(?<=_)({tag_pattern})(?=_|$)", base, flags=re.IGNORECASE) is not None


def _base_names_compatible(
    source_base: str,
    target_base: str,
    *,
    loh_tag: str = "LOH",
    allow_het_to_hem: bool = False,
) -> tuple[bool, str]:
    """
    Decide whether two base clone names are compatible for cell-level reassignment.

    Exact base-name equality always passes.  When allow_het_to_hem=True,
    a Het source may also match a Hem/Hom target if the zygosity-normalised
    base names are identical, e.g. TP53_R248W_Het -> TP53_R248W_Hem.
    """
    source_base = _get_base_name(str(source_base), loh_tag=loh_tag)
    target_base = _get_base_name(str(target_base), loh_tag=loh_tag)

    if source_base == target_base:
        return True, "exact"

    if allow_het_to_hem:
        if _zygosity_normalized_base_name(source_base, loh_tag=loh_tag) == _zygosity_normalized_base_name(target_base, loh_tag=loh_tag):
            source_is_het = _has_zygosity_tag(source_base, ("Het",), loh_tag=loh_tag)
            target_is_hem_or_hom = _has_zygosity_tag(target_base, ("Hem", "Hom"), loh_tag=loh_tag)
            if source_is_het and target_is_hem_or_hom:
                return True, "zygosity_normalized_het_to_hem"

    return False, "different_base"


def _extract_mutation(label: str) -> str | None:
    """
    Extract the TP53 mutation token from a clone label.

    Handles both underscore-delimited and colon-delimited formats:
      'TP53_R248W_Het'                      → 'TP53_R248W'
      'TP53_R248W_Hem LOH-C2 chr5q'        → 'TP53_R248W'
      'TP53:R273H_Het'                      → 'TP53:R273H'
      'TP53_E204Ter_Hem_BCORL1_T907Lfs18'  → 'TP53_E204Ter'

    Strategy: find 'TP53' followed by a separator (_  or :) then the
    mutation token, terminated by _ followed by Het/Hem/Hom.
    """
    import re
    # Try colon format first: TP53:R248W_Het
    m = re.search(r"(TP53:\S+?)_(Het|Hem|Hom)", label)
    if m:
        return m.group(1)
    # Underscore format: TP53_R248W_Het
    m = re.search(r"(TP53_[A-Za-z]\d+\S*?)_(Het|Hem|Hom)", label)
    if m:
        return m.group(1)
    return None



# ── Block-aware / ADO-tolerant rescue helpers ───────────────────────
def _build_variant_to_amplicon(state) -> dict[str, str]:
    """Return {variant_id: amplicon_id} from Sample.dna.col_attrs, falling back to variant IDs."""
    out: dict[str, str] = {}
    try:
        col_attrs = state.Sample.dna.col_attrs
        ids = list(col_attrs.get("id", []))
        amps = list(col_attrs.get("amplicon", []))
        if len(ids) == len(amps) and len(ids) > 0:
            out.update({str(i): str(a) for i, a in zip(ids, amps)})
    except Exception:
        pass
    try:
        for v in state.df.columns:
            out.setdefault(str(v), str(v))
    except Exception:
        pass
    return out


def _variant_pos(v: str) -> int:
    try:
        return int(str(v).split(":")[1].split("-")[0])
    except Exception:
        return 10**18


def _loh_block_variants_for_label(clone_label: str, state, arm: str | None = None) -> list[str] | None:
    """
    Return persisted/cytoband-inferred LOH block variants for *clone_label*.
    If *arm* is supplied, restrict to that arm's variants.
    """
    from .loh_config import extract_chr_arms, arm_variants
    import re as _re

    label = str(clone_label)
    out: list[str] = []

    # 1) Exact Phase 1 persisted map.
    block_map = getattr(state, "clone_block_variants", None) or {}
    if label in block_map and block_map[label]:
        out.extend([str(v) for v in block_map[label]])

    # 2) Phase 2 merged labels may not be exact keys.  If an original label is
    # embedded in the merged label, recover its persisted block variants.
    if not out and block_map:
        for k, vals in block_map.items():
            ks = str(k)
            if ks and (ks in label or label in ks):
                out.extend([str(v) for v in vals])

    # 3) Cytoband fallback from labels such as chr20p:p12.3-p11.21.
    if not out:
        cytobands_df = getattr(state, "cytobands_df", None)
        if cytobands_df is not None and len(cytobands_df) > 0:
            cytoband_pattern = _re.compile(
                r"(chr(?:\d+|X|Y)[pq]):([pq][\w.]+(?:-[pq][\w.]+)?)"
            )
            for arm_label, band_range in cytoband_pattern.findall(label):
                if arm is not None and arm_label != arm:
                    continue
                avars = arm_variants(arm_label, state)
                if not avars:
                    continue
                if "-" in band_range:
                    start_band, end_band = band_range.split("-", 1)
                else:
                    start_band = end_band = band_range
                arm_bands = cytobands_df[cytobands_df["chromosome arm"] == arm_label]
                if len(arm_bands) == 0:
                    continue
                matching = []
                in_range = False
                for _, row in arm_bands.sort_values("start").iterrows():
                    band_name = str(row["cytoband"])
                    if band_name == start_band:
                        in_range = True
                    if in_range:
                        matching.append(row)
                    if band_name == end_band:
                        break
                if not matching:
                    for _, row in arm_bands.sort_values("start").iterrows():
                        band_name = str(row["cytoband"])
                        if band_name.startswith(start_band) or band_name.startswith(end_band):
                            matching.append(row)
                if not matching:
                    continue
                region_start = min(int(r["start"]) for r in matching)
                region_end = max(int(r["end"]) for r in matching)
                for v in avars:
                    pos = _variant_pos(v)
                    if region_start <= pos <= region_end:
                        out.append(str(v))

    if arm is not None:
        try:
            arm_set = set(map(str, arm_variants(arm, state)))
            out = [v for v in out if v in arm_set]
        except Exception:
            pass

    out = [v for v in dict.fromkeys(out) if v in state.df.columns]
    return out if out else None


def _rescue_variants_for_clone_arm(loh_lbl: str, arm: str, state, cfg, arm_variants_fn) -> list[str]:
    """Use the LOH block for rescue when available; otherwise fall back to all arm variants."""
    if bool(getattr(cfg, "phase4_rescue_use_loh_block_variants", True)):
        bvars = _loh_block_variants_for_label(loh_lbl, state, arm=arm)
        if bvars and len(bvars) >= 2:
            return bvars
    return list(arm_variants_fn(arm, state))


def _allowed_rescue_ado_amplicons(n_amplicons: int, cfg) -> int:
    """
    Number of independent amplicons allowed to be missing/non-supporting during rescue.

    False-positive control: for small blocks, do not accept 2/3 or 3/4
    support. A one-amplicon ADO allowance is only used once the block has
    enough independent informative amplicons, controlled by
    phase4_rescue_min_amplicons_to_allow_ado (default 5).
    """
    n_amplicons = int(max(0, n_amplicons))
    min_n_for_ado = int(getattr(cfg, "phase4_rescue_min_amplicons_to_allow_ado", 5))
    if n_amplicons < min_n_for_ado:
        return 0

    explicit = getattr(cfg, "phase4_rescue_max_ado_amplicons", 0)
    if explicit is None:
        expected = float(getattr(cfg, "phase4_rescue_expected_amplicon_ado", 0.10))
        allowed = int(np.ceil(max(0.0, expected) * n_amplicons))
    else:
        allowed = int(explicit)
    return max(0, min(max(0, n_amplicons - 1), allowed))


def _informative_variant_set_for_rescue(cfg) -> set[str] | None:
    """
    Return cfg.informative_variants as strings, or None if unavailable.

    Rescue amp_support is calculated from germline-informative variants only
    when cfg.informative_variants is supplied. Somatic/context variants may
    remain in the LOH block for plotting and original block validation, but
    they must not inflate rescued-cell amplicon support.
    """
    vals = getattr(cfg, "informative_variants", None)
    if vals is None:
        return None
    try:
        vals = list(vals)
    except Exception:
        return None
    if len(vals) == 0:
        return None
    return set(str(v) for v in vals)


def _direction(vals: np.ndarray, low_af: float, high_af: float) -> str:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "NA"
    n_low = int(np.sum(vals <= low_af))
    n_high = int(np.sum(vals >= high_af))
    if n_low > 0 and n_high == 0:
        return "LOW"
    if n_high > 0 and n_low == 0:
        return "HIGH"
    if n_low > 0 and n_high > 0:
        return "MIXED"
    return "MID"


def _score_cell_against_loh_profile_ado_tolerant(
    cell_vals: np.ndarray,
    profile_vals: np.ndarray,
    variants: list[str],
    *,
    v2a: dict[str, str],
    cfg,
    low_af: float,
    high_af: float,
) -> dict:
    """
    Block-aware rescue score.

    Instead of requiring a fixed extreme fraction across all variants, score
    independent amplicons. When cfg.informative_variants is supplied, somatic/
    context variants are excluded from amp_support, so support is counted only
    on germline-informative block amplicons. A cell can be rescued if it
    supports at least n_amplicons - allowed_ADO_amplicons of the LOH block.
    By default, small blocks must have complete informative-amplicon support
    (3/3 or 4/4). One failed/ADO-like amplicon is only allowed for larger
    blocks, e.g. 4/5, once phase4_rescue_min_amplicons_to_allow_ado is met.
    """
    variants = [str(v) for v in variants]
    cell_vals = np.asarray(cell_vals, dtype=float)
    profile_vals = np.asarray(profile_vals, dtype=float)

    exclude_somatic = bool(getattr(cfg, "phase4_rescue_exclude_somatic_from_amp_support", True))
    informative_set = _informative_variant_set_for_rescue(cfg) if exclude_somatic else None

    amp_to_idx_all: dict[str, list[int]] = {}
    amp_to_idx: dict[str, list[int]] = {}
    n_somatic_excluded_variants = 0
    somatic_excluded_amps: set[str] = set()

    for i, v in enumerate(variants):
        amp = str(v2a.get(v, v))
        amp_to_idx_all.setdefault(amp, []).append(i)

        if informative_set is not None and str(v) not in informative_set:
            n_somatic_excluded_variants += 1
            somatic_excluded_amps.add(amp)
            continue

        amp_to_idx.setdefault(amp, []).append(i)

    n_total_block_amp = len(amp_to_idx_all)
    n_total_amp = len(amp_to_idx)
    allowed = _allowed_rescue_ado_amplicons(n_total_amp, cfg)
    min_required_floor = int(getattr(cfg, "phase4_rescue_min_required_amplicons", 2))
    n_required_amp = max(min_required_floor, n_total_amp - allowed)
    n_required_amp = min(n_required_amp, n_total_amp)

    amp_variant_extreme_frac = float(getattr(cfg, "phase4_rescue_amplicon_variant_extreme_frac", 0.75))
    require_dir = bool(getattr(cfg, "phase4_rescue_require_direction_match", True))

    pass_amp_indices: list[int] = []
    scoreable_amp = 0
    n_pass_amp = 0
    amp_status = []

    for amp, idxs in amp_to_idx.items():
        idx = np.asarray(idxs, dtype=int)
        cv = cell_vals[idx]
        pv = profile_vals[idx]
        finite = np.isfinite(cv) & np.isfinite(pv)
        if finite.sum() == 0:
            amp_status.append((amp, "missing"))
            continue

        scoreable_amp += 1
        cvf = cv[finite]
        pvf = pv[finite]
        extreme_frac = float(np.mean((cvf <= low_af) | (cvf >= high_af)))
        cell_extreme = extreme_frac >= amp_variant_extreme_frac

        cell_dir = _direction(cvf, low_af, high_af)
        prof_dir = _direction(pvf, low_af, high_af)
        dir_ok = True
        if require_dir and prof_dir in {"LOW", "HIGH"}:
            dir_ok = cell_dir == prof_dir

        if cell_extreme and dir_ok:
            n_pass_amp += 1
            pass_amp_indices.extend(idx[finite].tolist())
            amp_status.append((amp, f"pass:{cell_dir}"))
        else:
            why = "mid" if not cell_extreme else f"dir:{cell_dir}!={prof_dir}"
            amp_status.append((amp, why))

    amp_support_frac = n_pass_amp / n_total_amp if n_total_amp else 0.0
    support_ok = n_pass_amp >= n_required_amp

    min_corr_variants = int(getattr(cfg, "phase4_rescue_min_corr_variants", 3))
    pass_amp_indices = sorted(set(pass_amp_indices))
    r_val = np.nan
    corr_ok = False

    if len(pass_amp_indices) >= min_corr_variants:
        cv = cell_vals[pass_amp_indices]
        pv = profile_vals[pass_amp_indices]
        finite = np.isfinite(cv) & np.isfinite(pv)
        cv = cv[finite]
        pv = pv[finite]
        if cv.size >= min_corr_variants:
            sd_c = float(np.nanstd(cv))
            sd_p = float(np.nanstd(pv))
            if sd_c < 1e-6 or sd_p < 1e-6:
                # Correlation is undefined for constant LOW/HIGH blocks.  Treat
                # it as a match only if the absolute profile difference is small.
                max_abs_diff = float(getattr(cfg, "phase4_rescue_constant_profile_max_abs_diff", 15.0))
                if bool(getattr(cfg, "phase4_rescue_constant_profiles_are_match", True)):
                    if float(np.nanmean(np.abs(cv - pv))) <= max_abs_diff:
                        r_val = 1.0
                    else:
                        r_val = 0.0
                else:
                    r_val = np.nan
            else:
                try:
                    r_val, _ = pearsonr(cv, pv)
                except Exception:
                    r_val = np.nan

    pc_min_r = float(getattr(cfg, "phase4_percell_min_corr", 0.60))
    if np.isfinite(r_val):
        corr_ok = r_val >= pc_min_r
    elif bool(getattr(cfg, "phase4_rescue_allow_amplicon_support_without_corr", False)):
        corr_ok = support_ok

    passes = bool(support_ok and corr_ok)

    return {
        "passes": passes,
        "r": float(r_val) if np.isfinite(r_val) else np.nan,
        "amp_support_frac": float(amp_support_frac),
        "n_pass_amplicons": int(n_pass_amp),
        "n_required_amplicons": int(n_required_amp),
        "n_total_amplicons": int(n_total_amp),
        "n_total_block_amplicons": int(n_total_block_amp),
        "n_scoreable_amplicons": int(scoreable_amp),
        "n_allowed_ado_amplicons": int(allowed),
        "n_somatic_excluded_variants": int(n_somatic_excluded_variants),
        "n_somatic_excluded_amplicons": int(len(somatic_excluded_amps)),
        "amp_support_excludes_somatic": bool(informative_set is not None and exclude_somatic),
        "amp_status": amp_status,
    }

def _het_ado_check(
    state: "PipelineState",
    cfg: "LOHConfig",
) -> int:
    """
    Phase 4b: WT/Het → Hem/Hom ADO/somatic-amplicon check.

    Updated rule:
      • Target clones are Hem-like and, optionally, Hom-like clones.
      • Candidate clones are WT-like or Het-like clones.
      • A candidate cell is compared to the Hem/Hom target profile across the
        full AF space, excluding only the amplicon(s) carrying the target
        somatic mutation(s).
      • No amplicon ADO probability model is used.
      • Passing cells are relabelled as ADO and removed from state.df / df_numeric.
    """
    from .loh_config import (
        LOH_TAG, ADO_LABEL, vprint, is_hem_like, fmt_float, is_wt_clone,
    )
    import re as _re

    if not bool(getattr(cfg, "ado_check_enabled", True)):
        return 0

    min_corr = float(getattr(cfg, "ado_min_corr", 0.80))
    max_mad = float(getattr(cfg, "ado_max_median_abs_diff", 15.0))
    min_variants = int(getattr(cfg, "ado_min_comparison_variants", 20))
    exclude_somatic_amp = bool(getattr(cfg, "ado_exclude_somatic_amplicon", True))
    include_hom = bool(getattr(cfg, "ado_include_hom_targets", True))
    wt_global_targets = bool(getattr(cfg, "ado_wt_compare_all_hemhom_loh_targets", True))

    vprint("\n" + "-" * 60, cfg)
    vprint("PHASE 4.1: ADO Check (WT/Het → Hem/Hom, excluding somatic amplicon)", cfg)
    vprint("-" * 60, cfg)
    vprint(
        f"  ADO criteria: per-cell r ≥ {min_corr} and median |ΔAF| ≤ {max_mad:g}",
        cfg,
    )
    vprint(
        "  Comparison: all available AF variants except the amplicon covering "
        "the target Hem/Hom somatic mutation.",
        cfg,
    )
    vprint("  No ADO-combination probability model is used.", cfg)

    # ── Helpers ─────────────────────────────────────────────────────
    def _is_hom_like(label: str) -> bool:
        return _re.search(
            r"(^|[^A-Za-z0-9])hom([^A-Za-z0-9]|$)",
            str(label),
            flags=_re.IGNORECASE,
        ) is not None or _re.search(r"_Hom\b", str(label)) is not None

    def _is_hemhom_like(label: str) -> bool:
        return is_hem_like(label) or (include_hom and _is_hom_like(label))

    def _dedupe_preserve_order(xs: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for x in xs:
            sx = str(x)
            if sx in seen:
                continue
            seen.add(sx)
            out.append(sx)
        return out

    def _somatic_tokens(lbl: str) -> list[str]:
        """Return mutation cores such as TP53_C277Y or BRAF_G466A from clone label."""
        s = str(lbl)
        # Handles TP53_C277Y_Hem, TP53:C277Y_Hom, BRAF_G466A_Het, etc.
        pats = [
            r"\b([A-Za-z][A-Za-z0-9]*[:_][A-Za-z0-9][A-Za-z0-9*.+-]*?)_(?:Het|Hem|Hom)\b",
        ]
        out = []
        for pat in pats:
            for m in _re.finditer(pat, s):
                tok = m.group(1).strip()
                if tok and tok not in out:
                    out.append(tok)
        return out

    def _ado_base_key(lbl: str) -> str:
        """
        Base key for pairing Het↔Hem/Hom.  Removes LOH labels and zygosity
        suffixes while retaining non-somatic clone descriptors.
        """
        base = _get_base_name(lbl, LOH_TAG)
        # Replace mutation zygosity tokens by mutation core.
        base = _re.sub(
            r"\b([A-Za-z][A-Za-z0-9]*[:_][A-Za-z0-9][A-Za-z0-9*.+-]*?)_(Het|Hem|Hom)\b",
            r"\1",
            base,
        ).strip()
        # Drop leading WT token for WT candidates.
        parts = base.split()
        if parts and parts[0] == "WT":
            parts = parts[1:]
        return " ".join(parts).strip()

    # Build variant -> amplicon and variant -> annotation text maps.
    v2a = _build_variant_to_amplicon(state)
    variant_cols = [str(v) for v in getattr(state, "variant_cols", []) if str(v) in state.df.columns]
    if not variant_cols:
        variant_cols = [str(v) for v in state.df_numeric.columns if str(v) in state.df.columns]

    col_text: dict[str, str] = {v: v for v in variant_cols}
    try:
        col_attrs = state.Sample.dna.col_attrs
        ids = [str(x) for x in col_attrs.get("id", [])]
        id_to_i = {vid: i for i, vid in enumerate(ids)}
        for v in variant_cols:
            i = id_to_i.get(v)
            pieces = [v]
            if i is not None:
                for key, arr in col_attrs.items():
                    try:
                        if i < len(arr):
                            pieces.append(str(arr[i]))
                    except Exception:
                        continue
            col_text[v] = " ".join(pieces)
    except Exception:
        pass

    def _normalise(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower()

    def _amplicons_for_somatic_tokens(tokens: list[str]) -> set[str]:
        amps: set[str] = set()
        for tok in tokens:
            tok_norm = _normalise(tok)
            if ":" in tok:
                gene, mut = tok.split(":", 1)
            elif "_" in tok:
                gene, mut = tok.split("_", 1)
            else:
                gene, mut = tok, ""
            gene_norm = _normalise(gene)
            mut_norm = _normalise(mut)
            for v, text in col_text.items():
                txt_norm = _normalise(text)
                match = False
                if tok_norm and tok_norm in txt_norm:
                    match = True
                elif gene_norm and mut_norm and gene_norm in txt_norm and mut_norm in txt_norm:
                    match = True
                if match:
                    amps.add(str(v2a.get(v, v)))
        return amps

    def _comparison_variants_for_target(target_lbl: str) -> tuple[list[str], set[str], list[str]]:
        tokens = _somatic_tokens(target_lbl)
        somatic_amps = _amplicons_for_somatic_tokens(tokens) if exclude_somatic_amp else set()
        comps = [v for v in variant_cols if str(v2a.get(v, v)) not in somatic_amps]
        comps = _dedupe_preserve_order(comps)
        return comps, somatic_amps, tokens

    # ── Identify targets and candidates ──────────────────────────────
    target_clones: dict[str, list] = {}
    target_loh_clones: dict[str, list] = {}
    candidate_het: dict[str, list] = {}
    candidate_wt: dict[str, list] = {}

    for lbl, cells in state.clone_to_cells.items():
        lbl_str = str(lbl)
        if not cells or (not _is_active_loh(lbl_str) and not _is_active_non_loh(lbl_str)):
            continue
        if _is_hemhom_like(lbl_str):
            target_clones[lbl_str] = cells
            if _is_active_loh(lbl_str):
                target_loh_clones[lbl_str] = cells
            continue
        if "_Het" in lbl_str:
            candidate_het[lbl_str] = cells
        elif is_wt_clone(lbl_str, cfg):
            candidate_wt[lbl_str] = cells

    if not target_clones:
        vprint("  No Hem/Hom target clones found — skipping.", cfg)
        return 0
    if not candidate_het and not candidate_wt:
        vprint("  No Het or WT candidate clones found — skipping.", cfg)
        return 0

    vprint(f"  Hem/Hom targets: {len(target_clones)}  (LOH targets for WT: {len(target_loh_clones)})", cfg)
    vprint(f"  Het candidates:  {len(candidate_het)}", cfg)
    vprint(f"  WT candidates:   {len(candidate_wt)}", cfg)

    targets_by_key: dict[str, list[str]] = {}
    loh_targets_by_key: dict[str, list[str]] = {}
    for target_lbl in target_clones:
        k = _ado_base_key(target_lbl)
        if k:
            targets_by_key.setdefault(k, []).append(target_lbl)
    for target_lbl in target_loh_clones:
        k = _ado_base_key(target_lbl)
        if k:
            loh_targets_by_key.setdefault(k, []).append(target_lbl)

    target_profile_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    ado_barcodes: set[str] = set()
    best_hits: dict[str, tuple[float, float, int, int, str, str]] = {}

    def _evaluate_against_targets(cand_lbl: str, cand_cells: list, target_labels: list[str]) -> None:
        if not target_labels:
            return
        vprint(
            f"\n  Candidate: {cand_lbl} ({len(cand_cells)} cells) → "
            f"targets={len(target_labels)} Hem/Hom clone(s)",
            cfg,
        )
        for target_lbl in target_labels:
            target_cells = target_clones.get(target_lbl, [])
            if not target_cells:
                continue
            comp_vars, excluded_amps, tokens = _comparison_variants_for_target(target_lbl)
            if len(comp_vars) < min_variants:
                vprint(
                    f"    Target {target_lbl}: skipped; only {len(comp_vars)} comparison variants "
                    f"after excluding somatic amp(s) {sorted(excluded_amps)} for tokens={tokens}",
                    cfg,
                )
                continue
            target_bcs = [str(bc) for bc in target_cells if str(bc) in state.df.index]
            if len(target_bcs) < 2:
                continue
            key = (target_lbl, tuple(comp_vars))
            if key in target_profile_cache:
                target_profile = target_profile_cache[key]
            else:
                target_profile = state.df.loc[target_bcs, comp_vars].mean(axis=0).values.astype(float)
                target_profile_cache[key] = target_profile
            if np.nanstd(target_profile) < 1e-6:
                continue

            vprint(
                f"    Target {target_lbl}: compare {len(comp_vars)} variants; "
                f"excluded_somatic_amplicons={sorted(excluded_amps) if excluded_amps else 'none'}",
                cfg,
            )

            for bc in cand_cells:
                bc_str = str(bc)
                if bc_str in ado_barcodes or bc_str not in state.df.index:
                    continue
                cell_vals = state.df.loc[bc_str, comp_vars].values.astype(float)
                mask = np.isfinite(cell_vals) & np.isfinite(target_profile)
                n = int(mask.sum())
                if n < min_variants:
                    continue
                if np.nanstd(cell_vals[mask]) < 1e-6:
                    continue
                r, _ = pearsonr(cell_vals[mask], target_profile[mask])
                mad = float(np.nanmedian(np.abs(cell_vals[mask] - target_profile[mask])))
                if np.isfinite(r) and r >= min_corr and mad <= max_mad:
                    ado_barcodes.add(bc_str)
                    best_hits[bc_str] = (float(r), mad, n, len(excluded_amps), target_lbl, cand_lbl)

    # Het candidates: use same mutation/base key when available.
    for het_lbl, het_cells in sorted(candidate_het.items(), key=lambda kv: -len(kv[1])):
        k = _ado_base_key(het_lbl)
        targets = targets_by_key.get(k, [])
        vprint(f"\n  Het candidate key={k!r} → Hem/Hom targets={len(targets)}", cfg)
        _evaluate_against_targets(het_lbl, het_cells, targets)

    # WT candidates: compare to same-key LOH Hem/Hom targets; if no key, allow
    # comparison against all Hem/Hom LOH targets because the somatic amplicon is
    # explicitly excluded from scoring.
    all_loh_targets = list(target_loh_clones.keys())
    for wt_lbl, wt_cells in sorted(candidate_wt.items(), key=lambda kv: -len(kv[1])):
        k = _ado_base_key(wt_lbl)
        targets = loh_targets_by_key.get(k, []) if k else []
        if wt_global_targets and not targets:
            targets = all_loh_targets
        vprint(f"\n  WT candidate key={k!r} → Hem/Hom(LOH) targets={len(targets)}", cfg)
        _evaluate_against_targets(wt_lbl, wt_cells, targets)

    # ── Apply ADO labels ──────────────────────────────────────────────
    if ado_barcodes:
        vprint(f"\n  {_YELLOW}ADO flagged cells: {len(ado_barcodes)}{_RESET}", cfg)
        for bc_str, (r, mad, n, n_excl, target_lbl, cand_lbl) in sorted(
            best_hits.items(), key=lambda kv: (-kv[1][0], kv[1][1])
        ):
            vprint(
                f"    {_YELLOW}⚠ {bc_str}: r={fmt_float(r)}, median|ΔAF|={mad:.2f}, "
                f"nvar={n}, excluded_somatic_amplicons={n_excl}, "
                f"candidate={cand_lbl}, best_match={target_lbl} → ADO{_RESET}",
                cfg,
            )

        state.df = state.df.loc[~state.df.index.isin(ado_barcodes)].copy()
        state.df_numeric = state.df_numeric.loc[~state.df_numeric.index.isin(ado_barcodes)].copy()

        sample_barcodes = pd.Series(state.Sample.dna.row_attrs["barcode"]).astype(str)
        ado_mask = sample_barcodes.isin(ado_barcodes).values
        clone_arr = state.Sample.dna.row_attrs["Clone"]
        clone_arr = np.where(ado_mask, ADO_LABEL, clone_arr)
        state.Sample.dna.row_attrs["Clone"] = clone_arr
        state.refresh()
        vprint(f"\n  ✓ Relabelled {len(ado_barcodes)} cell(s) as '{ADO_LABEL}'", cfg)
    else:
        vprint("\n  No ADO cells detected across WT/Het candidates.", cfg)

    return len(ado_barcodes)

def _arm_pooled_rescue(
    state,
    cfg,
    LOH_TAG: str,
    arm_variants_fn,
    extract_chr_arms_fn,
    low_af: float,
    high_af: float,
) -> int:
    """
    Fix 1 — Arm-level pooled rescue pass (Phase 4c).

    For each LOH arm currently in state, build a pooled mean AF profile from
    ALL cells carrying LOH on that arm (regardless of which LOH clone they
    belong to).  Then, for each non-LOH clone whose base-genotype name matches
    AT LEAST ONE existing LOH clone's base name, check every individual cell
    against the pooled profile.  Cells that pass the per-cell correlation and
    extreme-fraction thresholds are rescued into:
      (a) The existing matching LOH clone for that arm + base-name, if one
          exists; or
      (b) A newly created LOH clone label, if none exists yet for that
          base-name × arm combination.

    This catches the case where chr4q LOH in TET2_... cells was never rescued
    because there was no same-name chr4q LOH clone surviving Phase 1 — the
    existing Phase 4 per-cell rescue only runs *against* surviving LOH clones.

    Returns the number of new rescue operations performed.
    """
    from .loh_config import vprint, fmt_float
    import re as _re

    pc_min_r = float(getattr(cfg, "phase4_percell_min_corr", 0.60))
    pc_min_ef = float(getattr(cfg, "phase4_percell_min_extreme", 0.70))
    min_rescued_cells = int(getattr(cfg, "phase4_arm_rescue_min_cells", cfg.min_final_clone_size))
    enabled = bool(getattr(cfg, "phase4_arm_rescue_enabled", True))

    if not enabled:
        return 0

    vprint("\n" + "-" * 60, cfg)
    vprint("PHASE 4.2: Arm-level pooled rescue", cfg)
    vprint("-" * 60, cfg)
    allowed_msg = getattr(cfg, "phase4_rescue_max_ado_amplicons", 1)
    vprint(
        f"  Criteria: block-aware rescue, per-cell r ≥ {pc_min_r}, "
        f"allow ≤{allowed_msg} non-supporting/ADO amplicon(s) "
        f"only when block has ≥{getattr(cfg, 'phase4_rescue_min_amplicons_to_allow_ado', 5)} informative amplicons, "
        f"min rescued cells ≥ {min_rescued_cells}",
        cfg,
    )

    state.refresh()

    # ── Step 1: identify all arms currently with LOH cells ──────────
    arm_to_loh_cells: dict[str, list] = {}
    arm_to_block_variants: dict[str, list[str]] = {}
    for lbl, cells in state.clone_to_cells.items():
        if not _is_active_loh(lbl):
            continue
        for arm in extract_chr_arms_fn(lbl):
            arm_to_loh_cells.setdefault(arm, [])
            arm_to_loh_cells[arm].extend(cells)
            bvars = _rescue_variants_for_clone_arm(str(lbl), arm, state, cfg, arm_variants_fn)
            if bvars:
                arm_to_block_variants.setdefault(arm, [])
                arm_to_block_variants[arm].extend([str(v) for v in bvars])

    if not arm_to_loh_cells:
        vprint("  No LOH arms found — skipping.", cfg)
        return 0

    # ── Step 2: build per-arm pooled profiles ───────────────────────
    # Use union of surviving LOH block variants for that arm when available;
    # this avoids rescuing on noisy arm-wide variants outside the discovered block.
    arm_profiles: dict[str, tuple[np.ndarray, list[str]]] = {}
    for arm, cells in arm_to_loh_cells.items():
        variants = list(dict.fromkeys(arm_to_block_variants.get(arm, [])))
        if len(variants) < 2:
            variants = list(arm_variants_fn(arm, state))
        variants = [str(v) for v in variants if str(v) in state.df.columns]
        if len(variants) < 2:
            continue
        bcs = [str(c) for c in cells if str(c) in state.df.index]
        if len(bcs) < 2:
            continue
        prof = state.df.loc[bcs, variants].mean(axis=0).values.astype(float)
        arm_profiles[arm] = (prof, variants)

    if not arm_profiles:
        vprint("  Could not build any arm profiles — skipping.", cfg)
        return 0

    # ── Step 3: build base-name index for existing LOH clones ────────
    # base_name → {arm → loh_label}
    loh_base_arm: dict[str, dict[str, str]] = {}
    for lbl in state.clone_to_cells:
        if not _is_active_loh(lbl):
            continue
        base = _get_base_name(lbl, LOH_TAG)
        for arm in extract_chr_arms_fn(lbl):
            loh_base_arm.setdefault(base, {})[arm] = lbl

    # ── Step 4: collect non-LOH clones ───────────────────────────────
    non_loh_clones = {
        lbl: cells
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_non_loh(lbl) and len(cells) > 0
    }

    if not non_loh_clones:
        vprint("  No non-LOH clones — skipping.", cfg)
        return 0

    # ── Step 5: for each non-LOH clone, check each arm ───────────────
    ops_done = 0

    for non_loh_lbl, non_loh_cells in sorted(
        non_loh_clones.items(), key=lambda kv: -len(kv[1])
    ):
        base = _get_base_name(non_loh_lbl, LOH_TAG)

        # Only process clones whose base name appears in at least one LOH clone
        if base not in loh_base_arm:
            continue

        for arm, (prof, variants) in arm_profiles.items():
            # Skip arm if this non-LOH clone's base already has an LOH clone
            # for this arm AND the non-LOH clone was already processed by the
            # standard Phase 4 loop (we only handle NEW arm assignments here)
            already_has_loh_for_arm = arm in loh_base_arm.get(base, {})

            # Also skip if the non-LOH clone itself already carries this arm
            if arm in extract_chr_arms_fn(non_loh_lbl):
                continue

            # Per-cell rescue against pooled profile
            candidate_bcs = [str(c) for c in non_loh_cells if str(c) in state.df.index]
            if not candidate_bcs:
                continue

            rescued_bcs: list[str] = []
            v2a = _build_variant_to_amplicon(state)

            for bc_str in candidate_bcs:
                cell_vals = state.df.loc[bc_str, variants].values.astype(float)
                score = _score_cell_against_loh_profile_ado_tolerant(
                    cell_vals,
                    prof,
                    variants,
                    v2a=v2a,
                    cfg=cfg,
                    low_af=low_af,
                    high_af=high_af,
                )
                if score["passes"]:
                    rescued_bcs.append(bc_str)

            if len(rescued_bcs) < min_rescued_cells:
                continue

            # Determine target LOH label
            if already_has_loh_for_arm:
                target_lbl = loh_base_arm[base][arm]
            else:
                # Create a new LOH clone label for this base × arm
                existing_c = []
                for lbl2 in state.clone_to_cells:
                    if not _is_active_loh(lbl2):
                        continue
                    if arm not in extract_chr_arms_fn(lbl2):
                        continue
                    m = _re.search(r"LOH-C(\d+)", str(lbl2))
                    if m:
                        existing_c.append(int(m.group(1)))
                next_c = max(existing_c, default=-1) + 1
                target_lbl = f"{base} LOH-C{next_c} {arm}"

            vprint(
                f"\n  [{arm}] '{non_loh_lbl}' ({len(non_loh_cells)} cells) "
                f"→ {len(rescued_bcs)} cells rescued into '{target_lbl}'",
                cfg,
            )
            for bc_str in rescued_bcs:
                vprint(f"    ✓ {bc_str} → {target_lbl}", cfg)

            # Apply relabelling
            clone_arr = state.Sample.dna.row_attrs["Clone"]
            barcode_arr = pd.Series(
                state.Sample.dna.row_attrs["barcode"]
            ).astype(str).values

            for bc_str in rescued_bcs:
                mask = barcode_arr == bc_str
                if mask.any():
                    clone_arr[mask] = target_lbl

            state.Sample.dna.row_attrs["Clone"] = clone_arr

            # Update clone_to_cells
            existing_in_target = list(state.clone_to_cells.get(target_lbl, []))
            state.clone_to_cells[target_lbl] = list(
                set(existing_in_target + rescued_bcs)
            )
            remaining = [c for c in non_loh_cells if str(c) not in set(rescued_bcs)]
            if remaining:
                state.clone_to_cells[non_loh_lbl] = remaining
            elif non_loh_lbl in state.clone_to_cells:
                del state.clone_to_cells[non_loh_lbl]

            # Update loh_base_arm index for subsequent iterations
            loh_base_arm.setdefault(base, {})[arm] = target_lbl

            ops_done += 1

    vprint(f"\n  Arm-level pooled rescue: {ops_done} operation(s) performed.", cfg)
    state.refresh()
    return ops_done


def _loh_arm_expansion(
    state,
    cfg,
    LOH_TAG: str,
    arm_variants_fn,
    extract_chr_arms_fn,
    low_af: float,
    high_af: float,
) -> int:
    """
    Phase 4c-ii — LOH arm expansion (second pass over existing LOH clones).

    For each existing LOH clone, check whether its cells *also* carry LOH
    on additional arms not yet in the clone label.  The test uses the
    per-arm pooled profile (built from ALL cells currently assigned LOH on
    that arm across all clones).

    • If ≥ ``unanimous_frac`` of the clone's cells pass the per-cell
      correlation and extreme-fraction thresholds on a new arm, the whole
      clone is relabelled to include that arm.
    • If fewer pass but still ≥ ``min_cells``, the passing cells are split
      into a new clone that carries the expanded label, and the remainder
      keep the original label.
    • Otherwise the arm is skipped for that clone.

    Returns the number of expansion operations performed.
    """
    from .loh_config import vprint, fmt_float
    import re as _re

    enabled = bool(getattr(cfg, "phase4_arm_expansion_enabled", True))
    if not enabled:
        return 0

    pc_min_r = float(getattr(cfg, "phase4_arm_expansion_min_corr", 0.60))
    pc_min_ef = float(getattr(cfg, "phase4_arm_expansion_min_extreme", 0.70))
    unanimous_frac = float(getattr(cfg, "phase4_arm_expansion_unanimous_frac", 0.90))
    min_cells = int(getattr(cfg, "phase4_arm_expansion_min_cells", 5))

    vprint("\n" + "-" * 60, cfg)
    vprint("PHASE 4.3: LOH arm expansion (existing clone second pass)", cfg)
    vprint("-" * 60, cfg)
    vprint(
        f"  Criteria: per-cell r ≥ {pc_min_r}, extreme ≥ {pc_min_ef}, "
        f"unanimous ≥ {unanimous_frac:.0%}, min cells ≥ {min_cells}",
        cfg,
    )

    state.refresh()

    # ── Step 1: identify all arms with LOH cells & build pooled profiles ─
    arm_to_loh_cells: dict[str, list] = {}
    for lbl, cells in state.clone_to_cells.items():
        if not _is_active_loh(lbl):
            continue
        for arm in extract_chr_arms_fn(lbl):
            arm_to_loh_cells.setdefault(arm, [])
            arm_to_loh_cells[arm].extend(cells)

    arm_profiles: dict[str, tuple[np.ndarray, list[str]]] = {}
    for arm, cells in arm_to_loh_cells.items():
        variants = arm_variants_fn(arm, state)
        if len(variants) < 2:
            continue
        bcs = [str(c) for c in cells if str(c) in state.df.index]
        if len(bcs) < 2:
            continue
        prof = state.df.loc[bcs, variants].mean(axis=0).values.astype(float)
        if np.nanstd(prof) < 1e-6:
            continue
        arm_profiles[arm] = (prof, variants)

    if not arm_profiles:
        vprint("  No arm profiles available — skipping.", cfg)
        return 0

    vprint(f"  Pooled profiles built for arms: {sorted(arm_profiles.keys())}", cfg)

    # ── Step 2: iterate over each LOH clone ──────────────────────────────
    ops_done = 0

    # Snapshot clone list to avoid mutating dict during iteration
    loh_snapshot = [
        (lbl, list(cells))
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_loh(lbl) and len(cells) > 0
    ]

    for clone_lbl, clone_cells in loh_snapshot:
        existing_arms = set(extract_chr_arms_fn(clone_lbl))

        # Check each arm not already carried by this clone
        for arm, (prof, variants) in arm_profiles.items():
            if arm in existing_arms:
                continue

            # Per-cell test against pooled profile for this arm
            candidate_bcs = [str(c) for c in clone_cells if str(c) in state.df.index]
            if not candidate_bcs:
                continue

            passing: list[str] = []
            failing: list[str] = []

            for bc_str in candidate_bcs:
                cell_vals = state.df.loc[bc_str, variants].values.astype(float)
                mask = np.isfinite(cell_vals) & np.isfinite(prof)
                if mask.sum() < 3:
                    failing.append(bc_str)
                    continue
                if np.nanstd(cell_vals[mask]) < 1e-6:
                    failing.append(bc_str)
                    continue

                r_val, _ = pearsonr(cell_vals[mask], prof[mask])
                if not np.isfinite(r_val):
                    failing.append(bc_str)
                    continue

                valid = cell_vals[np.isfinite(cell_vals)]
                ef = (
                    float(np.mean((valid <= low_af) | (valid >= high_af)))
                    if len(valid) > 0
                    else 0.0
                )

                if r_val >= pc_min_r and ef >= pc_min_ef:
                    passing.append(bc_str)
                else:
                    failing.append(bc_str)

            if len(passing) < min_cells:
                continue

            pass_frac = len(passing) / max(len(candidate_bcs), 1)

            # ── Build the expanded label ─────────────────────────────
            # Find the next LOH-C index for this arm globally
            existing_c: list[int] = []
            for lbl2 in list(state.clone_to_cells.keys()):
                if not _is_active_loh(lbl2):
                    continue
                if arm not in extract_chr_arms_fn(lbl2):
                    continue
                m = _re.search(r"LOH-C(\d+)", str(lbl2))
                if m:
                    existing_c.append(int(m.group(1)))
            next_c = max(existing_c, default=-1) + 1

            expanded_lbl = f"{clone_lbl} LOH-C{next_c} {arm}"

            if pass_frac >= unanimous_frac:
                # ── Unanimous: relabel entire clone ──────────────────
                vprint(
                    f"\n  {_GREEN}[{arm}] '{clone_lbl}' → RELABEL entire clone "
                    f"({len(passing)}/{len(candidate_bcs)} = {pass_frac:.0%} pass){_RESET}",
                    cfg,
                )
                vprint(f"    New label: '{expanded_lbl}'", cfg)

                clone_arr = state.Sample.dna.row_attrs["Clone"]
                barcode_arr = pd.Series(
                    state.Sample.dna.row_attrs["barcode"]
                ).astype(str).values

                # Relabel all cells currently in this clone
                all_clone_bcs = [str(c) for c in clone_cells]
                for bc_str in all_clone_bcs:
                    mask = barcode_arr == bc_str
                    if mask.any():
                        clone_arr[mask] = expanded_lbl

                state.Sample.dna.row_attrs["Clone"] = clone_arr

                # Update clone_to_cells
                state.clone_to_cells[expanded_lbl] = list(
                    state.clone_to_cells.pop(clone_lbl, clone_cells)
                )

                ops_done += 1

                # Update existing_arms so subsequent arm checks for this
                # clone use the new label context
                existing_arms.add(arm)

                # Update clone_lbl reference for subsequent arm iterations
                clone_lbl = expanded_lbl

            else:
                # ── Split: passing cells get expanded label, rest keep original ─
                vprint(
                    f"\n  {_YELLOW}[{arm}] '{clone_lbl}' → SPLIT "
                    f"({len(passing)}/{len(candidate_bcs)} = {pass_frac:.0%} pass, "
                    f"below {unanimous_frac:.0%} unanimity){_RESET}",
                    cfg,
                )
                vprint(f"    Expanded label: '{expanded_lbl}' ({len(passing)} cells)", cfg)
                vprint(f"    Remaining in:   '{clone_lbl}' ({len(failing)} cells)", cfg)

                clone_arr = state.Sample.dna.row_attrs["Clone"]
                barcode_arr = pd.Series(
                    state.Sample.dna.row_attrs["barcode"]
                ).astype(str).values

                for bc_str in passing:
                    mask = barcode_arr == bc_str
                    if mask.any():
                        clone_arr[mask] = expanded_lbl

                state.Sample.dna.row_attrs["Clone"] = clone_arr

                # Update clone_to_cells
                state.clone_to_cells[expanded_lbl] = passing
                remaining_cells = [
                    c for c in clone_cells if str(c) not in set(passing)
                ]
                if remaining_cells:
                    state.clone_to_cells[clone_lbl] = remaining_cells
                elif clone_lbl in state.clone_to_cells:
                    del state.clone_to_cells[clone_lbl]

                ops_done += 1

    vprint(f"\n  LOH arm expansion: {ops_done} operation(s) performed.", cfg)
    state.refresh()
    return ops_done



def _safe_similarity_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    constant_profile_max_mad: float,
    constant_profiles_are_match: bool = True,
) -> float:
    """
    Robust clone-profile correlation.

    Pearson correlation is undefined when either LOH profile is essentially
    constant. For real LOH blocks this is common, especially when all variants
    sit near 0 or 100. In that case we can optionally treat profiles as a
    perfect match if their median absolute AF difference is small.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size < 2:
        return np.nan

    mad = float(np.nanmedian(np.abs(x - y)))

    if float(np.nanstd(x)) < 1e-6 or float(np.nanstd(y)) < 1e-6:
        if constant_profiles_are_match and mad <= constant_profile_max_mad:
            return 1.0
        return np.nan

    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return np.nan


def _profile_extreme_fraction(vals: np.ndarray, low_af: float, high_af: float) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean((vals <= low_af) | (vals >= high_af)))


def _directions_compatible(direction_a: str, direction_b: str) -> bool:
    """
    LOW and HIGH must match exactly.
    MIXED is compatible with LOW/HIGH/MIXED because mixed-direction LOH can be
    produced by informative SNPs phased on opposite alleles.
    """
    direction_a = str(direction_a)
    direction_b = str(direction_b)

    if "NA" in {direction_a, direction_b}:
        return False

    if direction_a == direction_b:
        return True

    if "MIXED" in {direction_a, direction_b}:
        return True

    return False


def _cross_loh_clone_similarity_check(
    state: "PipelineState",
    cfg: "LOHConfig",
    *,
    LOH_TAG: str,
    arm_variants_fn,
    extract_chr_arms_fn,
    low_af: float,
    high_af: float,
) -> pd.DataFrame:
    """
    End-of-Phase-4 diagnostic.

    Compare LOH clones against other LOH clones over their respective LOH
    blocks. If two clones have highly similar AF profiles on the same arm/block,
    print a yellow manual-merge suggestion.

    This is deliberately diagnostic only:
      • no clone labels are changed
      • no cells are relabelled
      • results are stored in state.results["phase4_cross_loh_similarity"]
    """
    from .loh_config import vprint, fmt_float

    enabled = bool(getattr(cfg, "phase4_cross_loh_similarity_enabled", True))
    if not enabled:
        return pd.DataFrame()

    min_size_cfg = getattr(cfg, "phase4_cross_loh_min_clone_size", None)
    if min_size_cfg is None:
        min_size_cfg = cfg.min_final_clone_size
    min_size = int(min_size_cfg)

    min_shared_variants = int(getattr(cfg, "phase4_cross_loh_min_shared_variants", 4))
    suggest_r = float(getattr(cfg, "phase4_cross_loh_suggest_corr", 0.95))
    suggest_mad = float(getattr(cfg, "phase4_cross_loh_suggest_median_abs_delta", 10.0))
    suggest_extreme = float(getattr(cfg, "phase4_cross_loh_suggest_extreme_frac", 0.85))
    min_block_overlap = float(getattr(cfg, "phase4_cross_loh_min_block_overlap", 0.50))
    require_same_base = bool(getattr(cfg, "phase4_cross_loh_require_same_base", False))
    constant_profiles_are_match = bool(
        getattr(cfg, "phase4_cross_loh_constant_profiles_are_match", True)
    )

    vprint("\n" + "-" * 60, cfg)
    vprint("PHASE 4.4: Cross-LOH Clone Similarity Check", cfg)
    vprint("-" * 60, cfg)
    vprint(
        "  What: compare LOH clones against other LOH clones over their "
        "respective same-arm LOH block variants.",
        cfg,
    )
    vprint(
        "  Yellow warning: suggested manual merge/review if "
        f"r ≥ {suggest_r}, median |ΔAF| ≤ {suggest_mad:g}, "
        f"extreme fraction ≥ {suggest_extreme}, and block overlap ≥ {min_block_overlap:.0%}.",
        cfg,
    )
    if require_same_base:
        vprint("  Same base genotype is required for a merge suggestion.", cfg)
    else:
        vprint("  Same base genotype is not required; differing bases are flagged for manual review.", cfg)

    state.refresh()

    loh_snapshot = [
        (str(lbl), list(cells))
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_loh(lbl) and len(cells) >= min_size
    ]

    vprint(f"  LOH clones tested: {len(loh_snapshot)}", cfg)

    if len(loh_snapshot) < 2:
        vprint("  No cross-LOH comparisons needed.", cfg)
        return pd.DataFrame()

    rows: list[dict] = []

    for i in range(len(loh_snapshot)):
        clone_a, cells_a = loh_snapshot[i]
        arms_a = list(extract_chr_arms_fn(clone_a))
        base_a = _get_base_name(clone_a, LOH_TAG)

        for j in range(i + 1, len(loh_snapshot)):
            clone_b, cells_b = loh_snapshot[j]
            arms_b = list(extract_chr_arms_fn(clone_b))
            base_b = _get_base_name(clone_b, LOH_TAG)

            shared_arms = sorted(set(arms_a).intersection(set(arms_b)))
            if not shared_arms:
                continue

            same_base = base_a == base_b
            if require_same_base and not same_base:
                continue

            for arm in shared_arms:
                block_a = _loh_block_variants_for_label(clone_a, state, arm=arm)
                block_b = _loh_block_variants_for_label(clone_b, state, arm=arm)

                if not block_a:
                    block_a = list(arm_variants_fn(arm, state))
                if not block_b:
                    block_b = list(arm_variants_fn(arm, state))

                block_a = [str(v) for v in dict.fromkeys(block_a) if str(v) in state.df.columns]
                block_b = [str(v) for v in dict.fromkeys(block_b) if str(v) in state.df.columns]

                if len(block_a) < min_shared_variants or len(block_b) < min_shared_variants:
                    continue

                block_a_set = set(block_a)
                block_b_set = set(block_b)
                shared_vars = [v for v in block_a if v in block_b_set]

                denom = max(1, min(len(block_a_set), len(block_b_set)))
                block_overlap = len(set(shared_vars)) / denom

                if len(shared_vars) < min_shared_variants:
                    continue

                bcs_a = [str(bc) for bc in cells_a if str(bc) in state.df.index]
                bcs_b = [str(bc) for bc in cells_b if str(bc) in state.df.index]

                if len(bcs_a) < min_size or len(bcs_b) < min_size:
                    continue

                profile_a = (
                    state.df.loc[bcs_a, shared_vars]
                    .apply(pd.to_numeric, errors="coerce")
                    .mean(axis=0)
                    .values
                    .astype(float)
                )
                profile_b = (
                    state.df.loc[bcs_b, shared_vars]
                    .apply(pd.to_numeric, errors="coerce")
                    .mean(axis=0)
                    .values
                    .astype(float)
                )

                finite = np.isfinite(profile_a) & np.isfinite(profile_b)
                if int(finite.sum()) < min_shared_variants:
                    continue

                x = profile_a[finite]
                y = profile_b[finite]
                used_vars = [v for v, ok in zip(shared_vars, finite) if bool(ok)]

                median_abs_delta = float(np.nanmedian(np.abs(x - y)))
                r = _safe_similarity_corr(
                    x,
                    y,
                    constant_profile_max_mad=suggest_mad,
                    constant_profiles_are_match=constant_profiles_are_match,
                )

                extreme_a = _profile_extreme_fraction(x, low_af, high_af)
                extreme_b = _profile_extreme_fraction(y, low_af, high_af)
                mean_extreme = float(np.nanmean([extreme_a, extreme_b]))

                direction_a = _direction(x, low_af, high_af)
                direction_b = _direction(y, low_af, high_af)
                direction_ok = _directions_compatible(direction_a, direction_b)

                suggest_merge = (
                    np.isfinite(r)
                    and r >= suggest_r
                    and median_abs_delta <= suggest_mad
                    and mean_extreme >= suggest_extreme
                    and block_overlap >= min_block_overlap
                    and direction_ok
                )

                row = {
                    "clone_a": clone_a,
                    "clone_b": clone_b,
                    "base_a": base_a,
                    "base_b": base_b,
                    "same_base": bool(same_base),
                    "arm": arm,
                    "n_a": int(len(bcs_a)),
                    "n_b": int(len(bcs_b)),
                    "n_block_a_variants": int(len(block_a)),
                    "n_block_b_variants": int(len(block_b)),
                    "n_shared_block_variants": int(len(used_vars)),
                    "block_overlap": float(block_overlap),
                    "r": float(r) if np.isfinite(r) else np.nan,
                    "median_abs_delta": float(median_abs_delta),
                    "extreme_frac_a": float(extreme_a) if np.isfinite(extreme_a) else np.nan,
                    "extreme_frac_b": float(extreme_b) if np.isfinite(extreme_b) else np.nan,
                    "mean_extreme_frac": float(mean_extreme) if np.isfinite(mean_extreme) else np.nan,
                    "direction_a": direction_a,
                    "direction_b": direction_b,
                    "direction_compatible": bool(direction_ok),
                    "suggest_merge": bool(suggest_merge),
                }
                rows.append(row)

                comparison_msg = (
                    f"  {arm}: {clone_a}  <->  {clone_b}\n"
                    f"    n={len(bcs_a)} vs {len(bcs_b)}, "
                    f"shared_block_vars={len(used_vars)}, "
                    f"block_overlap={block_overlap:.0%}, "
                    f"r={fmt_float(r)}, "
                    f"median |ΔAF|={median_abs_delta:.1f}, "
                    f"extreme={mean_extreme:.3f}, "
                    f"dir={direction_a}/{direction_b}, "
                    f"same_base={same_base}"
                )

                if suggest_merge:
                    caution = ""
                    if not same_base:
                        caution = (
                            "\n    Note: base genotype labels differ; review genotype-defining "
                            "somatic variants before manually merging."
                        )
                    vprint(
                        f"{_YELLOW}{comparison_msg}\n"
                        "    ⚠ SUGGESTED MANUAL MERGE/REVIEW: same-arm LOH block "
                        f"profiles are highly similar.{caution}{_RESET}",
                        cfg,
                    )
                else:
                    vprint(comparison_msg + "\n    no merge suggestion", cfg)

    out = pd.DataFrame(rows)

    if out.empty:
        vprint("  No eligible same-arm LOH clone pairs found.", cfg)
    else:
        n_suggest = int(out["suggest_merge"].sum())
        if n_suggest > 0:
            vprint(f"{_YELLOW}  Cross-LOH merge suggestions: {n_suggest} pair(s).{_RESET}", cfg)
        else:
            vprint("  Cross-LOH merge suggestions: 0 pair(s).", cfg)

    try:
        state.results["phase4_cross_loh_similarity"] = out
    except Exception:
        pass

    return out


def _merge_small_loh_clones_after_annotation(
    state: "PipelineState",
    cfg: "LOHConfig",
    *,
    LOH_TAG: str,
    arm_variants_fn,
    extract_chr_arms_fn,
    low_af: float,
    high_af: float,
) -> pd.DataFrame:
    """
    End-of-Phase-4 cell-level cleanup.

    This pass is deliberately *cell-level*, not clone-level.  It handles three
    biologically useful reassignment directions after final LOH annotation:

      1. LOH cell      -> larger compatible LOH clone
      2. LOH cell      -> same-base non-LOH parent clone
      3. non-LOH cell  -> same-base LOH clone

    Direction (3) is important for cases where a subset of cells in the base
    clone visibly carries the same AF pattern as an LOH clone, but was not
    captured by earlier LOH rescue/annotation.

    A cell is relabelled only to the largest passing compatible target.  The
    diagnostic table is written to state.results["phase4_small_loh_merge"].
    Each row is one source-cell/target comparison.
    """
    from .loh_config import vprint, fmt_float, is_fp_label

    enabled = bool(getattr(cfg, "phase4_merge_small_loh_after_annotation_enabled", True))
    if not enabled:
        return pd.DataFrame()

    promote_fp_to_loh = bool(getattr(cfg, "phase4_promote_fp_to_loh", False))
    promote_fp_min_source_cells = int(
        getattr(cfg, "phase4_promote_fp_min_source_cells", 1)
    )
    promote_fp_require_exact_arm_set = bool(
        getattr(cfg, "phase4_promote_fp_require_exact_arm_set", True)
    )

    max_source_cfg = getattr(cfg, "phase4_merge_small_loh_max_source_cells", None)
    max_source_cells = None if max_source_cfg is None else int(max_source_cfg)

    min_target_cfg = getattr(cfg, "phase4_merge_small_loh_min_target_cells", None)
    if min_target_cfg is None:
        min_target_cfg = cfg.min_final_clone_size
    min_target_cells = int(min_target_cfg)

    include_base_non_loh_target = bool(
        getattr(cfg, "phase4_merge_small_loh_include_base_non_loh_target", True)
    )
    include_base_non_loh_sources = bool(
        getattr(cfg, "phase4_merge_small_loh_include_base_non_loh_sources", True)
    )
    base_source_min_cells_cfg = getattr(
        cfg, "phase4_merge_small_loh_base_source_min_cells", None
    )
    if base_source_min_cells_cfg is None:
        base_source_min_cells_cfg = cfg.min_final_clone_size
    base_source_min_cells = int(base_source_min_cells_cfg)

    require_same_base = bool(getattr(cfg, "phase4_merge_small_loh_require_same_base", True))
    require_source_arms_subset = bool(
        getattr(cfg, "phase4_merge_small_loh_require_source_arms_subset", True)
    )
    allow_equal_arm_set_to_larger_target = bool(
        getattr(
            cfg,
            "phase4_merge_small_loh_allow_equal_arm_set_to_larger_target",
            False,
        )
    )
    compare_all_target_arms = bool(
        getattr(cfg, "phase4_merge_small_loh_compare_all_target_arms", True)
    )
    loh_to_loh_require_larger_target = bool(
        getattr(cfg, "phase4_merge_small_loh_loh_to_loh_require_larger_target", True)
    )
    loh_to_loh_require_arm_overlap = bool(
        getattr(cfg, "phase4_merge_small_loh_loh_to_loh_require_arm_overlap", True)
    )
    min_shared_variants = int(getattr(cfg, "phase4_merge_small_loh_min_shared_variants", 4))
    min_r = float(getattr(cfg, "phase4_merge_small_loh_min_corr", 0.95))
    max_mad = float(getattr(cfg, "phase4_merge_small_loh_max_median_abs_delta", 10.0))
    min_extreme = float(getattr(cfg, "phase4_merge_small_loh_min_extreme_frac", 0.85))
    min_block_overlap = float(getattr(cfg, "phase4_merge_small_loh_min_block_overlap", 0.50))
    ambiguous_delta = float(getattr(cfg, "phase4_merge_small_loh_ambiguous_corr_delta", 0.02))
    constant_profiles_are_match = bool(
        getattr(cfg, "phase4_merge_small_loh_constant_profiles_are_match", True)
    )

    # For LOH -> base non-LOH parent targets, similarity to the parent AF profile
    # is the criterion.  We intentionally do not require LOH-like extreme fraction
    # for the parent target, because a true parent-like cell should be mid/het over
    # the source LOH block.  MAD + correlation remain strict.
    base_target_require_low_extreme = bool(
        getattr(cfg, "phase4_merge_small_loh_base_target_require_low_extreme", False)
    )
    base_target_max_cell_extreme = float(
        getattr(cfg, "phase4_merge_small_loh_base_target_max_cell_extreme_frac", 0.50)
    )
    base_target_max_target_extreme = float(
        getattr(cfg, "phase4_merge_small_loh_base_target_max_target_extreme_frac", 0.50)
    )

    # Base non-LOH -> LOH promotion uses separate thresholds.  These are
    # intentionally less strict than LOH->LOH clone-profile cleanup because a
    # single cell compared to a clone mean can have lower Pearson r even when
    # the LOH block is visually and biologically consistent.  The guardrail is
    # explicit LOH-like support in the cell over the target block.
    base_to_loh_min_r = float(getattr(cfg, "phase4_merge_base_to_loh_min_corr", 0.80))
    base_to_loh_max_mad = float(getattr(cfg, "phase4_merge_base_to_loh_max_median_abs_delta", 15.0))
    base_to_loh_min_cell_extreme = float(getattr(cfg, "phase4_merge_base_to_loh_min_cell_extreme_frac", 0.70))
    base_to_loh_min_target_extreme = float(getattr(cfg, "phase4_merge_base_to_loh_min_target_extreme_frac", 0.85))
    base_to_loh_require_direction = bool(getattr(cfg, "phase4_merge_base_to_loh_require_direction_match", True))
    base_to_loh_min_matching_arms_frac = float(getattr(cfg, "phase4_merge_base_to_loh_min_matching_arms_frac", 1.00))
    base_to_loh_allow_het_to_hem = bool(getattr(cfg, "phase4_merge_base_to_loh_allow_het_to_hem", True))

    vprint("\n" + "-" * 60, cfg)
    vprint("PHASE 4.5: Cell-level merge into largest compatible target", cfg)
    vprint("-" * 60, cfg)
    vprint(
        "  Rule: each cell is scored independently against compatible targets. "
        "LOH cells can move to larger LOH clones or the same-base parent; "
        "same-base non-LOH cells can move into the largest matching LOH clone.",
        cfg,
    )
    vprint(
        f"  Per-cell thresholds for LOH targets: r ≥ {min_r}, median |ΔAF| ≤ {max_mad:g}, "
        f"extreme fraction ≥ {min_extreme}, block overlap ≥ {min_block_overlap:.0%}.",
        cfg,
    )
    vprint(
        f"  Per-cell thresholds for base non-LOH target: r ≥ {min_r}, median |ΔAF| ≤ {max_mad:g}; "
        f"LOH-like extreme fraction is not required"
        + (
            f", and cell/target extreme must be ≤ {base_target_max_cell_extreme:g}/"
            f"{base_target_max_target_extreme:g}"
            if base_target_require_low_extreme else ""
        )
        + ".",
        cfg,
    )
    vprint(
        f"  Per-cell thresholds for base non-LOH → LOH promotion: "
        f"r ≥ {base_to_loh_min_r}, median |ΔAF| ≤ {base_to_loh_max_mad:g}, "
        f"cell extreme ≥ {base_to_loh_min_cell_extreme}, "
        f"target extreme ≥ {base_to_loh_min_target_extreme}, "
        f"matching arms ≥ {base_to_loh_min_matching_arms_frac:.0%}; "
        f"direction_match={base_to_loh_require_direction}; "
        f"Het→Hem/Hom base-match={base_to_loh_allow_het_to_hem}.",
        cfg,
    )
    if promote_fp_to_loh:
        vprint(
            f"  Possible-LOH (FP) rescue: ON (min source n={promote_fp_min_source_cells}; "
            f"require_exact_arm_set={promote_fp_require_exact_arm_set}) — "
            f"FP cells scored on the base→LOH path into same-base LOH clones; "
            f"non-matching cells stay FP.",
            cfg,
        )
    source_size_txt = "unrestricted" if max_source_cells is None else f"≤ {max_source_cells}"
    vprint(
        f"  LOH source clone size: {source_size_txt}; base non-LOH sources enabled="
        f"{include_base_non_loh_sources} (min n={base_source_min_cells}); "
        f"target size ≥ {min_target_cells}; merge target policy=largest passing compatible; "
        f"same_base={require_same_base}; include_base_non_loh_target={include_base_non_loh_target}; "
        f"compare_all_target_arms={compare_all_target_arms}",
        cfg,
    )
    if loh_to_loh_require_larger_target:
        policy = "  LOH→LOH policy: only a less-complex clone into a larger strict-superset-arm clone."
        if allow_equal_arm_set_to_larger_target:
            policy = (
                "  LOH→LOH policy: only a smaller clone into a larger same-base "
                "target; source arm set must be a strict subset or an exact same-arm match."
            )
        vprint(policy, cfg)
    else:
        vprint(
            "  LOH→LOH policy: outlier reassignment ON — any same-base sibling "
            f"sharing ≥1 arm (arm_overlap={loh_to_loh_require_arm_overlap}), "
            "size-independent; per-cell similarity gates still enforced.",
            cfg,
        )

    state.refresh()

    # Phase 3b can explicitly recover a biologically meaningful shared-arm
    # parent plus a child-only-arm subclone. Preserve this lineage so the
    # generic Phase 4d larger-superset merge does not immediately collapse the
    # parent back into its child.
    preserve_nested_lineages = bool(
        getattr(cfg, "phase4_preserve_nested_loh_lineages", True)
    )
    protected_nested_pairs = {
        (
            str(record.get("parent_label")),
            str(record.get("child_label")),
        )
        for record in (getattr(state, "nested_loh_lineages", []) or [])
        if isinstance(record, dict)
        and record.get("parent_label")
        and record.get("child_label")
    }

    if preserve_nested_lineages and protected_nested_pairs:
        vprint(
            "  Preserving Phase 3b nested LOH parent-child lineage(s): "
            + "; ".join(
                f"{parent} → {child}"
                for parent, child in sorted(protected_nested_pairs)
            ),
            cfg,
        )

    rows: list[dict] = []
    pending_moves: dict[str, str] = {}
    move_records: list[dict] = []

    def _cells_in_df(cells: list) -> list[str]:
        return [str(c) for c in cells if str(c) in state.df_numeric.index]

    def _target_profile_for_cells(cells: list, variants: list[str]) -> np.ndarray | None:
        cells = _cells_in_df(cells)
        variants = [str(v) for v in variants if str(v) in state.df_numeric.columns]
        if not cells or not variants:
            return None
        arr = (
            state.df_numeric.loc[cells, variants]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=0)
            .values
            .astype(float)
        )
        return arr if np.any(np.isfinite(arr)) else None

    def _cell_profile(cell_id: str, variants: list[str]) -> np.ndarray | None:
        cell_id = str(cell_id)
        variants = [str(v) for v in variants if str(v) in state.df_numeric.columns]
        if cell_id not in state.df_numeric.index or not variants:
            return None
        arr = (
            pd.to_numeric(state.df_numeric.loc[cell_id, variants], errors="coerce")
            .values
            .astype(float)
        )
        return arr if np.any(np.isfinite(arr)) else None

    def _arm_blocks_for_pair(
        source_label: str,
        target_label: str,
        arm: str,
        *,
        target_type: str,
        source_type: str,
    ) -> tuple[list[str], list[str], list[str], float]:
        """
        Return source block, target block, shared variants and overlap.

        For base_non_loh sources there is no source LOH block, so we compare
        the cell over the target LOH block.  For base_non_loh targets there is
        no target LOH block, so we compare over the source LOH block.
        """
        if source_type in ("base_non_loh", "possible_loh"):
            source_block = _loh_block_variants_for_label(target_label, state, arm=arm)
        else:
            source_block = _loh_block_variants_for_label(source_label, state, arm=arm)

        if target_type == "loh":
            target_block = _loh_block_variants_for_label(target_label, state, arm=arm)
        else:
            target_block = source_block

        if not source_block:
            source_block = list(arm_variants_fn(arm, state))
        if not target_block:
            target_block = list(arm_variants_fn(arm, state))

        source_block = [
            str(v) for v in dict.fromkeys(source_block)
            if str(v) in state.df_numeric.columns
        ]
        target_block = [
            str(v) for v in dict.fromkeys(target_block)
            if str(v) in state.df_numeric.columns
        ]

        target_set = set(target_block)
        shared = [v for v in source_block if v in target_set]
        denom = max(1, min(len(set(source_block)), len(set(target_block))))
        overlap = len(set(shared)) / denom
        return source_block, target_block, shared, float(overlap)

    def _score_cell_to_target(
        *,
        cell_id: str,
        source_label: str,
        source_n: int,
        source_type: str,
        target_label: str,
        target_cells: list,
        target_n: int,
        target_type: str,
        arms_to_check: list[str],
        source_base: str,
        target_base: str,
        source_arms: list[str],
        target_arms: list[str],
        base_match_type: str = "exact",
    ) -> dict:
        arm_metrics: list[dict] = []
        pair_ok = True
        fail_reason = ""

        for arm in arms_to_check:
            source_block, target_block, shared_vars, block_overlap = _arm_blocks_for_pair(
                source_label,
                target_label,
                arm,
                target_type=target_type,
                source_type=source_type,
            )

            if len(shared_vars) < min_shared_variants:
                pair_ok = False
                fail_reason = f"{arm}:shared_variants<{min_shared_variants}"
                break
            if block_overlap < min_block_overlap:
                pair_ok = False
                fail_reason = f"{arm}:block_overlap<{min_block_overlap:.2f}"
                break

            cell_profile = _cell_profile(cell_id, shared_vars)
            target_profile = _target_profile_for_cells(target_cells, shared_vars)
            if cell_profile is None or target_profile is None:
                pair_ok = False
                fail_reason = f"{arm}:missing_profile"
                break

            finite = np.isfinite(cell_profile) & np.isfinite(target_profile)
            n_callable = int(finite.sum())
            if n_callable < min_shared_variants:
                pair_ok = False
                fail_reason = (
                    f"{arm}:insufficient_callable_variants("
                    f"n={n_callable}<{min_shared_variants})"
                )
                arm_metrics.append({
                    "arm": arm,
                    "r": float("nan"),
                    "median_abs_delta": float("nan"),
                    "mean_extreme": float("nan"),
                    "cell_extreme": float("nan"),
                    "target_extreme": float("nan"),
                    "block_overlap": float(block_overlap),
                    "n_shared_variants": int(len(shared_vars)),
                    "n_callable_variants": n_callable,
                    "direction_cell": "NA",
                    "direction_target": "NA",
                })
                break

            x = cell_profile[finite]
            y = target_profile[finite]
            r = _safe_similarity_corr(
                x,
                y,
                constant_profile_max_mad=max_mad,
                constant_profiles_are_match=constant_profiles_are_match,
            )
            median_abs_delta = float(np.nanmedian(np.abs(x - y)))
            extreme_cell = _profile_extreme_fraction(x, low_af, high_af)
            extreme_target = _profile_extreme_fraction(y, low_af, high_af)
            mean_extreme = float(np.nanmean([extreme_cell, extreme_target]))
            direction_cell = _direction(x, low_af, high_af)
            direction_target = _direction(y, low_af, high_af)

            if target_type == "base_non_loh":
                # LOH -> parent/base target: high similarity and low delta are
                # enough.  Do not require LOH directional/extreme support.
                direction_ok = True
                arm_ok = (
                    np.isfinite(r)
                    and r >= min_r
                    and median_abs_delta <= max_mad
                )
                if base_target_require_low_extreme:
                    arm_ok = (
                        arm_ok
                        and np.isfinite(extreme_cell)
                        and np.isfinite(extreme_target)
                        and extreme_cell <= base_target_max_cell_extreme
                        and extreme_target <= base_target_max_target_extreme
                    )
            else:
                # Any LOH target.  Base_non_loh -> LOH promotion is scored with
                # its own thresholds because single-cell-to-clone Pearson is
                # noisier than clone-to-clone similarity.  The key guardrail is
                # explicit LOH-like support in the source cell over the target
                # LOH block.
                if source_type in ("base_non_loh", "possible_loh"):
                    direction_ok = (
                        _directions_compatible(direction_cell, direction_target)
                        if base_to_loh_require_direction else True
                    )
                    arm_ok = (
                        np.isfinite(r)
                        and r >= base_to_loh_min_r
                        and median_abs_delta <= base_to_loh_max_mad
                        and np.isfinite(extreme_cell)
                        and extreme_cell >= base_to_loh_min_cell_extreme
                        and np.isfinite(extreme_target)
                        and extreme_target >= base_to_loh_min_target_extreme
                        and direction_ok
                    )
                else:
                    direction_ok = _directions_compatible(direction_cell, direction_target)
                    arm_ok = (
                        np.isfinite(r)
                        and r >= min_r
                        and median_abs_delta <= max_mad
                        and mean_extreme >= min_extreme
                        and direction_ok
                    )

            if not arm_ok:
                pair_ok = False
                bits = []
                if target_type == "loh" and source_type in ("base_non_loh", "possible_loh"):
                    if not np.isfinite(r):
                        bits.append("noninformative_profile_correlation")
                    elif r < base_to_loh_min_r:
                        bits.append(f"r<{base_to_loh_min_r}")
                    if median_abs_delta > base_to_loh_max_mad:
                        bits.append(f"mad>{base_to_loh_max_mad:g}")
                    if not np.isfinite(extreme_cell) or extreme_cell < base_to_loh_min_cell_extreme:
                        bits.append(f"cell_extreme<{base_to_loh_min_cell_extreme}")
                    if not np.isfinite(extreme_target) or extreme_target < base_to_loh_min_target_extreme:
                        bits.append(f"target_extreme<{base_to_loh_min_target_extreme}")
                    if not direction_ok:
                        bits.append("direction_mismatch")
                else:
                    if not np.isfinite(r):
                        bits.append("noninformative_profile_correlation")
                    elif r < min_r:
                        bits.append(f"r<{min_r}")
                    if median_abs_delta > max_mad:
                        bits.append(f"mad>{max_mad:g}")
                    if target_type == "loh":
                        if not np.isfinite(mean_extreme) or mean_extreme < min_extreme:
                            bits.append(f"extreme<{min_extreme}")
                        if not direction_ok:
                            bits.append("direction_mismatch")
                    elif base_target_require_low_extreme:
                        if not np.isfinite(extreme_cell) or extreme_cell > base_target_max_cell_extreme:
                            bits.append(f"cell_extreme>{base_target_max_cell_extreme:g}")
                        if not np.isfinite(extreme_target) or extreme_target > base_target_max_target_extreme:
                            bits.append(f"target_extreme>{base_target_max_target_extreme:g}")
                fail_reason = f"{arm}:" + ";".join(bits)
                break

            arm_metrics.append({
                "arm": arm,
                "r": float(r),
                "median_abs_delta": median_abs_delta,
                "mean_extreme": mean_extreme,
                "cell_extreme": extreme_cell,
                "target_extreme": extreme_target,
                "block_overlap": float(block_overlap),
                "n_shared_variants": int(len(shared_vars)),
                "direction_cell": direction_cell,
                "direction_target": direction_target,
            })

        if arm_metrics:
            mean_r = float(np.nanmean([m["r"] for m in arm_metrics]))
            max_delta = float(np.nanmax([m["median_abs_delta"] for m in arm_metrics]))
            min_ext = float(np.nanmin([m["mean_extreme"] for m in arm_metrics]))
            max_cell_ext = float(np.nanmax([m["cell_extreme"] for m in arm_metrics]))
            max_target_ext = float(np.nanmax([m["target_extreme"] for m in arm_metrics]))
            min_overlap_seen = float(np.nanmin([m["block_overlap"] for m in arm_metrics]))
            total_shared = int(np.sum([m["n_shared_variants"] for m in arm_metrics]))
        else:
            mean_r = float("nan")
            max_delta = float("nan")
            min_ext = float("nan")
            max_cell_ext = float("nan")
            max_target_ext = float("nan")
            min_overlap_seen = float("nan")
            total_shared = 0

        bc = state.cell_to_barcode.get(str(cell_id), str(cell_id))
        return {
            "source_cell": str(cell_id),
            "source_barcode": str(bc),
            "source_clone": source_label,
            "source_type": source_type,
            "target_clone": target_label,
            "target_type": target_type,
            "source_base": source_base,
            "target_base": target_base,
            "same_base": bool(source_base == target_base),
            "base_match": bool(base_match_type != "different_base"),
            "base_match_type": str(base_match_type),
            "zygosity_normalized_base_match": bool(str(base_match_type).startswith("zygosity_normalized")),
            "source_n": int(source_n),
            "target_n": int(target_n),
            "source_arms": ",".join(source_arms),
            "target_arms": ",".join(target_arms),
            "checked_arms": ",".join(arms_to_check),
            "mean_r": mean_r,
            "max_median_abs_delta": max_delta,
            "min_mean_extreme_frac": min_ext,
            "max_cell_extreme_frac": max_cell_ext,
            "max_target_extreme_frac": max_target_ext,
            "min_block_overlap": min_overlap_seen,
            "total_shared_variants": total_shared,
            "passes": bool(pair_ok),
            "merged": False,
            "fail_reason": fail_reason,
        }

    def _compatible_loh_target_specs_for_source(
        *,
        source_label: str,
        source_n: int,
        source_base: str,
        source_arms: list[str],
        source_type: str,
    ) -> list[dict]:
        target_specs: list[dict] = []

        for target_label, target_cells in [
            (str(lbl), list(cells))
            for lbl, cells in state.clone_to_cells.items()
            if _is_active_loh(lbl) and str(lbl) != str(source_label) and len(cells) > 0
        ]:
            if (
                source_type == "loh"
                and preserve_nested_lineages
                and (str(source_label), str(target_label)) in protected_nested_pairs
            ):
                continue

            target_n = len(target_cells)
            if target_n < min_target_cells:
                continue

            # LOH->LOH cleanup moves only into a larger LOH clone, unless
            # outlier reassignment is enabled (larger-target not required).
            # Base->LOH promotion is always allowed regardless of size.
            if (
                source_type == "loh"
                and loh_to_loh_require_larger_target
                and target_n <= source_n
            ):
                continue

            target_base = _get_base_name(target_label, LOH_TAG)
            allow_zyg_match = bool(
                base_to_loh_allow_het_to_hem
                and source_type in ("base_non_loh", "possible_loh")
            )
            base_ok, base_match_type = _base_names_compatible(
                source_base,
                target_base,
                loh_tag=LOH_TAG,
                allow_het_to_hem=allow_zyg_match,
            )
            if require_same_base and not base_ok:
                continue

            target_arms = sorted(set(extract_chr_arms_fn(target_label)))
            if not target_arms:
                continue

            source_arm_set = set(source_arms)
            target_arm_set = set(target_arms)
            if source_type == "possible_loh":
                # A generic Phase 4 rescue must not silently discard an FP
                # arm (e.g. chr7q+chr5q -> chr5q). Arm-loss/addition is
                # resolved explicitly in Phase 3a/3b, where per-arm cell
                # evidence is available.
                if not source_arm_set:
                    continue
                if promote_fp_require_exact_arm_set:
                    if source_arm_set != target_arm_set:
                        continue
                elif not source_arm_set.issubset(target_arm_set):
                    continue

            if source_type == "loh":
                # Require a proper source-arm subset: only a less-complex
                # LOH clone can merge into a more-complex LOH clone. Outlier mode drops this rule and
                # instead require the sibling to share at least one arm, so a
                # cell mis-binned across sibling clones (e.g. chr5p vs chr5q,
                # both carrying chr7p) can move to the clone it actually fits.
                outlier_mode = not loh_to_loh_require_larger_target
                if require_source_arms_subset and not outlier_mode:
                    proper_subset = source_arm_set < target_arm_set
                    equal_arm_set_allowed = (
                        allow_equal_arm_set_to_larger_target
                        and source_arm_set == target_arm_set
                    )
                    if not (proper_subset or equal_arm_set_allowed):
                        continue
                elif loh_to_loh_require_arm_overlap:
                    if not (source_arm_set & target_arm_set):
                        continue

            if source_type in ("base_non_loh", "possible_loh"):
                # No source LOH block to anchor on; check the target LOH arms.
                arms_to_check = target_arms
            else:
                arms_to_check = (
                    target_arms
                    if compare_all_target_arms
                    else sorted(source_arm_set.intersection(target_arm_set))
                )
            if not arms_to_check:
                continue

            target_specs.append({
                "target_label": target_label,
                "target_cells": target_cells,
                "target_n": target_n,
                "target_base": target_base,
                "base_match_type": base_match_type,
                "target_type": "loh",
                "target_arms": target_arms,
                "arms_to_check": arms_to_check,
            })

        return target_specs

    source_specs: list[dict] = []

    # A) LOH-labelled source cells: can move to larger LOH target or base parent.
    for source_label, source_cells in sorted(
        [
            (str(lbl), list(cells))
            for lbl, cells in state.clone_to_cells.items()
            if _is_active_loh(lbl) and len(cells) > 0
        ],
        key=lambda x: (len(x[1]), x[0]),
    ):
        source_n = len(source_cells)
        if max_source_cells is not None and source_n > max_source_cells:
            continue
        source_base = _get_base_name(source_label, LOH_TAG)
        source_arms = sorted(set(extract_chr_arms_fn(source_label)))
        if not source_arms:
            continue
        source_specs.append({
            "source_label": source_label,
            "source_cells": source_cells,
            "source_n": source_n,
            "source_type": "loh",
            "source_base": source_base,
            "source_arms": source_arms,
        })

    # B) Same-base non-LOH source cells: can be promoted into the largest
    # matching LOH clone.  This is the fix for base cells that visibly match
    # a retained LOH heatmap pattern.
    if include_base_non_loh_sources:
        for source_label, source_cells in sorted(
            [
                (str(lbl), list(cells))
                for lbl, cells in state.clone_to_cells.items()
                if _is_active_non_loh(lbl) and len(cells) >= base_source_min_cells
            ],
            key=lambda x: (-len(x[1]), x[0]),
        ):
            # Only include non-LOH sources that actually have a same-base LOH clone.
            has_same_base_loh = any(
                _is_active_loh(lbl)
                and _base_names_compatible(
                    str(source_label),
                    _get_base_name(str(lbl), LOH_TAG),
                    loh_tag=LOH_TAG,
                    allow_het_to_hem=base_to_loh_allow_het_to_hem,
                )[0]
                for lbl in state.clone_to_cells.keys()
            )
            if not has_same_base_loh:
                continue
            source_specs.append({
                "source_label": source_label,
                "source_cells": source_cells,
                "source_n": len(source_cells),
                "source_type": "base_non_loh",
                "source_base": str(source_label),
                "source_arms": [],
            })

    # C) Possible-LOH (FP) source cells: opt-in rescue of FP cells that
    # individually match a validated same-base LOH clone. Scored on the same
    # base->LOH per-cell path; non-matching cells stay FP.
    if promote_fp_to_loh:
        from .loh_config import fp_base_name
        for source_label, source_cells in sorted(
            [
                (str(lbl), list(cells))
                for lbl, cells in state.clone_to_cells.items()
                if is_fp_label(lbl) and len(cells) >= promote_fp_min_source_cells
            ],
            key=lambda x: (-len(x[1]), x[0]),
        ):
            fp_base = fp_base_name(source_label)
            has_same_base_loh = any(
                _is_active_loh(lbl)
                and _base_names_compatible(
                    fp_base,
                    _get_base_name(str(lbl), LOH_TAG),
                    loh_tag=LOH_TAG,
                    allow_het_to_hem=base_to_loh_allow_het_to_hem,
                )[0]
                for lbl in state.clone_to_cells.keys()
            )
            if not has_same_base_loh:
                continue
            source_arms = sorted(set(extract_chr_arms_fn(source_label)))
            if not source_arms:
                # Without a parseable source arm-set, retain the FP clone.
                # The generic rescue route cannot prove that no event is lost.
                continue
            source_specs.append({
                "source_label": source_label,
                "source_cells": source_cells,
                "source_n": len(source_cells),
                "source_type": "possible_loh",
                "source_base": fp_base,
                "source_arms": source_arms,
            })

    if not source_specs:
        vprint("  No eligible cell-level source clones found.", cfg)
        return pd.DataFrame()

    for spec_src in source_specs:
        state.refresh()
        source_label = str(spec_src["source_label"])
        source_type = str(spec_src["source_type"])
        source_cells = [
            c for c in state.clone_to_cells.get(source_label, [])
            if str(c) in state.df_numeric.index
        ]
        source_n = len(source_cells)
        if source_n == 0:
            continue
        if source_type == "loh" and max_source_cells is not None and source_n > max_source_cells:
            continue

        source_base = str(spec_src["source_base"])
        source_arms = sorted(set(spec_src.get("source_arms", [])))

        target_specs = _compatible_loh_target_specs_for_source(
            source_label=source_label,
            source_n=source_n,
            source_base=source_base,
            source_arms=source_arms,
            source_type=source_type,
        )

        # LOH source cells can also revert to same-base non-LOH parent.
        if source_type == "loh" and include_base_non_loh_target:
            base_cells = list(state.clone_to_cells.get(source_base, []))
            base_n = len(base_cells)
            if base_n >= min_target_cells and base_n > source_n:
                target_specs.append({
                    "target_label": source_base,
                    "target_cells": base_cells,
                    "target_n": base_n,
                    "target_base": source_base,
                    "base_match_type": "exact",
                    "target_type": "base_non_loh",
                    "target_arms": source_arms,
                    "arms_to_check": source_arms,
                })

        if not target_specs:
            continue

        for cell_id in source_cells:
            if str(cell_id) in pending_moves:
                continue

            candidate_rows: list[dict] = []
            for spec in target_specs:
                row = _score_cell_to_target(
                    cell_id=str(cell_id),
                    source_label=source_label,
                    source_n=source_n,
                    source_type=source_type,
                    target_label=spec["target_label"],
                    target_cells=spec["target_cells"],
                    target_n=spec["target_n"],
                    target_type=spec["target_type"],
                    arms_to_check=spec["arms_to_check"],
                    source_base=source_base,
                    target_base=spec["target_base"],
                    source_arms=source_arms,
                    target_arms=spec["target_arms"],
                    base_match_type=spec.get("base_match_type", "exact"),
                )
                candidate_rows.append(row)

            passing = [r for r in candidate_rows if r.get("passes", False)]
            largest_passing_n = max([int(r["target_n"]) for r in passing], default=-1)
            for rrow in candidate_rows:
                rrow["largest_passing_target_n"] = int(largest_passing_n)
                rrow["is_largest_passing_target"] = (
                    int(rrow.get("target_n", -1)) == int(largest_passing_n)
                    and bool(rrow.get("passes", False))
                )

            rows.extend(candidate_rows)
            if not passing:
                continue

            # Relabel to the largest passing target. If same-sized targets pass,
            # choose by similarity unless the result is ambiguous.
            largest_passing = [
                r for r in passing
                if int(r.get("target_n", -1)) == int(largest_passing_n)
            ]
            largest_passing = sorted(
                largest_passing,
                key=lambda r: (
                    -float(r["mean_r"]),
                    float(r["max_median_abs_delta"]),
                    0 if str(r.get("target_type")) == "base_non_loh" else 1,
                    str(r["target_clone"]),
                ),
            )
            best = largest_passing[0]

            if len(largest_passing) > 1:
                second = largest_passing[1]
                if abs(float(best["mean_r"]) - float(second["mean_r"])) <= ambiguous_delta:
                    vprint(
                        f"  {_YELLOW}skip ambiguous cell merge: {best['source_barcode']} "
                        f"from {source_label} has multiple same-size similar targets "
                        f"(top r={fmt_float(best['mean_r'])}, second r={fmt_float(second['mean_r'])}).{_RESET}",
                        cfg,
                    )
                    continue

            if str(best["target_clone"]) == str(source_label):
                continue

            pending_moves[str(cell_id)] = str(best["target_clone"])
            move_records.append(best)

            best_key = (
                str(best["source_cell"]),
                str(best["source_clone"]),
                str(best["target_clone"]),
                str(best.get("target_type", "")),
            )
            for rrow in reversed(rows):
                rkey = (
                    str(rrow.get("source_cell")),
                    str(rrow.get("source_clone")),
                    str(rrow.get("target_clone")),
                    str(rrow.get("target_type", "")),
                )
                if rkey == best_key and rrow.get("passes", False):
                    rrow["merged"] = True
                    break

    if pending_moves:
        current = state.current_clones.copy()
        clone_arr = current.copy()
        moved_by_target: dict[str, int] = {}

        for cell_id, target_label in pending_moves.items():
            bc = state.cell_to_barcode.get(cell_id, str(cell_id))
            idx = state.sample_idx_for_barcode(str(bc))
            if idx is None:
                continue
            clone_arr.iloc[idx] = target_label
            moved_by_target[target_label] = moved_by_target.get(target_label, 0) + 1

        state.set_clone_labels(clone_arr)

        for target_label, n_moved in sorted(moved_by_target.items(), key=lambda x: str(x[0])):
            target_type = "base non-LOH clone" if _is_active_non_loh(target_label) else "LOH clone"
            vprint(
                f"  {_GREEN}MERGE cells into largest passing compatible {target_type}: "
                f"{n_moved} cell(s) → {target_label}{_RESET}",
                cfg,
            )

        if len(move_records) <= 25:
            for rec in move_records:
                vprint(
                    f"    {_GREEN}✓ {rec['source_barcode']}: {rec['source_clone']} → {rec['target_clone']} "
                    f"({rec['source_type']}→{rec['target_type']}; arms={rec['checked_arms']}, "
                    f"r={fmt_float(rec['mean_r'])}, "
                    f"max |ΔAF|={rec['max_median_abs_delta']:.1f}, "
                    f"cell_extreme={rec['max_cell_extreme_frac']:.3f}, "
                    f"target_extreme={rec['max_target_extreme_frac']:.3f}){_RESET}",
                    cfg,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        vprint("  No eligible cell-to-target merge candidates found.", cfg)
    else:
        n_merged_cells = int(out.get("merged", pd.Series(dtype=bool)).sum())
        if n_merged_cells == 0:
            vprint("  Cell-to-target merges performed: 0", cfg)
            # Print the closest base->LOH candidates so threshold failures are
            # visible in the notebook log rather than hidden in the diagnostic
            # table.  This is particularly useful when base clone heatmaps show
            # a few cells with apparent LOH-like stripes.
            try:
                cand = out.loc[
                    out["source_type"].isin(["base_non_loh", "possible_loh"])
                    & out["target_type"].eq("loh")
                ].copy()
                if len(cand) > 0:
                    cand["_rank_r"] = pd.to_numeric(cand["mean_r"], errors="coerce").fillna(-9)
                    cand["_rank_ext"] = pd.to_numeric(cand["min_mean_extreme_frac"], errors="coerce").fillna(-9)
                    cand["_rank_delta"] = pd.to_numeric(cand["max_median_abs_delta"], errors="coerce").fillna(999)
                    cand = cand.sort_values(
                        ["_rank_r", "_rank_ext", "_rank_delta"],
                        ascending=[False, False, True],
                    ).head(10)
                    vprint("  Closest failed base→LOH candidates:", cfg)
                    for _, rr in cand.iterrows():
                        vprint(
                            f"    {rr.get('source_barcode', rr.get('source_cell'))}: "
                            f"{rr.get('source_clone')} → {rr.get('target_clone')} | "
                            f"arms={rr.get('checked_arms')}, "
                            f"r={fmt_float(rr.get('mean_r'))}, "
                            f"max |ΔAF|={float(rr.get('max_median_abs_delta')):.1f}, "
                            f"cell_extreme={float(rr.get('max_cell_extreme_frac')):.3f}, "
                            f"target_extreme={float(rr.get('max_target_extreme_frac')):.3f}, "
                            f"fail={rr.get('fail_reason')}",
                            cfg,
                        )
            except Exception:
                pass
        else:
            merged_rows = out.loc[out.get("merged", False).astype(bool)].copy()
            n_to_base = int(merged_rows["target_type"].eq("base_non_loh").sum()) if "target_type" in merged_rows.columns else 0
            n_to_loh = n_merged_cells - n_to_base
            n_from_base = int(merged_rows["source_type"].eq("base_non_loh").sum()) if "source_type" in merged_rows.columns else 0
            n_from_fp = int(merged_rows["source_type"].eq("possible_loh").sum()) if "source_type" in merged_rows.columns else 0
            vprint(
                f"  Cell-to-target merges performed: {n_merged_cells} cell(s) "
                f"({n_to_loh} → LOH clone, {n_to_base} → base non-LOH clone; "
                f"{n_from_base} promoted from base non-LOH; "
                f"{n_from_fp} rescued from possible-LOH/FP).",
                cfg,
            )

    try:
        state.results["phase4_small_loh_merge"] = out
    except Exception:
        pass

    return out


# ── Phase 4e: residual base-clone LOH discovery ──────────────────────
def _next_residual_loh_number(state) -> int:
    """Return the next globally unique LOH-C integer."""
    nums = []
    for label in state.clone_to_cells.keys():
        nums.extend(int(x) for x in re.findall(r"LOH-C(\d+)", str(label)))
    return max(nums) + 1 if nums else 0


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end runs of True values."""
    mask = np.asarray(mask, dtype=bool).ravel()
    runs: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _residual_component_homogeneous(X: np.ndarray, *, min_variants: int, min_mean_r: float, max_mad: float) -> tuple[bool, dict]:
    """Conservative cell-to-centroid validation for a residual LOH component."""
    from .loh_config import pearson_safe

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < int(min_variants):
        return False, {"reason": "insufficient_cells_or_variants", "mean_r": float("nan"), "n_usable_r": 0}

    centroid = np.nanmedian(X, axis=0)
    rs: list[float] = []
    mads: list[float] = []
    for row in X:
        keep = np.isfinite(row) & np.isfinite(centroid)
        if int(keep.sum()) < int(min_variants):
            continue
        r = pearson_safe(row[keep], centroid[keep], min_n=int(min_variants))
        if r is not None and np.isfinite(r):
            rs.append(float(r))
        mads.append(float(np.nanmedian(np.abs(row[keep] - centroid[keep]))))

    median_mad = float(np.nanmedian(mads)) if mads else float("nan")
    if len(rs) >= 2:
        mean_r = float(np.mean(rs))
        passed = mean_r >= float(min_mean_r) and np.isfinite(median_mad) and median_mad <= float(max_mad)
        return passed, {"reason": "pass" if passed else "mean_correlation_or_mad", "mean_r": mean_r, "median_abs_diff": median_mad, "n_usable_r": len(rs)}

    # A genuinely uniform LOW/HIGH block can have a non-informative Pearson r.
    # In that case require strict AF distance agreement rather than inventing a
    # correlation signal.
    passed = np.isfinite(median_mad) and median_mad <= float(max_mad)
    return passed, {"reason": "constant_profile_distance_pass" if passed else "insufficient_usable_correlations", "mean_r": float("nan"), "median_abs_diff": median_mad, "n_usable_r": len(rs)}


def _discover_residual_base_loh_subclones(state, cfg) -> pd.DataFrame:
    """Discover conservative de novo LOH subgroups in remaining base clones.

    This runs only after Phase 4 rescue against established targets. It scans
    the residual cells of each active base clone arm-by-arm, requiring a
    direction-consistent connected component, a shared contiguous extreme
    block, and independent component homogeneity. It never relabels cells
    that have no coherent new event.
    """
    from .loh_config import arm_variants, is_small_label, vprint

    if not bool(getattr(cfg, "phase4_residual_base_loh_discovery_enabled", True)):
        return pd.DataFrame()

    min_cells = int(getattr(cfg, "phase4_residual_base_loh_min_candidate_cells", cfg.min_final_clone_size))
    min_run = int(getattr(cfg, "phase4_residual_base_loh_min_contiguous_variants", cfg.min_contiguous_variants_initial))
    min_extreme = float(getattr(cfg, "phase4_residual_base_loh_min_extreme_fraction", cfg.min_extreme_fraction))
    min_shared = int(getattr(cfg, "phase4_residual_base_loh_min_shared_variants", cfg.min_shared_variants))
    min_mean_r = float(getattr(cfg, "phase4_residual_base_loh_min_mean_corr", cfg.phase3_min_mean_corr_to_mean))
    max_mad = float(getattr(cfg, "phase4_residual_base_loh_max_cell_median_abs_diff", 15.0))
    low = float(cfg.low_allele_threshold)
    high = float(cfg.high_allele_threshold)

    rows: list[dict] = []
    if not hasattr(state, "clone_block_variants") or state.clone_block_variants is None:
        state.clone_block_variants = {}
    current = state.current_clones.copy()
    next_num = _next_residual_loh_number(state)
    changed = False

    base_groups = [
        (str(label), [c for c in cells if c in state.df_numeric.index])
        for label, cells in state.clone_to_cells.items()
        if _is_active_non_loh(label) and not is_small_label(label) and len(cells) >= min_cells
    ]

    if not base_groups:
        return pd.DataFrame()

    vprint("\n------------------------------------------------------------", cfg)
    vprint("PHASE 4.6: Residual base-clone LOH discovery", cfg)
    vprint("------------------------------------------------------------", cfg)
    vprint(
        f"  Criteria: ≥{min_cells} cells; ≥{min_run} contiguous extreme variants; "
        f"shared directional variants ≥{min_shared}; mean cell-to-centroid r ≥{min_mean_r:.2f}.",
        cfg,
    )

    # Store accepted cells per base so one cell cannot be assigned to two
    # independent de novo arm discoveries in the same pass.
    used_cells_by_base: dict[str, set] = {}

    for base_label, base_cells in sorted(base_groups, key=lambda item: (-len(item[1]), item[0])):
        used_cells_by_base.setdefault(base_label, set())
        for arm in state.chr_arms:
            arm_vars = [str(v) for v in arm_variants(arm, state) if str(v) in state.df_numeric.columns]
            if len(arm_vars) < min_run:
                continue

            candidate_cells = [c for c in base_cells if c not in used_cells_by_base[base_label]]
            if len(candidate_cells) < min_cells:
                continue

            X = state.df_numeric.loc[candidate_cells, arm_vars].to_numpy(dtype=float)
            extreme = np.isfinite(X) & ((X <= low) | (X >= high))
            signs = np.zeros_like(X, dtype=np.int8)
            signs[np.isfinite(X) & (X <= low)] = -1
            signs[np.isfinite(X) & (X >= high)] = 1

            segment_mask = np.zeros_like(extreme, dtype=bool)
            eligible_idx: list[int] = []
            for i, row_extreme in enumerate(extreme):
                runs = [run for run in _contiguous_runs(row_extreme) if (run[1] - run[0] + 1) >= min_run]
                if not runs:
                    continue
                eligible_idx.append(i)
                for start, end in runs:
                    segment_mask[i, start:end + 1] = True

            if len(eligible_idx) < min_cells:
                continue

            # Connected components: cells must share enough same-direction
            # extreme variants inside their contiguous candidate segments.
            adjacency = {i: set() for i in eligible_idx}
            for pos_i, i in enumerate(eligible_idx):
                for j in eligible_idx[pos_i + 1:]:
                    shared = segment_mask[i] & segment_mask[j] & (signs[i] != 0) & (signs[i] == signs[j])
                    if int(shared.sum()) >= min_shared:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

            seen: set[int] = set()
            for start_i in eligible_idx:
                if start_i in seen:
                    continue
                stack = [start_i]
                seen.add(start_i)
                component: list[int] = []
                while stack:
                    i = stack.pop()
                    component.append(i)
                    for j in adjacency[i]:
                        if j not in seen:
                            seen.add(j)
                            stack.append(j)

                if len(component) < min_cells:
                    continue

                comp_X = X[np.asarray(component, dtype=int), :]
                called = np.isfinite(comp_X).sum(axis=0)
                ext_count = ((comp_X <= low) | (comp_X >= high)).sum(axis=0)
                ext_frac = np.divide(ext_count, called, out=np.zeros_like(ext_count, dtype=float), where=called > 0)
                valid_runs = [run for run in _contiguous_runs(ext_frac >= min_extreme) if (run[1] - run[0] + 1) >= min_run]
                if not valid_runs:
                    continue

                # Choose the most strongly supported longest block.
                best_run = sorted(
                    valid_runs,
                    key=lambda run: (run[1] - run[0] + 1, float(np.mean(ext_frac[run[0]:run[1] + 1])), -run[0]),
                    reverse=True,
                )[0]
                block_idx = list(range(best_run[0], best_run[1] + 1))
                block_vars = [arm_vars[k] for k in block_idx]
                passed, metrics = _residual_component_homogeneous(
                    comp_X[:, block_idx],
                    min_variants=min_run,
                    min_mean_r=min_mean_r,
                    max_mad=max_mad,
                )

                comp_cells = [candidate_cells[i] for i in component]
                status = "rejected_homogeneity"
                new_label = ""
                if passed:
                    new_label = f"{base_label} LOH-C{next_num} {arm}"
                    next_num += 1
                    status = "new_loh_subclone"
                    for cell in comp_cells:
                        bc = state.cell_to_barcode.get(cell, str(cell))
                        idx = state.sample_idx_for_barcode(str(bc))
                        if idx is not None:
                            current.iloc[idx] = new_label
                    state.clone_block_variants[new_label] = list(block_vars)
                    used_cells_by_base[base_label].update(comp_cells)
                    changed = True
                    mean_r_text = metrics.get("mean_r", float("nan"))
                    mean_r_text = "NA" if not np.isfinite(mean_r_text) else f"{float(mean_r_text):.3f}"
                    vprint(
                        f"  ✓ {base_label}: {len(comp_cells)} cell(s) → {new_label} "
                        f"(block={len(block_vars)}v, mean_r={mean_r_text})",
                        cfg,
                    )

                rows.append({
                    "base_label": base_label,
                    "arm": arm,
                    "status": status,
                    "new_label": new_label,
                    "n_cells": len(comp_cells),
                    "block_variants": ";".join(block_vars),
                    "block_n_variants": len(block_vars),
                    "block_extreme_fraction": float(np.mean(ext_frac[block_idx])),
                    **metrics,
                    "barcodes": ";".join(str(state.cell_to_barcode.get(c, c)) for c in comp_cells),
                })

    if changed:
        state.set_clone_labels(current)

    audit = pd.DataFrame(rows)
    if not audit.empty:
        state.results["phase4_residual_base_loh_discovery"] = audit
    if audit.empty:
        vprint("  No coherent de novo residual LOH subgroups found.", cfg)
    else:
        n_new = int((audit["status"] == "new_loh_subclone").sum())
        vprint(f"  Residual LOH discovery: {n_new} accepted subgroup(s).", cfg)
    return audit



# ── Phase 4f: final residual base-clone refinement ───────────────────
def _final_residual_arm_components(
    *,
    base_cells: list[str],
    arm: str,
    state,
    cfg,
    min_cells: int,
    min_run: int,
    min_extreme: float,
    min_shared: int,
    min_mean_r: float,
    max_mad: float,
) -> list[dict]:
    """Return coherent, direction-compatible residual components for one arm.

    Components are built only from cells with a contiguous extreme-AF run. An
    edge links two cells when they share ``min_shared`` extreme variants in the
    same LOW/HIGH orientation. This keeps the final refinement arm-specific
    and prevents a reciprocal mixture from being promoted as one LOH event.
    """
    from .loh_config import arm_variants

    low = float(cfg.low_allele_threshold)
    high = float(cfg.high_allele_threshold)
    arm_vars = [
        str(v) for v in arm_variants(arm, state)
        if str(v) in state.df_numeric.columns
    ]
    if len(arm_vars) < min_run:
        return []

    cells = [str(c) for c in base_cells if str(c) in state.df_numeric.index]
    if len(cells) < min_cells:
        return []

    X = state.df_numeric.loc[cells, arm_vars].to_numpy(dtype=float)
    extreme = np.isfinite(X) & ((X <= low) | (X >= high))
    signs = np.zeros_like(X, dtype=np.int8)
    signs[np.isfinite(X) & (X <= low)] = -1
    signs[np.isfinite(X) & (X >= high)] = 1

    segment_mask = np.zeros_like(extreme, dtype=bool)
    eligible_idx: list[int] = []
    for i, row_extreme in enumerate(extreme):
        runs = [
            run for run in _contiguous_runs(row_extreme)
            if (run[1] - run[0] + 1) >= min_run
        ]
        if not runs:
            continue
        eligible_idx.append(i)
        for start, end in runs:
            segment_mask[i, start:end + 1] = True

    if len(eligible_idx) < min_cells:
        return []

    adjacency = {i: set() for i in eligible_idx}
    for pos_i, i in enumerate(eligible_idx):
        for j in eligible_idx[pos_i + 1:]:
            shared = (
                segment_mask[i]
                & segment_mask[j]
                & (signs[i] != 0)
                & (signs[i] == signs[j])
            )
            if int(shared.sum()) >= min_shared:
                adjacency[i].add(j)
                adjacency[j].add(i)

    out: list[dict] = []
    seen: set[int] = set()
    for start_i in eligible_idx:
        if start_i in seen:
            continue
        stack = [start_i]
        seen.add(start_i)
        component: list[int] = []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)

        if len(component) < min_cells:
            continue

        comp_X = X[np.asarray(component, dtype=int), :]
        called = np.isfinite(comp_X).sum(axis=0)
        ext_count = ((comp_X <= low) | (comp_X >= high)).sum(axis=0)
        ext_frac = np.divide(
            ext_count,
            called,
            out=np.zeros_like(ext_count, dtype=float),
            where=called > 0,
        )
        valid_runs = [
            run for run in _contiguous_runs(ext_frac >= min_extreme)
            if (run[1] - run[0] + 1) >= min_run
        ]
        if not valid_runs:
            continue

        best_run = sorted(
            valid_runs,
            key=lambda run: (
                run[1] - run[0] + 1,
                float(np.mean(ext_frac[run[0]:run[1] + 1])),
                -run[0],
            ),
            reverse=True,
        )[0]
        block_idx = list(range(best_run[0], best_run[1] + 1))
        block_vars = [arm_vars[k] for k in block_idx]
        passed, metrics = _residual_component_homogeneous(
            comp_X[:, block_idx],
            min_variants=min_run,
            min_mean_r=min_mean_r,
            max_mad=max_mad,
        )
        if not passed:
            continue

        out.append({
            "arm": str(arm),
            "cells": [cells[i] for i in component],
            "block_variants": block_vars,
            "block_n_variants": len(block_vars),
            "block_extreme_fraction": float(np.mean(ext_frac[block_idx])),
            "metrics": metrics,
        })

    return out


def _existing_exact_arm_target(*, base_label: str, arms: tuple[str, ...], state) -> str | None:
    """Return an unambiguous active LOH target with the exact arm set."""
    from .loh_config import extract_chr_arms, get_base_clone, is_loh_label, sort_chromosome_arms

    wanted = tuple(sort_chromosome_arms(list(arms)))
    matches = []
    for label, cells in state.clone_to_cells.items():
        if not is_loh_label(label) or not cells:
            continue
        if str(get_base_clone(str(label))) != str(base_label):
            continue
        got = tuple(sort_chromosome_arms(list(dict.fromkeys(extract_chr_arms(str(label))))))
        if got == wanted:
            matches.append(str(label))
    return matches[0] if len(matches) == 1 else None


def _final_residual_direction_signature(
    X: np.ndarray,
    *,
    min_variants: int,
    low: float,
    high: float,
    min_per_variant_agreement: float,
    min_block_pass_frac: float,
    min_extreme_cells_per_variant: int,
) -> tuple[bool, dict]:
    """Require a shared, variant-wise LOW/HIGH signature within one component.

    A multi-arm event may be LOW on one arm and HIGH on another.  This helper
    is called *per arm*, so each variant is evaluated against its own component
    consensus and no pooled cross-arm direction is imposed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < int(min_variants):
        return False, {
            "reason": "insufficient_cells_or_variants",
            "n_evaluated": 0,
            "n_passing": 0,
            "block_pass_frac": float("nan"),
        }

    signs = np.zeros(X.shape, dtype=np.int8)
    signs[np.isfinite(X) & (X <= float(low))] = -1
    signs[np.isfinite(X) & (X >= float(high))] = 1

    evaluated = 0
    passing = 0
    agreements: list[float] = []
    for j in range(X.shape[1]):
        n_low = int(np.sum(signs[:, j] == -1))
        n_high = int(np.sum(signs[:, j] == 1))
        n_extreme = n_low + n_high
        if n_extreme < int(min_extreme_cells_per_variant):
            continue
        evaluated += 1
        agreement = float(max(n_low, n_high) / n_extreme)
        agreements.append(agreement)
        if agreement >= float(min_per_variant_agreement):
            passing += 1

    frac = float(passing / evaluated) if evaluated else float("nan")
    ok = (
        evaluated >= int(min_variants)
        and passing >= int(min_variants)
        and np.isfinite(frac)
        and frac >= float(min_block_pass_frac)
    )
    return ok, {
        "reason": "pass" if ok else "directional_signature_inconsistent",
        "n_evaluated": int(evaluated),
        "n_passing": int(passing),
        "block_pass_frac": frac,
        "per_variant_agreements": agreements,
    }


def _final_residual_lineage_size(state, base_label: str) -> int:
    """Return all currently labelled cells from the same genotype lineage."""
    from .loh_config import get_base_clone

    n = 0
    for label, cells in state.clone_to_cells.items():
        try:
            if str(get_base_clone(str(label))) == str(base_label):
                n += len(cells)
        except Exception:
            continue
    return int(n)


def _final_refine_remaining_base_clones(state, cfg) -> pd.DataFrame:
    """One conservative pre-Phase-5 residual base-clone LOH validation pass.

    This is intentionally *not* iterative and never creates de novo ``small``
    groups.  A residual signature is accepted only when it has at least the
    final clone size, a contiguous high-extreme block on every represented arm,
    at least three independent block amplicons, independent per-arm
    homogeneity, a shared variant-wise directional signature, and an ADO tail
    probability that is unlikely under the configured amplicon model.
    """
    from .ado_probability import score_ado_combination
    from .loh_config import is_small_label, sort_chromosome_arms, vprint

    if not bool(getattr(cfg, "phase4_final_residual_refinement_enabled", True)):
        return pd.DataFrame()

    min_active = max(
        int(getattr(cfg, "min_final_clone_size", 5)),
        int(getattr(cfg, "phase4_final_residual_refinement_min_active_cells", cfg.min_final_clone_size)),
    )
    min_run = int(
        getattr(
            cfg,
            "phase4_final_residual_refinement_min_contiguous_variants",
            cfg.min_contiguous_variants_initial,
        )
    )
    min_extreme = float(
        getattr(cfg, "phase4_final_residual_refinement_min_extreme_fraction", cfg.min_extreme_fraction)
    )
    min_shared = int(
        getattr(cfg, "phase4_final_residual_refinement_min_shared_variants", cfg.min_shared_variants)
    )
    min_mean_r = float(
        getattr(cfg, "phase4_final_residual_refinement_min_mean_corr", cfg.phase3_min_mean_corr_to_mean)
    )
    max_mad = float(
        getattr(cfg, "phase4_final_residual_refinement_max_cell_median_abs_diff", 15.0)
    )
    min_amplicons = int(
        getattr(cfg, "phase4_final_residual_refinement_min_different_amplicons", cfg.min_different_amplicons)
    )
    require_ado_unlikely = bool(
        getattr(cfg, "phase4_final_residual_refinement_require_ado_unlikely", True)
    )
    ado_threshold = float(
        getattr(cfg, "phase4_final_residual_refinement_ado_pvalue_threshold", cfg.ado_probability_pvalue_threshold)
    )
    ado_default_rate = float(
        getattr(cfg, "phase4_final_residual_refinement_ado_rate_per_amplicon", cfg.ado_probability_default_amplicon_rate)
    )

    if not hasattr(state, "clone_block_variants") or state.clone_block_variants is None:
        state.clone_block_variants = {}

    state.refresh()
    current = state.current_clones.copy()
    next_num = _next_residual_loh_number(state)
    v2a = _build_variant_to_amplicon(state)
    audit_rows: list[dict] = []
    changed = 0

    vprint("\n------------------------------------------------------------", cfg)
    vprint("PHASE 4.7: Final conservative residual base-clone validation", cfg)
    vprint("------------------------------------------------------------", cfg)
    vprint(
        f"  Single pass only. New active LOH requires n≥{min_active}; "
        f"≥{min_run} contiguous extreme variants; ≥{min_amplicons} independent amplicons; "
        "per-arm homogeneity; shared directional signature; and ADO-unlikely support.",
        cfg,
    )

    base_groups = [
        (str(label), [str(c) for c in cells if str(c) in state.df_numeric.index])
        for label, cells in state.clone_to_cells.items()
        if _is_active_non_loh(label) and not is_small_label(label) and len(cells) >= min_active
    ]

    for base_label, base_cells in sorted(base_groups, key=lambda item: (-len(item[1]), item[0])):
        arm_components: dict[str, list[dict]] = {}
        membership: dict[str, list[tuple[str, int]]] = {cell: [] for cell in base_cells}

        for arm in state.chr_arms:
            comps = _final_residual_arm_components(
                base_cells=base_cells,
                arm=str(arm),
                state=state,
                cfg=cfg,
                min_cells=min_active,
                min_run=min_run,
                min_extreme=min_extreme,
                min_shared=min_shared,
                min_mean_r=min_mean_r,
                max_mad=max_mad,
            )
            if not comps:
                continue
            arm_components[str(arm)] = comps
            for comp_i, comp in enumerate(comps):
                for cell in comp["cells"]:
                    membership.setdefault(str(cell), []).append((str(arm), comp_i))

        signature_groups: dict[tuple[tuple[str, int], ...], list[str]] = {}
        for cell, signature in membership.items():
            if signature:
                key = tuple(sorted(signature, key=lambda x: x[0]))
                signature_groups.setdefault(key, []).append(str(cell))

        for signature, cells in sorted(signature_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(cells) < min_active:
                continue

            arms = tuple(sort_chromosome_arms([arm for arm, _ in signature]))
            union_block: list[str] = []
            per_arm: list[dict] = []
            valid = True

            for arm, comp_i in signature:
                comp = arm_components[arm][comp_i]
                block_vars = list(comp["block_variants"])
                X = state.df_numeric.loc[cells, block_vars].to_numpy(dtype=float)
                homo_pass, homo = _residual_component_homogeneous(
                    X,
                    min_variants=min_run,
                    min_mean_r=min_mean_r,
                    max_mad=max_mad,
                )
                direction_pass, direction = _final_residual_direction_signature(
                    X,
                    min_variants=min_run,
                    low=float(cfg.low_allele_threshold),
                    high=float(cfg.high_allele_threshold),
                    min_per_variant_agreement=float(cfg.directional_min_per_variant_agreement),
                    min_block_pass_frac=float(cfg.directional_min_block_pass_frac),
                    min_extreme_cells_per_variant=int(cfg.directional_min_extreme_cells_per_variant),
                )
                per_arm.append({
                    "arm": str(arm),
                    "block_vars": block_vars,
                    "homogeneity": homo,
                    "direction": direction,
                })
                union_block.extend(block_vars)
                if not homo_pass or not direction_pass:
                    valid = False

            union_block = list(dict.fromkeys(str(v) for v in union_block))
            block_amplicons = sorted({str(v2a.get(v, v)) for v in union_block})
            if len(block_amplicons) < min_amplicons:
                valid = False
                amplicon_reason = f"insufficient_independent_amplicons:{len(block_amplicons)}<{min_amplicons}"
            else:
                amplicon_reason = "pass"

            lineage_n = _final_residual_lineage_size(state, base_label)
            ado = score_ado_combination(
                block_amplicons=block_amplicons,
                amplicon_ado_rates=getattr(cfg, "amplicon_ado_rates", None),
                original_clone_size=lineage_n,
                observed_subclone_size=len(cells),
                required_amplicons=len(block_amplicons),
                rate_col=getattr(cfg, "ado_probability_rate_col", "smooth_ado_any_rate"),
                amplicon_col=getattr(cfg, "ado_probability_amplicon_col", "amplicon_id"),
                default_amplicon_ado_rate=ado_default_rate,
                pvalue_threshold=ado_threshold,
            )
            ado_pass = (not require_ado_unlikely) or (ado.get("classification") == "ADO_UNLIKELY")
            if not ado_pass:
                valid = False

            base_audit = {
                "base_label": base_label,
                "arms": ";".join(arms),
                "n_cells": len(cells),
                "block_variants": ";".join(union_block),
                "block_n_variants": len(union_block),
                "block_amplicons": ";".join(block_amplicons),
                "n_block_amplicons": len(block_amplicons),
                "amplicon_reason": amplicon_reason,
                "ado_classification": ado.get("classification"),
                "ado_p_at_least_observed": ado.get("p_at_least_observed"),
                "ado_expected_cells": ado.get("expected_ado_cells"),
                "ado_cell_probability": ado.get("single_cell_ado_probability"),
                "barcodes": ";".join(str(state.cell_to_barcode.get(c, c)) for c in cells),
                "per_arm_homogeneity": ";".join(
                    f"{entry['arm']}:{entry['homogeneity'].get('reason')}"
                    for entry in per_arm
                ),
                "per_arm_direction": ";".join(
                    f"{entry['arm']}:{entry['direction'].get('reason')}"
                    for entry in per_arm
                ),
            }

            if not valid:
                base_audit["status"] = "rejected_conservative_validation"
                audit_rows.append(base_audit)
                continue

            existing_target = _existing_exact_arm_target(
                base_label=base_label,
                arms=arms,
                state=state,
            )
            if existing_target is not None:
                new_label = existing_target
                status = "assigned_existing_exact_arm_target"
            else:
                tokens = " ".join(f"LOH-C{next_num} {arm}" for arm in arms)
                new_label = f"{base_label} {tokens}".strip()
                next_num += 1
                status = "new_active_loh_subclone"

            for cell in cells:
                bc = state.cell_to_barcode.get(cell, str(cell))
                idx = state.sample_idx_for_barcode(str(bc))
                if idx is not None and str(current.iloc[idx]) != new_label:
                    current.iloc[idx] = new_label
                    changed += 1
            state.clone_block_variants[new_label] = union_block

            base_audit["status"] = status
            base_audit["new_label"] = new_label
            audit_rows.append(base_audit)
            vprint(
                f"  ✓ {base_label}: {len(cells)} cell(s) → {new_label} "
                f"(arms={','.join(arms)}, block={len(union_block)}v, amplicons={len(block_amplicons)}, "
                f"ADO={ado.get('classification')})",
                cfg,
            )

    if changed:
        state.set_clone_labels(current)
        vprint(f"  Final conservative residual validation reassigned {changed} cell(s) before Phase 5.", cfg)
    else:
        vprint("  No residual base-clone signature passed conservative validation.", cfg)

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        state.results["phase4_final_residual_refinement"] = audit
    return audit

def run_phase4(
    state: "PipelineState",
    cfg: "LOHConfig | None" = None,
) -> dict:
    """
    For each non-LOH clone with ≥ min_final_clone_size cells, check
    correlation against LOH clones that share the same base genotype.
    Merge if correlation ≥ threshold.

    Returns the updated ``state.results`` dict.
    """
    from .loh_config import (
        LOH_TAG,
        vprint,
        arm_variants,
        extract_chr_arms,
        fmt_float,
    )

    if cfg is None:
        cfg = state.config

    min_size = cfg.min_final_clone_size
    corr_thresh = cfg.correlation_threshold
    extreme_thresh = cfg.merge_min_extreme_frac
    low_af = cfg.low_allele_threshold
    high_af = cfg.high_allele_threshold

    vprint("\n" + "=" * 60, cfg)
    vprint("PHASE 4: Non-LOH ↔ LOH Clone Correlation & Merge", cfg)
    vprint("=" * 60, cfg)

    # ── Identify LOH and non-LOH clones ───────────────────────────────
    loh_clones = {
        lbl: cells
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_loh(lbl) and len(cells) > 0
    }
    non_loh_clones = {
        lbl: cells
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_non_loh(lbl) and len(cells) >= min_size
    }

    if not loh_clones:
        vprint("  No LOH clones found — nothing to check.", cfg)
        vprint("=" * 60 + "\n", cfg)
        return state.results

    if not non_loh_clones:
        vprint("  No non-LOH clones (≥ min size) to check.", cfg)
        vprint("=" * 60 + "\n", cfg)
        return state.results

    # ── Build base-name index for LOH clones ──────────────────────────
    # base_name → list of (loh_label, arms)
    loh_by_base: dict[str, list[tuple[str, list[str]]]] = {}
    for lbl in loh_clones:
        base = _get_base_name(lbl, LOH_TAG)
        arms = extract_chr_arms(lbl)
        loh_by_base.setdefault(base, []).append((lbl, arms))

    vprint(f"\n  LOH clones:     {len(loh_clones)}", cfg)
    vprint(f"  Non-LOH clones: {len(non_loh_clones)}  (≥ {min_size} cells)", cfg)
    vprint(f"  LOH base names: {sorted(loh_by_base.keys())}", cfg)
    vprint(
        f"  Merge requires: r ≥ {corr_thresh}  AND  extreme fraction ≥ {extreme_thresh}",
        cfg,
    )

    # ── Check each non-LOH clone ──────────────────────────────────────
    merges_done = 0
    merges_skipped = 0

    for non_loh_lbl, non_loh_cells in sorted(
        non_loh_clones.items(), key=lambda kv: -len(kv[1])
    ):
        base = _get_base_name(non_loh_lbl, LOH_TAG)

        if base not in loh_by_base:
            vprint(
                f"\n  {non_loh_lbl}  ({len(non_loh_cells)} cells)"
                f"  — no matching LOH clone, skip.",
                cfg,
            )
            continue

        vprint(
            f"\n  {non_loh_lbl}  ({len(non_loh_cells)} cells)"
            f"  — checking against LOH clone(s):",
            cfg,
        )

        # Try each matching LOH clone
        for loh_lbl, loh_arms in loh_by_base[base]:
            loh_cells = state.clone_to_cells[loh_lbl]

            vprint(
                f"    vs {loh_lbl}  ({len(loh_cells)} cells, "
                f"arms: {', '.join(loh_arms)})",
                cfg,
            )

            # Compute correlation AND extreme fraction on each LOH arm
            arm_corrs = []
            arm_extremes = []
            for arm in loh_arms:
                variants = arm_variants(arm, state)
                if len(variants) < 2:
                    vprint(f"      {arm}: < 2 variants, skip", cfg)
                    continue

                loh_barcodes = [str(bc) for bc in loh_cells if str(bc) in state.df.index]
                non_loh_barcodes = [str(bc) for bc in non_loh_cells if str(bc) in state.df.index]

                if len(loh_barcodes) < 2 or len(non_loh_barcodes) < 2:
                    vprint(
                        f"      {arm}: too few cells in df "
                        f"(LOH={len(loh_barcodes)}, non-LOH={len(non_loh_barcodes)})",
                        cfg,
                    )
                    continue

                loh_profile = state.df.loc[loh_barcodes, variants].mean(axis=0).values
                non_loh_profile = state.df.loc[non_loh_barcodes, variants].mean(axis=0).values

                # Check for constant profiles (would give NaN correlation)
                if np.std(loh_profile) < 1e-6 or np.std(non_loh_profile) < 1e-6:
                    vprint(f"      {arm}: constant profile, skip", cfg)
                    continue

                r, p = pearsonr(loh_profile, non_loh_profile)
                arm_corrs.append((arm, r, p))

                non_loh_vals = state.df.loc[non_loh_barcodes, variants].values
                extreme_mask = (non_loh_vals <= low_af) | (non_loh_vals >= high_af)
                per_cell_extreme = np.nanmean(extreme_mask, axis=1)
                clone_extreme_frac = float(np.nanmean(per_cell_extreme))
                arm_extremes.append((arm, clone_extreme_frac))

                vprint(
                    f"      {arm}: r = {fmt_float(r)},  p = {fmt_float(p)}"
                    f"  |  extreme frac = {clone_extreme_frac:.3f}",
                    cfg,
                )

            if not arm_corrs:
                vprint(f"    → No valid correlations, skip.", cfg)
                merges_skipped += 1
                continue

            # Use mean correlation across arms
            mean_r = np.mean([r for _, r, _ in arm_corrs])
            # Use mean extreme fraction across arms
            mean_extreme = np.mean([ef for _, ef in arm_extremes]) if arm_extremes else 0.0

            vprint(
                f"    → Mean r = {fmt_float(mean_r)}  (threshold = {corr_thresh})"
                f"  |  Mean extreme frac = {mean_extreme:.3f}  (threshold = {extreme_thresh})",
                cfg,
            )

            passes_corr = mean_r >= corr_thresh
            passes_extreme = mean_extreme >= extreme_thresh

            # "Hint of similarity" trigger for per-cell rescue
            hint_mean_r = float(getattr(cfg, "phase4_percell_trigger_mean_corr", 0.35))
            hint_any_arm_r = float(getattr(cfg, "phase4_percell_trigger_any_arm_corr", 0.75))

            arm_r_values = [float(r) for _, r, _ in arm_corrs if np.isfinite(r)]
            max_arm_r = max(arm_r_values) if arm_r_values else float("-inf")

            has_similarity_hint = (
                (np.isfinite(mean_r) and mean_r >= hint_mean_r)
                or (np.isfinite(max_arm_r) and max_arm_r >= hint_any_arm_r)
            )

            if passes_corr and passes_extreme:
                # ── MERGE ───────────────────────────────────────────
                vprint(
                    f"    {_GREEN}✓ MERGE: {non_loh_lbl} → {loh_lbl}{_RESET}  "
                    f"({len(non_loh_cells)} cells)",
                    cfg,
                )

                clone_arr = state.Sample.dna.row_attrs["Clone"]
                barcode_arr = pd.Series(state.Sample.dna.row_attrs["barcode"]).astype(str).values

                for bc in non_loh_cells:
                    mask = barcode_arr == str(bc)
                    if mask.any():
                        clone_arr[mask] = loh_lbl

                state.Sample.dna.row_attrs["Clone"] = clone_arr

                state.clone_to_cells[loh_lbl] = list(set(list(loh_cells) + list(non_loh_cells)))
                if non_loh_lbl in state.clone_to_cells:
                    del state.clone_to_cells[non_loh_lbl]

                merges_done += 1
                break
            else:
                reasons = []
                if not passes_corr:
                    reasons.append(f"r={fmt_float(mean_r)} < {corr_thresh}")
                if not passes_extreme:
                    reasons.append(f"extreme={mean_extreme:.3f} < {extreme_thresh}")
                vprint(f"    {_RED}✗ No merge: {'; '.join(reasons)}{_RESET}", cfg)

                # ── Per-cell rescue ────────────────────────────────
                if cfg.phase4_percell_rescue and (passes_corr or has_similarity_hint):
                    if not passes_corr and has_similarity_hint:
                        vprint(
                            f"    Per-cell rescue triggered by similarity hint "
                            f"(mean_r={fmt_float(mean_r)}, max_arm_r={fmt_float(max_arm_r)}; "
                            f"hint thresholds: mean≥{hint_mean_r}, any-arm≥{hint_any_arm_r})",
                            cfg,
                        )

                    pc_min_r = cfg.phase4_percell_min_corr
                    pc_min_ef = cfg.phase4_percell_min_extreme
                    allowed_msg = getattr(cfg, "phase4_rescue_max_ado_amplicons", 1)
                    vprint(
                        f"    Per-cell rescue: block-aware, r ≥ {pc_min_r}, "
                        f"allow ≤{allowed_msg} non-supporting/ADO amplicon(s) per block",
                        cfg,
                    )

                    loh_profiles: dict[str, np.ndarray] = {}
                    arm_var_lists: dict[str, list[str]] = {}
                    v2a = _build_variant_to_amplicon(state)

                    for arm in loh_arms:
                        variants = _rescue_variants_for_clone_arm(loh_lbl, arm, state, cfg, arm_variants)
                        if len(variants) < 2:
                            continue
                        loh_bcs = [str(bc) for bc in loh_cells if str(bc) in state.df.index]
                        if len(loh_bcs) < 2:
                            continue
                        prof = state.df.loc[loh_bcs, variants].mean(axis=0).values.astype(float)
                        # Do not skip constant profiles: a true LOW/HIGH block can be constant.
                        loh_profiles[arm] = prof
                        arm_var_lists[arm] = variants

                    if loh_profiles:
                        rescued: list[tuple[str, float, float, int, int, int]] = []
                        not_rescued: list[tuple[str, float, float, int, int, int]] = []

                        for bc in non_loh_cells:
                            bc_str = str(bc)
                            if bc_str not in state.df.index:
                                continue

                            cell_rs = []
                            cell_support = []
                            n_pass_total = 0
                            n_req_total = 0
                            n_amp_total = 0
                            n_block_amp_total = 0
                            n_som_amp_total = 0

                            for arm, prof in loh_profiles.items():
                                variants = arm_var_lists[arm]
                                cell_vals = state.df.loc[bc_str, variants].values.astype(float)
                                score = _score_cell_against_loh_profile_ado_tolerant(
                                    cell_vals,
                                    prof,
                                    variants,
                                    v2a=v2a,
                                    cfg=cfg,
                                    low_af=low_af,
                                    high_af=high_af,
                                )
                                n_pass_total += int(score["n_pass_amplicons"])
                                n_req_total += int(score["n_required_amplicons"])
                                n_amp_total += int(score["n_total_amplicons"])
                                n_block_amp_total += int(score.get("n_total_block_amplicons", score["n_total_amplicons"]))
                                n_som_amp_total += int(score.get("n_somatic_excluded_amplicons", 0))
                                cell_support.append(float(score["amp_support_frac"]))
                                if np.isfinite(score["r"]):
                                    cell_rs.append(float(score["r"]))

                            avg_r = float(np.mean(cell_rs)) if cell_rs else np.nan
                            avg_support = float(np.mean(cell_support)) if cell_support else 0.0
                            support_ok = n_pass_total >= n_req_total and n_req_total > 0
                            corr_ok = (
                                np.isfinite(avg_r) and avg_r >= pc_min_r
                            ) or (
                                bool(getattr(cfg, "phase4_rescue_allow_amplicon_support_without_corr", False))
                                and support_ok
                            )

                            rescue_tuple = (
                                bc_str, avg_r, avg_support, n_pass_total, n_req_total,
                                n_amp_total, n_block_amp_total, n_som_amp_total,
                            )
                            if support_ok and corr_ok:
                                rescued.append(rescue_tuple)
                            else:
                                not_rescued.append(rescue_tuple)

                        if rescued:
                            for bc_str, r_val, support_val, n_pass, n_req, n_amp, n_block_amp, n_som_amp in rescued:
                                som_txt = ""
                                if n_som_amp > 0 or n_block_amp > n_amp:
                                    som_txt = f" excluding_somatic={n_som_amp}amp/{max(n_block_amp, n_amp)}block_amp"
                                vprint(
                                    f"      {_GREEN}✓ {bc_str}: r={fmt_float(r_val)}, "
                                    f"amp_support={n_pass}/{n_amp} required≥{n_req}"
                                    f"{som_txt} ({support_val:.0%}) → {loh_lbl}{_RESET}",
                                    cfg,
                                )

                            clone_arr = state.Sample.dna.row_attrs["Clone"]
                            barcode_arr = pd.Series(state.Sample.dna.row_attrs["barcode"]).astype(str).values

                            for bc_str, *_ in rescued:
                                mask = barcode_arr == bc_str
                                if mask.any():
                                    clone_arr[mask] = loh_lbl

                            state.Sample.dna.row_attrs["Clone"] = clone_arr

                            rescued_bcs = [bc for bc, *_ in rescued]
                            rescued_set = set(map(str, rescued_bcs))

                            state.clone_to_cells[loh_lbl] = list(
                                set(list(loh_cells) + rescued_bcs)
                            )

                            remaining = [
                                bc for bc in non_loh_cells
                                if str(bc) not in rescued_set
                            ]
                            if remaining:
                                state.clone_to_cells[non_loh_lbl] = remaining
                            elif non_loh_lbl in state.clone_to_cells:
                                del state.clone_to_cells[non_loh_lbl]

                            vprint(
                                f"    Per-cell rescue: {_GREEN}{len(rescued)} cell(s) → {loh_lbl}{_RESET}, "
                                f"{len(remaining)} cell(s) remain in {non_loh_lbl}",
                                cfg,
                            )
                            merges_done += 1

                            # Important: a partial per-cell rescue should not
                            # terminate testing for this base genotype.  Keep
                            # checking the remaining cells against other same-base
                            # LOH clones, e.g. chr12q first and then chr4q.
                            non_loh_cells = remaining

                            if len(non_loh_cells) < min_size:
                                vprint(
                                    f"    Remaining non-LOH cells below min size "
                                    f"({len(non_loh_cells)} < {min_size}); stop checking this clone.",
                                    cfg,
                                )
                                break

                            continue
                        else:
                            vprint("      No cells passed per-cell rescue.", cfg)

                elif cfg.phase4_percell_rescue:
                    vprint(
                        f"    Per-cell rescue skipped: no similarity hint "
                        f"(mean_r={fmt_float(mean_r)} < {hint_mean_r}, "
                        f"max_arm_r={fmt_float(max_arm_r)} < {hint_any_arm_r})",
                        cfg,
                    )

                merges_skipped += 1

    # Phase 4c pooled rescue and Phase 4c-ii arm expansion were removed.
    # They are intentionally not called because they can inflate small seeds
    # into broad false-positive LOH subclones.

    # ── Phase 4b: WT/Het → Hem/Hom ADO check ─────────────────────────
    ado_removed = _het_ado_check(state, cfg)

    # ── Refresh before end-of-phase diagnostics ───────────────────────
    # Phase 4 may have changed clone labels through full-clone merge,
    # per-cell rescue, and/or ADO relabelling. Refresh unconditionally when
    # anything changed so cross-LOH similarity uses the final Phase 4 state.
    if merges_done > 0 or ado_removed > 0:
        vprint(f"\n  Refreshing state after Phase 4 updates ...", cfg)
        state.refresh()

    # ── Phase 4c: Cross-LOH clone similarity diagnostic ───────────────
    phase4_cross_loh_similarity = _cross_loh_clone_similarity_check(
        state=state,
        cfg=cfg,
        LOH_TAG=LOH_TAG,
        arm_variants_fn=arm_variants,
        extract_chr_arms_fn=extract_chr_arms,
        low_af=low_af,
        high_af=high_af,
    )
    state.results["phase4_cross_loh_similarity"] = phase4_cross_loh_similarity

    # ── Phase 4d: iterative cell-level reassignment to a fixed point ───
    # Every pass rebuilds target profiles using the labels from the preceding
    # pass. This lets clone centroids become progressively better defined while
    # retaining an explicit convergence cap and cycle detection safeguard.
    iterate_4d = bool(getattr(cfg, "phase4_iterate_cell_merge", True))
    max_iters = (
        int(getattr(cfg, "phase4_cell_merge_max_iters", 10))
        if iterate_4d else 1
    )
    max_iters = max(1, max_iters)
    detect_cycles = bool(getattr(cfg, "phase4_cell_merge_detect_cycles", True))

    if iterate_4d:
        vprint(
            f"\n  Iterative Phase 4d reassignment: enabled; maximum {max_iters} pass(es). "
            "Each pass rebuilds clone profiles from the current assignments.",
            cfg,
        )

    small_loh_cells_merged = 0
    merge_history: list[pd.DataFrame] = []
    iteration_rows: list[dict] = []
    seen_label_states: set[tuple[str, ...]] = set()
    converged = False
    cycle_detected = False

    for iteration in range(1, max_iters + 1):
        labels_before = tuple(state.current_clones.astype(str).tolist())

        if detect_cycles and labels_before in seen_label_states:
            cycle_detected = True
            vprint(
                f"{_YELLOW}  Iterative cell assignment stopped before pass {iteration}: "
                "a previously observed full-cell label state recurred. "
                "This prevents reassignment oscillation; inspect the iteration audit."
                f"{_RESET}",
                cfg,
            )
            break

        seen_label_states.add(labels_before)

        merge_out = _merge_small_loh_clones_after_annotation(
            state=state,
            cfg=cfg,
            LOH_TAG=LOH_TAG,
            arm_variants_fn=arm_variants,
            extract_chr_arms_fn=extract_chr_arms,
            low_af=low_af,
            high_af=high_af,
        )

        diagnostic_merged_rows = 0
        candidate_rows = 0
        if isinstance(merge_out, pd.DataFrame):
            candidate_rows = int(len(merge_out))
            iter_audit = merge_out.copy()
            iter_audit.insert(0, "iteration", iteration)
            merge_history.append(iter_audit)
            if not iter_audit.empty and "merged" in iter_audit.columns:
                try:
                    diagnostic_merged_rows = int(
                        iter_audit["merged"].fillna(False).astype(bool).sum()
                    )
                except Exception:
                    diagnostic_merged_rows = 0

        labels_after = tuple(state.current_clones.astype(str).tolist())
        actual_reassignments = int(
            sum(before != after for before, after in zip(labels_before, labels_after))
        )

        iteration_rows.append(
            {
                "iteration": iteration,
                "candidate_comparisons": candidate_rows,
                "diagnostic_merged_rows": diagnostic_merged_rows,
                "actual_reassignments": actual_reassignments,
                "n_loh_cells_after_pass": int(state.loh_cell_count()),
                "n_active_loh_clones_after_pass": int(
                    len([label for label in state.clone_to_cells if _is_active_loh(label)])
                ),
            }
        )

        if iterate_4d:
            vprint(
                f"  Iterative cell assignment — pass {iteration}: "
                f"{actual_reassignments} cell(s) reassigned "
                f"({candidate_rows} candidate comparison row(s)).",
                cfg,
            )

        if actual_reassignments == 0:
            converged = True
            break

        small_loh_cells_merged += actual_reassignments
        state.refresh()

    if merge_history:
        phase4_small_loh_merge = pd.concat(merge_history, ignore_index=True)
    else:
        phase4_small_loh_merge = pd.DataFrame()

    phase4_cell_assignment_iterations = pd.DataFrame(iteration_rows)
    state.results["phase4_small_loh_merge"] = phase4_small_loh_merge
    state.results["phase4_cell_assignment_iterations"] = (
        phase4_cell_assignment_iterations
    )
    state.results["phase4_cell_assignment_convergence"] = {
        "iterative_enabled": bool(iterate_4d),
        "max_iterations": int(max_iters),
        "passes_run": int(len(iteration_rows)),
        "converged": bool(converged),
        "cycle_detected": bool(cycle_detected),
        "total_reassignments": int(small_loh_cells_merged),
    }

    if iterate_4d:
        per_pass = [
            int(row["actual_reassignments"])
            for row in iteration_rows
        ]
        if converged:
            vprint(
                f"  Cell assignment converged after {len(iteration_rows)} pass(es) "
                f"(per-pass reassignments: {per_pass}).",
                cfg,
            )
        elif cycle_detected:
            vprint(
                f"{_YELLOW}  Cell assignment stopped after {len(iteration_rows)} pass(es) "
                f"because a label-state cycle was detected "
                f"(per-pass reassignments: {per_pass}).{_RESET}",
                cfg,
            )
        else:
            last_count = per_pass[-1] if per_pass else 0
            vprint(
                f"{_YELLOW}  Cell assignment reached the iteration cap ({max_iters}) "
                f"without convergence; final pass reassigned {last_count} cell(s). "
                "Inspect state.results['phase4_cell_assignment_iterations'] and "
                "state.results['phase4_small_loh_merge']."
                f"{_RESET}",
                cfg,
            )

    if small_loh_cells_merged > 0:
        merges_done += small_loh_cells_merged

    # Legacy Phase 4e residual discovery is intentionally not run here.
    # Phase 4f below is the single conservative residual-base validation pass.
    residual_new_cells = 0

    # ── Phase 4f: final residual signature refinement before Phase 5 ───
    final_refinement = _final_refine_remaining_base_clones(state, cfg)
    final_refinement_cells = 0
    if isinstance(final_refinement, pd.DataFrame) and not final_refinement.empty:
        final_refinement_cells = int(final_refinement.loc[
            final_refinement["status"].isin([
                "assigned_existing_exact_arm_target",
                "new_active_loh_subclone",
            ]),
            "n_cells",
        ].sum())
        if final_refinement_cells > 0:
            state.refresh()
            vprint(
                f"  Final residual refinement reassigned {final_refinement_cells} cell(s) before Phase 5.",
                cfg,
            )

    # ── Summary ───────────────────────────────────────────────────────
    vprint(f"\n  Merges performed: {merges_done}", cfg)
    if merges_skipped > 0:
        vprint(f"  Merge candidates skipped (low correlation): {merges_skipped}", cfg)
    if ado_removed > 0:
        vprint(f"  ADO cells removed from Het/WT candidates: {ado_removed}", cfg)
    if small_loh_cells_merged > 0:
        vprint(f"  Small LOH cells merged into larger LOH clones: {small_loh_cells_merged}", cfg)

    if merges_done > 0 or ado_removed > 0 or small_loh_cells_merged > 0 or residual_new_cells > 0 or final_refinement_cells > 0:
        vprint(
            f"  LOH clones after:  {len([l for l in state.clone_to_cells if _is_active_loh(l)])}  "
            f"({state.loh_cell_count()} cells)",
            cfg,
        )
        vprint(f"  LOH labels:  {state.loh_clone_labels()}", cfg)
    else:
        vprint("\n  No merges needed — clone structure unchanged.", cfg)

    if isinstance(phase4_cross_loh_similarity, pd.DataFrame) and not phase4_cross_loh_similarity.empty:
        n_suggest = int(phase4_cross_loh_similarity["suggest_merge"].sum())
        if n_suggest > 0:
            vprint(f"{_YELLOW}  Manual LOH↔LOH merge/review suggestions: {n_suggest}{_RESET}", cfg)

    vprint("=" * 60 + "\n", cfg)

    return state.results
