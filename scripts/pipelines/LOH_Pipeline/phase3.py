"""
Phase 3 – LOH Clone QC: Outlier Removal, Homogeneity, Rescue, Reconciliation
==============================================================================

Steps executed in order:

1. **Per-cell outlier removal** — correlate every cell to clone mean on
   LOH-arm variants. Strip cells below ``mean(r) − k × std(r)``
   (default k=2).

2. **Rescue merges** — small LOH clones merged into a compatible LOH
   clone (same base, arms ⊆ target; small→small allowed).

3. **Clone-level homogeneity** — discard clones whose average
   cell-to-mean r is below ``phase3_min_mean_corr_to_mean``. Directional
   consistency is checked on the LOH block for single-arm clones and separately
   on each LOH arm for multi-arm clones. Optionally, a single discordant
   amplicon can be tolerated for strong blocks, consistent with ADO/dropout.

4. **Discard small LOH clones** — revert clones below
   ``min_final_clone_size``.

4b. **Flag reverted LOH outliers** — cells stripped of LOH that
   still show extreme AF pattern are retained as per-clone `FP: ...` labels.

5. **Results reconciliation** — prune ``state.results`` to match
   surviving labels.

Call after Phase 2::

    from phase3 import run_phase3
    results = run_phase3(state)

Controlled by ``cfg.skip_homogeneity_check``.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loh_config import LOHConfig, PipelineState


# ── ANSI helpers ───────────────────────────────────────────────────
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════
# Shared internal utilities
# ═══════════════════════════════════════════════════════════════════

def _loh_clone_map(state: "PipelineState") -> dict[str, list]:
    """``{clone_label: [cell_ids]}`` for active LOH labels only."""
    from .loh_config import is_loh_label
    return {
        cl: cells
        for cl, cells in state.clone_to_cells.items()
        if is_loh_label(cl)
    }


def _loh_arm_variants(clone_label: str, state: "PipelineState") -> list[str]:
    """All unique variants across the LOH arms in *clone_label*."""
    from .loh_config import extract_chr_arms, arm_variants
    all_vars: list[str] = []
    for arm in extract_chr_arms(clone_label):
        all_vars.extend(arm_variants(arm, state))
    return list(dict.fromkeys(all_vars))


def _loh_block_variants(
    clone_label: str,
    state: "PipelineState",
) -> list[str] | None:
    """
    For a localized LOH clone, return only the variants in the LOH block.

    Priority:
    1. ``state.clone_block_variants[clone_label]`` — persisted by Phase 1
       from the actual detected LOH block indices.
    2. Cytoband annotation in the clone label (e.g. ``chr5q:q32-q33.1``).
    3. None — caller falls back to all arm variants.
    """
    # ── 1) Phase 1 persisted block variants ───────────────────────────
    block_map = getattr(state, "clone_block_variants", None)
    if block_map and clone_label in block_map:
        bv = block_map[clone_label]
        if bv:
            return list(bv)

    # ── 2) Parse cytoband annotation from label ──────────────────────
    from .loh_config import extract_chr_arms, arm_variants

    cytobands_df = getattr(state, "cytobands_df", None)
    if cytobands_df is None or len(cytobands_df) == 0:
        return None

    label = str(clone_label)

    # Parse cytoband annotations like "chr5q:q32-q33.1" or "chr5q:q32"
    # from the clone label.
    cytoband_pattern = re.compile(
        r"(chr(?:\d+|X|Y)[pq]):([pq][\w.]+(?:-[pq][\w.]+)?)"
    )
    matches = cytoband_pattern.findall(label)
    if not matches:
        return None

    block_variants: list[str] = []

    for arm, band_range in matches:
        arm_vars = arm_variants(arm, state)
        if not arm_vars:
            continue

        # Parse band range: "q32-q33.1" → start_band="q32", end_band="q33.1"
        # or single band: "q32" → start_band="q32", end_band="q32".
        if "-" in band_range:
            start_band, end_band = band_range.split("-", 1)
        else:
            start_band = end_band = band_range

        # Get cytoband intervals for this arm.
        arm_bands = cytobands_df[cytobands_df["chromosome arm"] == arm]
        if len(arm_bands) == 0:
            continue

        # Find the genomic range covered by start_band through end_band.
        matching_bands = []
        in_range = False
        for _, row in arm_bands.sort_values("start").iterrows():
            band_name = str(row["cytoband"])
            if band_name == start_band:
                in_range = True
            if in_range:
                matching_bands.append(row)
            if band_name == end_band:
                break

        if not matching_bands:
            # Fallback: try matching as substrings, e.g. q32 matches q32.1.
            for _, row in arm_bands.sort_values("start").iterrows():
                band_name = str(row["cytoband"])
                if (
                    band_name.startswith(start_band)
                    or band_name.startswith(end_band)
                ):
                    matching_bands.append(row)

        if not matching_bands:
            continue

        region_start = min(int(r["start"]) for r in matching_bands)
        region_end = max(int(r["end"]) for r in matching_bands)

        # Filter variants to those within the cytoband region.
        for v in arm_vars:
            try:
                pos = int(str(v).split(":")[1].split("-")[0])
            except (IndexError, ValueError):
                continue
            if region_start <= pos <= region_end:
                block_variants.append(v)

    if not block_variants:
        return None

    return list(dict.fromkeys(block_variants))


def _variant_to_amplicon_map(state: "PipelineState") -> dict[str, str]:
    """
    Return {variant_id: amplicon_id} from Sample.dna.col_attrs.

    Falls back to variant ID when amplicon metadata is unavailable. This keeps
    the ADO-tolerance rule conservative: if we cannot prove two bad variants
    are in the same amplicon, they are treated as independent bad amplicons.
    """
    out: dict[str, str] = {}

    try:
        col_attrs = state.Sample.dna.col_attrs
        ids = list(col_attrs.get("id", []))
        amps = list(col_attrs.get("amplicon", []))

        if len(ids) == len(amps) and len(ids) > 0:
            out.update({str(v): str(a) for v, a in zip(ids, amps)})
    except Exception:
        pass

    try:
        for v in state.df_numeric.columns:
            out.setdefault(str(v), str(v))
    except Exception:
        pass

    return out


def _phase3_direction_pass_with_ado_tolerance(
    *,
    dir_res: dict,
    loh_vars: list[str],
    state: "PipelineState",
    cfg: "LOHConfig",
) -> tuple[bool, dict]:
    """
    ADO-aware rescue for Phase 3 directional-consistency failures.

    This is intentionally narrow. It only rescues a block that failed the
    hard directional gate when the failure is explainable by at most a small
    number of discordant amplicons, default one, in a sufficiently supported
    block.

    Example intended rescue:
        vars=block:8, dir=MIXED (7/8=88%) ✗
        → tolerate 1 bad amplicon when evaluated block variants >= 6.
    """

    if bool(dir_res.get("passes", False)):
        return True, {
            "ado_tolerated": False,
            "reason": "direction_original_pass",
        }

    if not bool(getattr(cfg, "phase3_allow_single_ado_amplicon_direction_fail", False)):
        return False, {
            "ado_tolerated": False,
            "reason": "phase3_direction_ado_tolerance_disabled",
        }

    min_block_variants = int(
        getattr(cfg, "phase3_direction_ado_min_block_variants", 6)
    )
    max_bad_amplicons = int(
        getattr(cfg, "phase3_direction_ado_max_bad_amplicons", 1)
    )
    min_good_variants = int(
        getattr(cfg, "phase3_direction_ado_min_good_variants", 5)
    )

    loh_vars = [str(v) for v in loh_vars]
    agreements = list(dir_res.get("per_variant_agreements", []))
    directions = list(dir_res.get("per_variant_directions", []))

    # Pad safely if an older direction helper returned shorter arrays.
    if len(agreements) < len(loh_vars):
        agreements = agreements + [float("nan")] * (len(loh_vars) - len(agreements))
    if len(directions) < len(loh_vars):
        directions = directions + ["NA"] * (len(loh_vars) - len(directions))

    evaluated: list[tuple[str, float, str]] = []
    for v, agreement, direction in zip(loh_vars, agreements, directions):
        try:
            ag = float(agreement)
        except Exception:
            continue
        if not np.isfinite(ag):
            continue
        evaluated.append((str(v), ag, str(direction)))

    n_evaluated = len(evaluated)
    if n_evaluated < min_block_variants:
        return False, {
            "ado_tolerated": False,
            "reason": f"evaluated_block_too_short:{n_evaluated}<{min_block_variants}",
            "n_evaluated": n_evaluated,
        }

    per_var_thr = float(getattr(cfg, "directional_min_per_variant_agreement", 0.95))

    good_variants = [v for v, ag, _ in evaluated if ag >= per_var_thr]
    bad_variants = [v for v, ag, _ in evaluated if ag < per_var_thr]

    n_good = len(good_variants)
    n_bad = len(bad_variants)

    if n_good < min_good_variants:
        return False, {
            "ado_tolerated": False,
            "reason": f"too_few_good_variants:{n_good}<{min_good_variants}",
            "n_evaluated": n_evaluated,
            "n_good_variants": n_good,
            "n_bad_variants": n_bad,
        }

    if n_bad == 0:
        # Defensive. If the block failed with no bad evaluated variants, do not
        # override. That points to an unexpected directional helper state.
        return False, {
            "ado_tolerated": False,
            "reason": "direction_failed_without_bad_variants",
            "n_evaluated": n_evaluated,
            "n_good_variants": n_good,
            "n_bad_variants": n_bad,
        }

    v2a = _variant_to_amplicon_map(state)
    bad_amplicons = sorted({str(v2a.get(str(v), str(v))) for v in bad_variants})
    n_bad_amplicons = len(bad_amplicons)

    if n_bad_amplicons <= max_bad_amplicons:
        return True, {
            "ado_tolerated": True,
            "reason": (
                f"single_amplicon_direction_ado_tolerated:"
                f"{n_good}/{n_evaluated} good variants, "
                f"{n_bad_amplicons} bad amplicon(s)"
            ),
            "n_evaluated": n_evaluated,
            "n_good_variants": n_good,
            "n_bad_variants": n_bad,
            "n_bad_amplicons": n_bad_amplicons,
            "bad_variants": bad_variants,
            "bad_amplicons": bad_amplicons,
        }

    return False, {
        "ado_tolerated": False,
        "reason": f"too_many_bad_amplicons:{n_bad_amplicons}>{max_bad_amplicons}",
        "n_evaluated": n_evaluated,
        "n_good_variants": n_good,
        "n_bad_variants": n_bad,
        "n_bad_amplicons": n_bad_amplicons,
        "bad_variants": bad_variants,
        "bad_amplicons": bad_amplicons,
    }


# ═══════════════════════════════════════════════════════════════════
# Step 1: Per-cell outlier removal
# ═══════════════════════════════════════════════════════════════════

def _per_cell_outlier_removal(state: "PipelineState", cfg: "LOHConfig") -> int:
    """
    Within each LOH clone, correlate every cell to the clone mean on
    LOH-arm variants. For clones larger than
    ``phase3_outlier_small_clone_max_cells`` (default 10), strip cells whose
    r falls below ``mean(r) − k × std(r)`` where k =
    ``phase3_outlier_k`` (default 2.0).

    For small clones (at or below the configured size), relative trimming is
    deliberately disabled because an extremely narrow within-clone correlation
    distribution can incorrectly strip a biologically coherent cell. These
    cells are removed only when r is below the absolute
    ``phase3_outlier_small_clone_abs_corr_floor`` (default 0.80).

    For localized LOH clones (with cytoband annotation), uses only the
    LOH-block variants for correlation.

    For clones with < 3 scoreable cells, skips entirely.

    Returns total number of cells stripped of LOH.
    """
    from .loh_config import (
        vprint, get_base_clone, pearson_safe, fmt_float,
    )

    min_vars = int(cfg.phase3_min_variants_for_cell_corr)
    k = float(getattr(cfg, "phase3_outlier_k", 2.0))

    loh_map = _loh_clone_map(state)
    total_stripped = 0

    if not loh_map:
        vprint("  (no LOH clones found)", cfg)
        return 0

    for clone_label, cells in loh_map.items():
        cells_in_df = [c for c in cells if c in state.df_numeric.index]
        if len(cells_in_df) < 3:
            continue

        # For localized LOH, use block variants; fall back to all arm variants.
        loh_vars = _loh_block_variants(clone_label, state)
        if loh_vars is None or len(loh_vars) < 3:
            loh_vars = _loh_arm_variants(clone_label, state)
        if len(loh_vars) < min_vars:
            continue

        # Clone mean profile.
        X = state.df_numeric.loc[cells_in_df, loh_vars].values.astype(float)
        mu = np.nanmean(X, axis=0)

        # Per-cell correlation to mean.
        cell_r: dict[str, float] = {}
        for i, cell_id in enumerate(cells_in_df):
            xi = X[i, :]
            mask = np.isfinite(xi) & np.isfinite(mu)
            if int(mask.sum()) < min_vars:
                continue
            r = pearson_safe(xi, mu, min_n=min_vars)
            if r is not None:
                cell_r[cell_id] = float(r)

        if len(cell_r) < 3:
            continue

        rs = np.array(list(cell_r.values()), dtype=float)
        mean_r = float(np.mean(rs))
        std_r = float(np.std(rs, ddof=1))

        small_clone_max_cells = getattr(
            cfg, "phase3_outlier_small_clone_max_cells", 10
        )
        small_clone_abs_floor = float(getattr(
            cfg, "phase3_outlier_small_clone_abs_corr_floor", 0.80
        ))
        use_absolute_only = (
            small_clone_max_cells is not None
            and len(cells_in_df) <= int(small_clone_max_cells)
        )

        if use_absolute_only:
            # Small clone: avoid a clone-relative cutoff that can become
            # spuriously stringent when almost every cell is near-identical.
            cutoff = small_clone_abs_floor
            outlier_ids = [
                cid for cid, r in cell_r.items()
                if r < small_clone_abs_floor
            ]
            method = (
                "small-clone absolute-only "
                f"(n={len(cells_in_df)}≤{int(small_clone_max_cells)}; "
                f"r<{small_clone_abs_floor:.3f}; relative mean-{k:.1f}*std disabled)"
            )
        else:
            # Larger clone: use the established clone-relative criterion.
            if std_r > 0:
                cutoff = mean_r - k * std_r
            else:
                # All correlations identical; avoid arbitrary stripping.
                cutoff = mean_r
            outlier_ids = [cid for cid, r in cell_r.items() if r < cutoff]
            method = f"mean-{k:.1f}*std (mean_r={mean_r:.3f}, std={std_r:.3f})"

        # Report the policy even when it retains every cell: this makes it
        # clear why a small-clone r=0.883 was not treated as an FP outlier.
        if use_absolute_only and not outlier_ids:
            vprint(
                f"  {clone_label} (n={len(cells_in_df)}): {method} → "
                f"{_GREEN}no outliers{_RESET}",
                cfg,
            )
            continue

        if not outlier_ids:
            continue

        vprint(
            f"  {clone_label} (n={len(cells_in_df)}): "
            f"{method}, cutoff={fmt_float(cutoff, 3)} → "
            f"{_RED}{len(outlier_ids)} outlier(s){_RESET}",
            cfg,
        )
        for cid in outlier_ids:
            vprint(
                f"    {_RED}strip {cid}  r={fmt_float(cell_r[cid], 3)}{_RESET}",
                cfg,
            )

        # Strip LOH annotation from outlier cells.
        current = state.current_clones.copy()
        base = get_base_clone(clone_label)
        for cid in outlier_ids:
            bc = state.cell_to_barcode.get(cid, str(cid))
            idx = state.sample_idx_for_barcode(str(bc))
            if idx is not None and str(current.iloc[idx]) == clone_label:
                current.iloc[idx] = base

        state.set_clone_labels(current)
        total_stripped += len(outlier_ids)

    return total_stripped


# ═══════════════════════════════════════════════════════════════════
# Step 2: Rescue merges
# ═══════════════════════════════════════════════════════════════════

def _rescue_merges(state: "PipelineState", cfg: "LOHConfig") -> int:
    """
    Merge small LOH clones into a compatible LOH clone (same base,
    arms ⊆ target; small→small allowed).

    Strict: source must match target mean on every target arm
    (r ≥ post_merge_correlation_threshold).

    Returns number of source labels merged.
    """
    from .loh_config import (
        vprint, get_base_clone, extract_chr_arms,
        arm_variants, mean_profile, pearson_safe, fmt_float,
    )

    loh_map = _loh_clone_map(state)
    min_size = int(cfg.min_final_clone_size)
    thr = float(cfg.post_merge_correlation_threshold)
    min_arm_vars = max(3, int(getattr(cfg, "min_variants_per_arm_for_corr", 5)))

    size_of = {cl: len(cells) for cl, cells in loh_map.items()}
    small_sources = [cl for cl, n in size_of.items() if n < min_size]
    if not small_sources:
        return 0

    # Union-find for transitive merges.
    parent = {cl: cl for cl in loh_map}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    planned: list[tuple[str, str, float]] = []

    for src in small_sources:
        if find(src) != src:
            continue

        src_cells = [c for c in loh_map.get(src, []) if c in state.df_numeric.index]
        if not src_cells:
            vprint(f"  {_RED}{src}  (n={size_of.get(src, 0)}) ✗{_RESET}", cfg)
            continue

        base = get_base_clone(src)
        src_arms = set(extract_chr_arms(src))

        best_tgt = None
        best_score = -1e9

        for tgt in loh_map:
            if tgt == src or get_base_clone(tgt) != base:
                continue
            tgt_arms = list(dict.fromkeys(extract_chr_arms(tgt)))
            # Rescue merges preserve event complexity: a source can only move
            # into a target with a proper superset of LOH arms.
            if not src_arms < set(tgt_arms):
                continue

            tgt_cells = [c for c in loh_map.get(tgt, []) if c in state.df_numeric.index]
            if not tgt_cells:
                continue

            ok = True
            rs_used: list[float] = []
            for arm in tgt_arms:
                vars_arm = [
                    v for v in arm_variants(arm, state)
                    if v in state.df_numeric.columns
                ]
                if len(vars_arm) < min_arm_vars:
                    ok = False
                    break
                mu_src = mean_profile(state.df_numeric, src_cells, vars_arm)
                mu_tgt = mean_profile(state.df_numeric, tgt_cells, vars_arm)
                r = pearson_safe(mu_src, mu_tgt, min_n=min_arm_vars)
                if r is None or float(r) < thr:
                    ok = False
                    break
                rs_used.append(float(r))

            if not ok:
                continue

            score = float(np.mean(rs_used)) if rs_used else -1e9
            if score > best_score:
                best_score = score
                best_tgt = tgt

        if best_tgt is None:
            vprint(f"  {_RED}{src}  (n={size_of.get(src, 0)}) ✗{_RESET}", cfg)
            continue

        planned.append((src, best_tgt, best_score))
        parent[find(src)] = find(best_tgt)
        vprint(
            f"  {_GREEN}{src}  (n={size_of.get(src, 0)}) → {best_tgt}  "
            f"(score={fmt_float(best_score)}) ✓{_RESET}",
            cfg,
        )

    if not planned:
        return 0

    current = state.current_clones.copy()
    applied = 0
    for src, tgt, _ in planned:
        tgt_root = find(tgt)
        if src == tgt_root:
            continue
        mask = current == src
        if mask.any():
            current[mask] = tgt_root
            applied += 1

    if applied:
        state.set_clone_labels(current)
    return applied


# ═══════════════════════════════════════════════════════════════════
# Step 3: Clone-level homogeneity
# ═══════════════════════════════════════════════════════════════════

def _homogeneity_metrics(
    X: np.ndarray,
    loh_vars: list[str],
    state: "PipelineState",
    cfg: "LOHConfig",
    *,
    min_vars: int = 3,
    skip_directional: bool = False,
) -> dict:
    """Evaluate correlation and directional consistency for one cell subset."""
    from .loh_config import pearson_safe, check_directional_consistency

    X = np.asarray(X, dtype=float)
    out = {
        "passes": False,
        "mean_r": float("nan"),
        "n_usable_r": 0,
        "dir_res": None,
        "dir_failed": False,
        "dir_ado_tolerated": False,
        "dir_ado_info": {},
        "reason": "",
    }
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < min_vars:
        out["reason"] = "insufficient_cells_or_variants"
        return out

    mu = np.nanmean(X, axis=0)
    rs = []
    for i in range(X.shape[0]):
        r = pearson_safe(X[i, :], mu, min_n=min_vars)
        if r is not None:
            rs.append(float(r))
    out["n_usable_r"] = len(rs)
    # Do not return before the directional check. A reciprocal 50:50 mixture
    # can make the parent clone mean nearly flat, yielding no valid Pearson r,
    # while the per-variant LOW/HIGH disagreement still contains exactly the
    # information needed to split the clone on the cell axis.
    if len(rs) >= 2:
        out["mean_r"] = float(np.mean(rs))

    dir_failed = False
    if (
        bool(getattr(cfg, "directional_check_enabled", True))
        and not bool(skip_directional)
    ):
        dir_res = check_directional_consistency(
            X,
            low_thr=cfg.low_allele_threshold,
            high_thr=cfg.high_allele_threshold,
            min_per_var_agreement=cfg.directional_min_per_variant_agreement,
            min_block_pass_frac=cfg.directional_min_block_pass_frac,
            min_extreme_cells_per_var=cfg.directional_min_extreme_cells_per_variant,
        )
        out["dir_res"] = dir_res
        if dir_res["n_variants_evaluated"] > 0 and not dir_res["passes"]:
            dir_pass_final, dir_ado_info = _phase3_direction_pass_with_ado_tolerance(
                dir_res=dir_res,
                loh_vars=loh_vars,
                state=state,
                cfg=cfg,
            )
            out["dir_ado_info"] = dir_ado_info
            if dir_pass_final and dir_ado_info.get("ado_tolerated", False):
                out["dir_ado_tolerated"] = True
            else:
                dir_failed = True
    out["dir_failed"] = bool(dir_failed)
    # Prioritise a failed directional check. It is actionable through the
    # Phase-3 cell-axis split even where a reciprocal mixture makes the
    # parent clone mean too flat for a valid correlation.
    if dir_failed:
        out["reason"] = "directional_discordance"
    elif out["n_usable_r"] < 2:
        out["reason"] = "insufficient_usable_correlations"
    elif out["mean_r"] < float(cfg.phase3_min_mean_corr_to_mean):
        out["reason"] = "mean_correlation"
    else:
        out["passes"] = True
        out["reason"] = "pass"
    return out


def _directional_summary(metrics: dict) -> str:
    """Compact, stable directional summary for Phase-3 verbosity."""
    if metrics.get("multi_arm_directional_mode") == "not_applied":
        return "dir=not_applied_multi_arm"
    per_arm = metrics.get("per_arm_metrics")
    if per_arm:
        parts: list[str] = []
        for arm, arm_metrics in per_arm.items():
            dir_res = arm_metrics.get("dir_res")
            if not dir_res or int(dir_res.get("n_variants_evaluated", 0)) == 0:
                parts.append(f"{arm}:not_evaluated")
                continue
            frac = dir_res.get("block_pass_frac", np.nan)
            frac_text = "NA" if not np.isfinite(frac) else f"{float(frac):.0%}"
            arm_text = (
                f"{arm}:{dir_res.get('dominant_direction', 'NA')} "
                f"({dir_res.get('n_variants_passing', 0)}/"
                f"{dir_res.get('n_variants_evaluated', 0)}={frac_text})"
            )
            if arm_metrics.get("dir_ado_tolerated", False):
                arm_text += ";ADO-tolerated"
            parts.append(arm_text)
        return "dir=per-arm[" + " | ".join(parts) + "]"

    dir_res = metrics.get("dir_res")
    if not dir_res or int(dir_res.get("n_variants_evaluated", 0)) == 0:
        return "dir=not_evaluated"
    frac = dir_res.get("block_pass_frac", np.nan)
    frac_text = "NA" if not np.isfinite(frac) else f"{float(frac):.0%}"
    text = (
        f"dir={dir_res.get('dominant_direction', 'NA')} "
        f"({dir_res.get('n_variants_passing', 0)}/"
        f"{dir_res.get('n_variants_evaluated', 0)}={frac_text})"
    )
    if metrics.get("dir_ado_tolerated", False):
        text += "; ADO-tolerated"
    return text


def _next_loh_clone_number(state: "PipelineState") -> int:
    """Next globally unique ``LOH-C<n>`` integer for derivative subclones."""
    nums = []
    for label in state.clone_to_cells.keys():
        nums.extend(int(x) for x in re.findall(r"LOH-C(\d+)", str(label)))
    return (max(nums) + 1) if nums else 0


def _split_directional_failure(
    *,
    clone_label: str,
    cells_in_df: list,
    X: np.ndarray,
    loh_vars: list[str],
    state: "PipelineState",
    cfg: "LOHConfig",
    next_clone_number: int,
) -> tuple[dict[object, str], dict, int]:
    """
    Split a directionally discordant clone along a cell-axis LOW/HIGH signature.

    Only one-LOH-token labels are split. Multi-arm labels are retained as a
    false-positive unit rather than inventing an ambiguous new multi-arm clone.
    Returned mapping covers every input cell. When the split itself succeeds,
    cells that cannot be allocated to a validated child are marked ``SUB:``
    (unresolved subclonal LOH), not technical false positives.
    """
    from .loh_config import (
        get_base_clone, make_fp_label, make_sub_label, check_directional_consistency,
    )

    fp_label = make_fp_label(clone_label)
    sub_label = make_sub_label(clone_label)
    default_assignment = {cell: sub_label for cell in cells_in_df}
    info = {
        "attempted": False,
        "reason": "",
        "n_discordant_variants": 0,
        "n_recovered_subclones": 0,
        "n_fp": 0,
        "n_sub": len(cells_in_df),
        "children": [],
    }

    if not bool(getattr(cfg, "phase3_split_directional_mixed", True)):
        info["reason"] = "directional_split_disabled"
        default_assignment = {cell: fp_label for cell in cells_in_df}
        info["n_fp"] = len(cells_in_df)
        info["n_sub"] = 0
        return default_assignment, info, next_clone_number

    tokens = re.findall(r"(?:^|\s)(LOH-C\d+\s+\S+)", str(clone_label))
    if len(tokens) != 1:
        info["reason"] = "multi_arm_or_unparseable_label"
        default_assignment = {cell: fp_label for cell in cells_in_df}
        info["n_fp"] = len(cells_in_df)
        info["n_sub"] = 0
        return default_assignment, info, next_clone_number

    dir_res = check_directional_consistency(
        X,
        low_thr=cfg.low_allele_threshold,
        high_thr=cfg.high_allele_threshold,
        min_per_var_agreement=cfg.directional_min_per_variant_agreement,
        min_block_pass_frac=cfg.directional_min_block_pass_frac,
        min_extreme_cells_per_var=cfg.directional_min_extreme_cells_per_variant,
    )
    agreements = list(dir_res.get("per_variant_agreements", []))
    discordant = [
        j for j, agreement in enumerate(agreements)
        if np.isfinite(agreement)
        and float(agreement) < float(cfg.directional_min_per_variant_agreement)
    ]
    info["n_discordant_variants"] = len(discordant)
    min_discordant = int(getattr(cfg, "phase3_directional_split_min_discordant_variants", 1))
    if len(discordant) < max(1, min_discordant):
        info["reason"] = "no_usable_discordant_variants"
        default_assignment = {cell: fp_label for cell in cells_in_df}
        info["n_fp"] = len(cells_in_df)
        info["n_sub"] = 0
        return default_assignment, info, next_clone_number

    # Select the most balanced discordant site as the initial LOW/HIGH anchor.
    low_thr, high_thr = cfg.low_allele_threshold, cfg.high_allele_threshold
    balance = []
    for j in discordant:
        col = X[:, j]
        n_low = int(np.sum(np.isfinite(col) & (col <= low_thr)))
        n_high = int(np.sum(np.isfinite(col) & (col >= high_thr)))
        balance.append((min(n_low, n_high), n_low + n_high, -j, j))
    _, _, _, anchor_j = max(balance)

    states = np.zeros((X.shape[0], len(discordant)), dtype=np.int8)
    for k, j in enumerate(discordant):
        states[:, k][np.isfinite(X[:, j]) & (X[:, j] <= low_thr)] = -1
        states[:, k][np.isfinite(X[:, j]) & (X[:, j] >= high_thr)] = 1
    anchor_k = discordant.index(anchor_j)
    seed_low = np.where(states[:, anchor_k] == -1)[0]
    seed_high = np.where(states[:, anchor_k] == 1)[0]
    if len(seed_low) == 0 or len(seed_high) == 0:
        info["reason"] = "anchor_does_not_separate_cells"
        default_assignment = {cell: fp_label for cell in cells_in_df}
        info["n_fp"] = len(cells_in_df)
        info["n_sub"] = 0
        return default_assignment, info, next_clone_number

    # Learn a per-variant sign signature from each anchor group, then assign
    # every informative cell to its closer signature. This handles haplotypes
    # such as [LOW,HIGH] versus [HIGH,LOW], where raw LOW/HIGH vote counts tie.
    signatures = []
    for seed in (seed_low, seed_high):
        sig = np.zeros(len(discordant), dtype=np.int8)
        for k in range(len(discordant)):
            vals = states[seed, k]
            n_low = int(np.sum(vals == -1))
            n_high = int(np.sum(vals == 1))
            sig[k] = -1 if n_low > n_high else (1 if n_high > n_low else 0)
        signatures.append(sig)

    groups = [[], []]
    for row_i in range(X.shape[0]):
        row = states[row_i]
        scores = []
        for sig in signatures:
            usable = (row != 0) & (sig != 0)
            scores.append(int(np.sum(row[usable] == sig[usable]) - np.sum(row[usable] != sig[usable])))
        if scores[0] == scores[1]:
            # Preserve exact anchor evidence; unresolved cells remain FP.
            if row[anchor_k] == -1:
                groups[0].append(row_i)
            elif row[anchor_k] == 1:
                groups[1].append(row_i)
        else:
            groups[int(scores[1] > scores[0])].append(row_i)

    min_cells = getattr(cfg, "phase3_directional_split_min_cells", None)
    min_cells = int(cfg.min_final_clone_size if min_cells is None else min_cells)
    assigned: dict[object, str] = dict(default_assignment)
    base = get_base_clone(clone_label)
    token = tokens[0]
    parent_block = (getattr(state, "clone_block_variants", {}) or {}).get(clone_label)

    info["attempted"] = True
    for group_i, row_idx in enumerate(groups, start=1):
        group_cells = [cells_in_df[i] for i in row_idx]
        if len(group_cells) < min_cells:
            continue
        metrics = _homogeneity_metrics(X[np.asarray(row_idx, dtype=int), :], loh_vars, state, cfg)
        if not metrics["passes"]:
            continue
        child_token = re.sub(r"LOH-C\d+", f"LOH-C{next_clone_number}", token, count=1)
        child_label = f"{base} {child_token}".strip()
        next_clone_number += 1
        for cell in group_cells:
            assigned[cell] = child_label
        info["children"].append({
            "label": child_label,
            "n_cells": len(group_cells),
            "metrics": metrics,
            "block_variants": list(parent_block) if parent_block else list(loh_vars),
        })

    info["n_recovered_subclones"] = len(info["children"])
    if info["children"] and bool(getattr(cfg, "phase3_directional_split_use_sub_label", True)):
        info["n_sub"] = sum(1 for label in assigned.values() if label == sub_label)
        info["n_fp"] = 0
    else:
        # No validated directional child exists: retain the original failure as FP.
        if not info["children"]:
            assigned = {cell: fp_label for cell in cells_in_df}
        info["n_fp"] = sum(1 for label in assigned.values() if label == fp_label)
        info["n_sub"] = 0
    info["reason"] = "split" if info["children"] else "no_subclone_passed_retest"
    return assigned, info, next_clone_number



# ═══════════════════════════════════════════════════════════════════
# Step 3a: High-similarity multi-arm directional decomposition
# ═══════════════════════════════════════════════════════════════════

def _multi_arm_loh_tokens(clone_label: str) -> list[tuple[str, str]]:
    """Return ordered ``(arm, LOH-token)`` pairs from a multi-arm label."""
    pattern = re.compile(
        r"(LOH-C\d+\s+(?P<arm>chr(?:\d+|X|Y)[pq])(?::[^\s]+)?)"
    )
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in pattern.finditer(str(clone_label)):
        full_token = str(match.group(1))
        arm = str(match.group("arm"))
        if arm in seen:
            continue
        out.append((arm, full_token))
        seen.add(arm)
    return out


def _multi_arm_block_for_arm(
    *,
    clone_label: str,
    cells_in_df: list,
    arm: str,
    token: str,
    state: "PipelineState",
    cfg: "LOHConfig",
    min_variants: int,
) -> list[str]:
    """Resolve a clone's persisted or cytoband-localised LOH block per arm."""
    from .loh_config import arm_variants

    arm_vars = [
        str(v) for v in arm_variants(str(arm), state)
        if str(v) in state.df_numeric.columns
    ]
    if not arm_vars:
        return []
    arm_set = set(arm_vars)

    # First derive the dominant contiguous LOH core directly from the parent
    # population. A pooled multi-arm block map can contain whole-arm variants;
    # using only its local extreme core prevents irrelevant flank variants from
    # manufacturing a directional failure during decomposition.
    source_cells = [c for c in cells_in_df if c in state.df_numeric.index]
    if source_cells:
        arm_af = state.df_numeric.loc[source_cells, arm_vars].apply(pd.to_numeric, errors="coerce")
        low_fraction = (arm_af <= float(cfg.low_allele_threshold)).mean(axis=0)
        high_fraction = (arm_af >= float(cfg.high_allele_threshold)).mean(axis=0)
        dominant_fraction = pd.concat([low_fraction, high_fraction], axis=1).max(axis=1)
        core_threshold = float(
            getattr(cfg, "phase3_multi_arm_decomposition_min_variant_extreme_fraction", 0.60)
        )
        passed = (dominant_fraction.reindex(arm_vars).fillna(0.0).to_numpy() >= core_threshold)
        best = None
        run_start = None
        for i, value in enumerate(passed):
            if value and run_start is None:
                run_start = i
            if (not value or i == len(passed) - 1) and run_start is not None:
                run_end = i if value and i == len(passed) - 1 else i - 1
                run_vars = arm_vars[run_start:run_end + 1]
                if len(run_vars) >= int(min_variants):
                    score = float(dominant_fraction.reindex(run_vars).mean())
                    candidate = (len(run_vars), score, run_start, run_vars)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
                run_start = None
        if best is not None:
            return list(best[3])

    persisted = [
        str(v)
        for v in (getattr(state, "clone_block_variants", {}) or {}).get(str(clone_label), [])
        if str(v) in arm_set
    ]
    persisted = list(dict.fromkeys(persisted))
    if len(persisted) >= int(min_variants):
        return persisted

    # Parse the arm-specific token itself so a multi-arm persisted map cannot
    # obscure a valid cytoband-localised block for this one arm.
    parsed = _loh_block_variants(str(token), state) or []
    parsed = list(dict.fromkeys(str(v) for v in parsed if str(v) in arm_set))
    if len(parsed) >= int(min_variants):
        return parsed

    # Conservative fallback: use the full available arm only when Phase 1 did
    # not persist enough block-specific variants. The later per-arm QC remains
    # strict, so this does not relax calling criteria.
    return arm_vars


def _multi_arm_consensus_support(
    X: np.ndarray,
    *,
    low_thr: float,
    high_thr: float,
    min_called: int,
    min_extreme_fraction: float,
    min_direction_match_fraction: float,
    min_consensus_variants: int,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Call per-cell support for one LOH arm against its clone consensus.

    Consensus is learnt independently for each variant as its dominant extreme
    direction. A cell supports the arm only when it has enough callable
    variants, is LOH-extreme across the arm block, and agrees with the
    variant-wise direction signature. Mid-range calls count against support;
    missing values do not.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return np.zeros(X.shape[0] if X.ndim == 2 else 0, dtype=bool), [], np.zeros(0, dtype=np.int8)

    states = np.zeros(X.shape, dtype=np.int8)
    states[np.isfinite(X) & (X <= float(low_thr))] = -1
    states[np.isfinite(X) & (X >= float(high_thr))] = 1

    consensus = np.zeros(X.shape[1], dtype=np.int8)
    for j in range(X.shape[1]):
        n_low = int(np.sum(states[:, j] == -1))
        n_high = int(np.sum(states[:, j] == 1))
        if n_low == 0 and n_high == 0:
            continue
        consensus[j] = -1 if n_low >= n_high else 1

    usable_consensus = consensus != 0
    if int(usable_consensus.sum()) < int(min_consensus_variants):
        return np.zeros(X.shape[0], dtype=bool), [
            {
                "n_called": int(np.isfinite(X[i]).sum()),
                "extreme_fraction": float("nan"),
                "direction_match_fraction": float("nan"),
                "n_consensus_called": 0,
                "support": False,
            }
            for i in range(X.shape[0])
        ], consensus

    support = np.zeros(X.shape[0], dtype=bool)
    metrics: list[dict] = []
    for i in range(X.shape[0]):
        finite = np.isfinite(X[i])
        n_called = int(finite.sum())
        n_extreme = int(np.sum(states[i, finite] != 0))
        extreme_fraction = float(n_extreme / n_called) if n_called else float("nan")
        direction_mask = finite & usable_consensus
        n_direction = int(direction_mask.sum())
        if n_direction:
            direction_match = float(np.mean(states[i, direction_mask] == consensus[direction_mask]))
        else:
            direction_match = float("nan")
        passes = (
            n_called >= int(min_called)
            and n_direction >= int(min_called)
            and np.isfinite(extreme_fraction)
            and extreme_fraction >= float(min_extreme_fraction)
            and np.isfinite(direction_match)
            and direction_match >= float(min_direction_match_fraction)
        )
        support[i] = bool(passes)
        metrics.append({
            "n_called": n_called,
            "extreme_fraction": extreme_fraction,
            "direction_match_fraction": direction_match,
            "n_consensus_called": n_direction,
            "support": bool(passes),
        })
    return support, metrics, consensus


def _existing_or_composed_multi_arm_label(
    *,
    base: str,
    combo_arms: tuple[str, ...],
    ordered_tokens: list[tuple[str, str]],
    state: "PipelineState",
) -> str:
    """Reuse an exact active arm-set label when available, otherwise compose one."""
    from .loh_config import extract_chr_arms, get_base_clone

    combo_set = set(combo_arms)
    matches = [
        str(label)
        for label, cells in state.clone_to_cells.items()
        if str(get_base_clone(str(label))) == str(base)
        and set(extract_chr_arms(str(label))) == combo_set
        and len(cells) > 0
    ]
    if len(matches) == 1:
        return matches[0]

    tokens = [token for arm, token in ordered_tokens if arm in combo_set]
    return f"{base} {' '.join(tokens)}".strip()


def _decompose_high_similarity_multi_arm_failure(
    *,
    clone_label: str,
    cells_in_df: list,
    X: np.ndarray,
    state: "PipelineState",
    cfg: "LOHConfig",
    parent_metrics: dict,
) -> tuple[dict[object, str], dict, list[dict]]:
    """Resolve a coherent directional-failing multi-arm clone arm by arm.

    A pooled directional failure does not automatically imply a false clone.
    When the clone has a high cell-to-centroid similarity, arm support is
    evaluated independently for every cell. The resulting arm-combination
    groups are retained only after each represented arm passes its own
    homogeneity/directional QC. Unsupported or undersized combinations remain
    in the original FP label.
    """
    from .loh_config import get_base_clone, make_fp_label, sort_chromosome_arms

    fp_label = make_fp_label(clone_label)
    default_assignment = {cell: fp_label for cell in cells_in_df}
    info = {
        "attempted": False,
        "reason": "",
        "n_recovered_groups": 0,
        "n_fp": len(cells_in_df),
        "arms": [],
        "groups": [],
        "audit_rows": [],
        "block_updates": {},
    }

    if not bool(getattr(cfg, "phase3_multi_arm_decomposition_enabled", True)):
        info["reason"] = "multi_arm_decomposition_disabled"
        return default_assignment, info, []

    parent_mean_r = float(parent_metrics.get("mean_r", float("nan")))
    min_parent_r = float(getattr(cfg, "phase3_multi_arm_decomposition_min_parent_mean_r", 0.90))
    if not np.isfinite(parent_mean_r) or parent_mean_r < min_parent_r:
        info["reason"] = f"parent_mean_r_below_gate:{parent_mean_r:.3f}<{min_parent_r:.3f}"
        return default_assignment, info, []

    ordered_tokens = _multi_arm_loh_tokens(clone_label)
    min_arms = int(getattr(cfg, "phase3_multi_arm_decomposition_min_arms", 2))
    if len(ordered_tokens) < min_arms:
        info["reason"] = "not_multi_arm"
        return default_assignment, info, []

    min_block = int(getattr(cfg, "phase3_multi_arm_decomposition_min_block_variants", 4))
    arm_blocks: dict[str, list[str]] = {}
    arm_support: dict[str, np.ndarray] = {}
    arm_support_metrics: dict[str, list[dict]] = {}
    arm_qc: dict[str, dict] = {}

    for arm, token in ordered_tokens:
        block = _multi_arm_block_for_arm(
            clone_label=clone_label,
            cells_in_df=cells_in_df,
            arm=arm,
            token=token,
            state=state,
            cfg=cfg,
            min_variants=min_block,
        )
        if len(block) < min_block:
            info["reason"] = f"{arm}:block_too_short:{len(block)}<{min_block}"
            return default_assignment, info, []
        arm_X = state.df_numeric.loc[cells_in_df, block].values.astype(float)
        support, per_cell_metrics, consensus = _multi_arm_consensus_support(
            arm_X,
            low_thr=cfg.low_allele_threshold,
            high_thr=cfg.high_allele_threshold,
            min_called=min_block,
            min_extreme_fraction=float(getattr(cfg, "phase3_multi_arm_decomposition_min_cell_extreme_fraction", 0.70)),
            min_direction_match_fraction=float(getattr(cfg, "phase3_multi_arm_decomposition_min_direction_match_fraction", 0.75)),
            min_consensus_variants=int(getattr(cfg, "phase3_multi_arm_decomposition_min_consensus_variants", min_block)),
        )
        arm_blocks[arm] = block
        arm_support[arm] = support
        arm_support_metrics[arm] = per_cell_metrics
        arm_qc[arm] = {
            "token": token,
            "n_block_variants": len(block),
            "n_consensus_variants": int(np.sum(consensus != 0)),
            "n_support": int(np.sum(support)),
        }

    info["attempted"] = True
    info["arms"] = [arm for arm, _ in ordered_tokens]

    # Partition cells by their complete supported-arm set. No arm is silently
    # dropped: cells that support no arm are retained as possible LOH / FP.
    grouped_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)
    arm_order = [arm for arm, _ in ordered_tokens]
    for i in range(len(cells_in_df)):
        combo = tuple(arm for arm in arm_order if bool(arm_support[arm][i]))
        grouped_rows[combo].append(i)

    base = get_base_clone(clone_label)
    min_cells_cfg = getattr(cfg, "phase3_multi_arm_decomposition_min_cells", None)
    min_cells = int(cfg.min_final_clone_size if min_cells_cfg is None else min_cells_cfg)
    require_arm_homogeneity = bool(
        getattr(cfg, "phase3_multi_arm_decomposition_require_per_arm_homogeneity", True)
    )

    assignments = dict(default_assignment)
    recovered_groups: list[dict] = []
    lineage_records: list[dict] = []

    for combo, row_idx in sorted(grouped_rows.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        group_cells = [cells_in_df[i] for i in row_idx]
        status = "retain_fp_no_arm_support"
        new_label = fp_label
        validation: dict[str, dict] = {}

        if combo and len(group_cells) >= min_cells:
            arm_passes = True
            union_block: list[str] = []
            for arm in combo:
                arm_X = state.df_numeric.loc[group_cells, arm_blocks[arm]].values.astype(float)
                metrics = _homogeneity_metrics(
                    arm_X,
                    arm_blocks[arm],
                    state,
                    cfg,
                    min_vars=min(3, len(arm_blocks[arm])),
                )
                validation[arm] = metrics
                union_block.extend(arm_blocks[arm])
                if require_arm_homogeneity and not bool(metrics.get("passes", False)):
                    arm_passes = False

            if arm_passes:
                combo_sorted = tuple(sort_chromosome_arms(list(combo)))
                new_label = _existing_or_composed_multi_arm_label(
                    base=base,
                    combo_arms=combo_sorted,
                    ordered_tokens=ordered_tokens,
                    state=state,
                )
                for cell in group_cells:
                    assignments[cell] = new_label
                union_block = list(dict.fromkeys(str(v) for v in union_block))
                info["block_updates"][new_label] = union_block
                recovered_groups.append({
                    "label": new_label,
                    "arms": combo_sorted,
                    "n_cells": len(group_cells),
                    "block_variants": union_block,
                    "validation": validation,
                })
                status = "recovered_arm_combination"
            else:
                status = "retain_fp_per_arm_homogeneity_fail"
        elif combo:
            status = f"retain_fp_group_too_small:{len(group_cells)}<{min_cells}"

        for i in row_idx:
            cell = cells_in_df[i]
            row = {
                "source_label": str(clone_label),
                "cell_id": str(cell),
                "assignment": status,
                "new_label": new_label,
                "supported_arms": ";".join(combo),
                "n_supported_arms": len(combo),
                "parent_mean_r": parent_mean_r,
            }
            for arm in arm_order:
                metric = arm_support_metrics[arm][i]
                row[f"{arm}_support"] = bool(metric["support"])
                row[f"{arm}_n_called"] = int(metric["n_called"])
                row[f"{arm}_extreme_fraction"] = metric["extreme_fraction"]
                row[f"{arm}_direction_match"] = metric["direction_match_fraction"]
            info["audit_rows"].append(row)

    # Mark parent→child arm-subset relations, so generic Phase 4d cannot
    # collapse a validated parent into its child on a later iterative pass.
    seen_lineages: set[tuple[str, str]] = set()
    for parent in recovered_groups:
        parent_set = set(parent["arms"])
        for child in recovered_groups:
            child_set = set(child["arms"])
            if parent["label"] == child["label"] or not parent_set < child_set:
                continue
            pair = (str(parent["label"]), str(child["label"]))
            if pair in seen_lineages:
                continue
            seen_lineages.add(pair)
            lineage_records.append({
                "parent_label": pair[0],
                "child_label": pair[1],
                "shared_arms": sorted(parent_set),
                "child_only_arms": sorted(child_set - parent_set),
                "n_parent": int(parent["n_cells"]),
                "n_child": int(child["n_cells"]),
                "discovery": "phase3a_multi_arm_decomposition",
            })

    info["groups"] = recovered_groups
    info["n_recovered_groups"] = len(recovered_groups)
    info["n_fp"] = int(sum(1 for x in assignments.values() if x == fp_label))
    info["reason"] = "decomposed" if recovered_groups else "no_arm_combination_passed_retest"
    return assignments, info, lineage_records



def _longest_contiguous_variant_run(
    variants: list[str],
    arm_order: list[str],
) -> list[str]:
    """Return the longest contiguous run of *variants* in ordered arm variants."""
    wanted = {str(v) for v in variants}
    ordered = [str(v) for v in arm_order if str(v) in wanted]
    if not ordered:
        return []

    positions = {str(v): i for i, v in enumerate(arm_order)}
    best: list[str] = []
    run: list[str] = []
    previous = None
    for v in ordered:
        pos = positions[str(v)]
        if previous is None or pos == previous + 1:
            run.append(str(v))
        else:
            if len(run) > len(best):
                best = run
            run = [str(v)]
        previous = pos
    if len(run) > len(best):
        best = run
    return best


def _multi_arm_phase1_block_for_arm(
    *,
    clone_label: str,
    arm: str,
    token: str,
    state: "PipelineState",
    min_variants: int,
) -> list[str]:
    """Resolve one arm's contiguous Phase-1 LOH block without re-inferring it.

    Phase 3 must validate the event discovered in Phase 1. It therefore uses
    the persisted block for this clone and arm; a cytoband-token parse is only
    a metadata fallback. It does not use a whole arm or a newly inferred core.
    """
    from .loh_config import arm_variants

    arm_order = [
        str(v) for v in arm_variants(str(arm), state)
        if str(v) in state.df_numeric.columns
    ]
    if not arm_order:
        return []
    arm_set = set(arm_order)

    block_map = getattr(state, "clone_block_variants", {}) or {}
    persisted = [
        str(v) for v in block_map.get(str(clone_label), [])
        if str(v) in arm_set
    ]
    block = _longest_contiguous_variant_run(persisted, arm_order)
    if len(block) >= int(min_variants):
        return block

    parsed = _loh_block_variants(str(token), state) or []
    parsed = [str(v) for v in parsed if str(v) in arm_set]
    block = _longest_contiguous_variant_run(parsed, arm_order)
    if len(block) >= int(min_variants):
        return block

    return []


def _multi_arm_homogeneity_metrics(
    *,
    clone_label: str,
    cells_in_df: list,
    pooled_X: np.ndarray,
    pooled_loh_vars: list[str],
    state: "PipelineState",
    cfg: "LOHConfig",
    min_vars: int = 4,
) -> dict:
    """Validate a multi-arm clone independently on each arm-specific LOH block.

    There is no pooled profile, pooled directional test, or arm-orientation
    comparison. Every represented LOH arm must independently pass homogeneity
    using its contiguous Phase-1 block of at least ``min_vars`` variants.
    """
    out = {
        "passes": False,
        "mean_r": float("nan"),
        "n_usable_r": 0,
        "dir_res": None,
        "dir_failed": False,
        "dir_ado_tolerated": False,
        "dir_ado_info": {},
        "reason": "",
        "per_arm_metrics": {},
        "per_arm_blocks": {},
        "multi_arm_directional_mode": "not_applied",
    }

    arm_tokens = _multi_arm_loh_tokens(clone_label)
    if len(arm_tokens) < 2:
        out["reason"] = "not_multi_arm"
        return out

    required_block = max(
        int(min_vars),
        int(getattr(cfg, "phase3_multi_arm_min_block_variants", 4)),
    )
    failed_arms: list[str] = []
    arm_mean_rs: list[float] = []
    total_usable = 0

    for arm, token in arm_tokens:
        arm_block = _multi_arm_phase1_block_for_arm(
            clone_label=clone_label,
            arm=arm,
            token=token,
            state=state,
            min_variants=required_block,
        )
        if len(arm_block) < required_block:
            out["per_arm_metrics"][arm] = {
                "passes": False,
                "mean_r": float("nan"),
                "n_usable_r": 0,
                "dir_res": None,
                "dir_failed": False,
                "dir_ado_tolerated": False,
                "reason": f"block_too_short:{len(arm_block)}<{required_block}",
            }
            out["per_arm_blocks"][arm] = arm_block
            failed_arms.append(arm)
            continue

        arm_X = state.df_numeric.loc[cells_in_df, arm_block].values.astype(float)
        arm_metrics = _homogeneity_metrics(
            arm_X,
            arm_block,
            state,
            cfg,
            min_vars=required_block,
            skip_directional=True,
        )
        out["per_arm_metrics"][arm] = arm_metrics
        out["per_arm_blocks"][arm] = arm_block
        total_usable += int(arm_metrics.get("n_usable_r", 0))
        if np.isfinite(arm_metrics.get("mean_r", float("nan"))):
            arm_mean_rs.append(float(arm_metrics["mean_r"]))
        if not bool(arm_metrics.get("passes", False)):
            failed_arms.append(arm)

    out["n_usable_r"] = total_usable
    # The weakest arm determines multi-arm clone homogeneity.
    if arm_mean_rs:
        out["mean_r"] = float(min(arm_mean_rs))

    if failed_arms:
        out["reason"] = "per_arm_homogeneity_fail:" + ",".join(failed_arms)
        return out

    if bool(getattr(cfg, "phase3_multi_arm_require_all_arms_pass", True)):
        out["passes"] = True
        out["reason"] = "pass"
    else:
        # Retained for backwards-compatible configuration semantics. The
        # standard configuration always requires every arm to pass.
        out["passes"] = bool(arm_mean_rs)
        out["reason"] = "pass" if out["passes"] else "no_passing_arms"
    return out

def _clone_homogeneity_check(state: "PipelineState", cfg: "LOHConfig") -> int:
    """Validate active LOH clones and retain reviewable QC failures.

    Single-arm clones retain the established correlation-plus-directional QC.
    Multi-arm clones are assessed independently on every arm-specific Phase-1
    block. No multi-arm decomposition or pooled directionality is attempted.
    A multi-arm clone with at least one passing arm and one failing arm is
    retained as ``SUB:`` by default; all-arm failures remain ``FP:``.
    """
    from .loh_config import vprint, fmt_float, make_fp_label, make_sub_label

    min_vars = max(4, int(getattr(cfg, "phase3_min_variants_for_cell_corr", 4)))
    loh_map = _loh_clone_map(state)
    if not loh_map:
        return 0

    current = state.current_clones.copy()
    next_clone_number = _next_loh_clone_number(state)
    n_changed = 0
    n_pass = 0
    n_fp = 0
    n_sub = 0
    n_split_parents = 0
    n_recovered = 0
    child_block_updates: dict[str, list[str]] = {}
    old_block_keys_to_remove: set[str] = set()

    for clone_label, cells in loh_map.items():
        cells_in_df = [c for c in cells if c in state.df_numeric.index]
        arm_tokens = _multi_arm_loh_tokens(clone_label)
        is_multi_arm = (
            bool(getattr(cfg, "phase3_multi_arm_homogeneity_per_arm", True))
            and len(arm_tokens) >= 2
        )

        loh_vars = _loh_block_variants(clone_label, state)
        var_source = "block"
        if loh_vars is None or len(loh_vars) < min_vars:
            loh_vars = _loh_arm_variants(clone_label, state)
            var_source = "arm"
        loh_vars = [str(v) for v in loh_vars if str(v) in state.df_numeric.columns]

        X = np.empty((len(cells_in_df), len(loh_vars)), dtype=float)
        if len(cells_in_df) >= 2:
            if is_multi_arm:
                # Deliberately do not use the pooled X for multi-arm QC.
                metrics = _multi_arm_homogeneity_metrics(
                    clone_label=clone_label,
                    cells_in_df=cells_in_df,
                    pooled_X=X,
                    pooled_loh_vars=loh_vars,
                    state=state,
                    cfg=cfg,
                    min_vars=min_vars,
                )
                var_source = "per-arm Phase-1 blocks"
            elif len(loh_vars) >= min_vars:
                X = state.df_numeric.loc[cells_in_df, loh_vars].values.astype(float)
                metrics = _homogeneity_metrics(
                    X, loh_vars, state, cfg, min_vars=min_vars
                )
            else:
                metrics = {
                    "passes": False,
                    "mean_r": float("nan"),
                    "n_usable_r": 0,
                    "dir_res": None,
                    "dir_failed": False,
                    "dir_ado_tolerated": False,
                    "reason": "insufficient_cells_or_variants",
                }
        else:
            metrics = {
                "passes": False,
                "mean_r": float("nan"),
                "n_usable_r": 0,
                "dir_res": None,
                "dir_failed": False,
                "dir_ado_tolerated": False,
                "reason": "insufficient_cells_or_variants",
            }

        if metrics["passes"]:
            n_pass += 1
            per_arm = metrics.get("per_arm_metrics", {})
            arm_text = ""
            if per_arm:
                arm_text = "; " + " | ".join(
                    f"{arm}:r={fmt_float(m.get('mean_r'), 3)}, "
                    f"block={len(metrics.get('per_arm_blocks', {}).get(arm, []))}v"
                    for arm, m in per_arm.items()
                )
            vprint(
                f"  {_GREEN}✓ PASS{_RESET} {clone_label}  "
                f"(n={len(cells_in_df)}, min_arm_mean_r={fmt_float(metrics['mean_r'], 3)}, "
                f"vars={var_source}, {_directional_summary(metrics)}{arm_text})",
                cfg,
            )
            continue

        reason = metrics.get("reason", "failed")
        dir_failed = bool(metrics.get("dir_failed", False))
        vprint(
            f"  {_RED}✗ FAIL{_RESET} {clone_label}  "
            f"(n={len(cells_in_df)}, reason={reason}, "
            f"mean_r={fmt_float(metrics.get('mean_r'), 3)}, "
            f"vars={var_source}, {_directional_summary(metrics)})",
            cfg,
        )

        if is_multi_arm and metrics.get("per_arm_metrics"):
            for _arm, _arm_metric in metrics["per_arm_metrics"].items():
                _arm_block_n = len(metrics.get("per_arm_blocks", {}).get(_arm, []))
                _arm_status = "PASS" if bool(_arm_metric.get("passes", False)) else "FAIL"
                vprint(
                    f"      ↳ {_arm}: {_arm_status}; "
                    f"mean_r={fmt_float(_arm_metric.get('mean_r'), 3)}; "
                    f"n_usable_r={int(_arm_metric.get('n_usable_r', 0))}; "
                    f"block={_arm_block_n}v; reason={_arm_metric.get('reason', 'NA')}",
                    cfg,
                )

        # Default hard failure is FP. A multi-arm clone with mixed arm-level
        # evidence is biologically unresolved rather than automatically false:
        # when at least one arm passes and at least one fails, retain the whole
        # cell population as review-only SUB by default. This preserves a
        # credible partial LOH signal without silently discarding the failing arm.
        per_arm_metrics = metrics.get("per_arm_metrics", {}) or {}
        passing_arms = [
            str(arm) for arm, arm_metric in per_arm_metrics.items()
            if bool(arm_metric.get("passes", False))
        ]
        failing_arms = [
            str(arm) for arm, arm_metric in per_arm_metrics.items()
            if not bool(arm_metric.get("passes", False))
        ]
        partial_multi_arm_failure = (
            bool(is_multi_arm)
            and bool(passing_arms)
            and bool(failing_arms)
            and str(reason).startswith("per_arm_homogeneity_fail:")
        )
        use_sub_for_partial_multi_arm = (
            partial_multi_arm_failure
            and bool(getattr(cfg, "phase3_multi_arm_partial_fail_use_sub_label", True))
        )

        if use_sub_for_partial_multi_arm:
            assignments = {cell: make_sub_label(clone_label) for cell in cells_in_df}
            vprint(
                f"      ↳ partial multi-arm support: "
                f"pass={','.join(passing_arms)}; fail={','.join(failing_arms)} "
                f"→ retain as SUB for review",
                cfg,
            )
        else:
            assignments = {cell: make_fp_label(clone_label) for cell in cells_in_df}

        split_info = None
        # Directional splitting is a single-arm repair only. Multi-arm labels
        # remain intact if they pass every arm's independent homogeneity test.
        if (
            dir_failed
            and not is_multi_arm
            and len(cells_in_df) >= 2
            and len(loh_vars) >= min_vars
        ):
            assignments, split_info, next_clone_number = _split_directional_failure(
                clone_label=clone_label,
                cells_in_df=cells_in_df,
                X=X,
                loh_vars=loh_vars,
                state=state,
                cfg=cfg,
                next_clone_number=next_clone_number,
            )

        if split_info is not None:
            if split_info.get("attempted", False):
                n_split_parents += 1
            for child in split_info.get("children", []):
                child_block_updates[child["label"]] = list(child["block_variants"])
                n_recovered += 1
                child_metrics = child["metrics"]
                vprint(
                    f"      ↳ {_GREEN}RECOVERED{_RESET} {child['label']} "
                    f"(n={child['n_cells']}, mean_r={fmt_float(child_metrics['mean_r'], 3)}, "
                    f"{_directional_summary(child_metrics)})",
                    cfg,
                )
            vprint(
                f"      ↳ split={split_info.get('reason')} "
                f"(discordant_variants={split_info.get('n_discordant_variants', 0)}, "
                f"recovered={split_info.get('n_recovered_subclones', 0)}, "
                f"SUB={split_info.get('n_sub', 0)}, FP={split_info.get('n_fp', 0)})",
                cfg,
            )

        fp_label = make_fp_label(clone_label)
        sub_label = make_sub_label(clone_label)
        fp_for_parent = 0
        sub_for_parent = 0
        for cell, new_label in assignments.items():
            bc = state.cell_to_barcode.get(cell, str(cell))
            idx = state.sample_idx_for_barcode(str(bc))
            if idx is None:
                continue
            if str(current.iloc[idx]) != str(new_label):
                current.iloc[idx] = new_label
                n_changed += 1
            if new_label == fp_label:
                fp_for_parent += 1
            elif new_label == sub_label:
                sub_for_parent += 1
        n_fp += fp_for_parent
        n_sub += sub_for_parent
        old_block_keys_to_remove.add(clone_label)
        if sub_for_parent:
            vprint(
                f"      ↳ {_YELLOW}SUB RETAINED{_RESET} {sub_for_parent} cell(s) as '{sub_label}' "
                "(unresolved directional subclone)",
                cfg,
            )
        if fp_for_parent:
            vprint(
                f"      ↳ {_YELLOW}FP RETAINED{_RESET} {fp_for_parent} cell(s) as '{fp_label}'",
                cfg,
            )

    if n_changed:
        state.set_clone_labels(current)

    block_map = getattr(state, "clone_block_variants", None)
    if block_map is not None:
        for old_label in old_block_keys_to_remove:
            block_map.pop(old_label, None)
        block_map.update(child_block_updates)

    vprint(
        f"  Homogeneity summary: passed={n_pass}; directional parents split={n_split_parents}; "
        f"recovered LOH subclones={n_recovered}; retained SUB cells={n_sub}; retained FP cells={n_fp}.",
        cfg,
    )
    return max(n_changed, n_recovered)


# ═══════════════════════════════════════════════════════════════════
# Step 3b: Automatic nested parent-child LOH rescue
# ═══════════════════════════════════════════════════════════════════

def _nested_profile_metrics(
    observed: pd.Series,
    reference: pd.Series,
    *,
    low_af: float,
    high_af: float,
) -> dict:
    """Profile similarity plus directional extreme agreement."""
    observed = pd.to_numeric(observed, errors="coerce")
    reference = pd.to_numeric(reference, errors="coerce")
    keep = observed.notna() & reference.notna()

    if int(keep.sum()) == 0:
        return {
            "n_called": 0,
            "r": float("nan"),
            "median_abs_diff": float("nan"),
            "extreme_match": float("nan"),
            "observed_range": float("nan"),
            "reference_range": float("nan"),
        }

    observed = observed.loc[keep]
    reference = reference.loc[keep]
    x = observed.to_numpy(dtype=float)
    y = reference.to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) == 0:
        return {
            "n_called": 0,
            "r": float("nan"),
            "median_abs_diff": float("nan"),
            "extreme_match": float("nan"),
            "observed_range": float("nan"),
            "reference_range": float("nan"),
        }

    observed_range = float(np.nanmax(x) - np.nanmin(x))
    reference_range = float(np.nanmax(y) - np.nanmin(y))

    if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = float("nan")

    ref_low = y <= float(low_af)
    ref_high = y >= float(high_af)
    informative = ref_low | ref_high
    if int(np.sum(informative)) > 0:
        direction_match = (
            (ref_low & (x <= float(low_af)))
            | (ref_high & (x >= float(high_af)))
        )
        extreme_match = float(np.mean(direction_match[informative]))
    else:
        extreme_match = float("nan")

    return {
        "n_called": int(len(x)),
        "r": r,
        "median_abs_diff": float(np.nanmedian(np.abs(x - y))),
        "extreme_match": extreme_match,
        "observed_range": observed_range,
        "reference_range": reference_range,
    }


def _nested_profile_passes(
    metrics: dict,
    *,
    min_r: float,
    max_median_abs_diff: float,
    min_extreme_match: float,
    min_called_variants: int,
    min_profile_dynamic_range_for_corr: float,
) -> bool:
    """Apply a conservative profile gate for automatic nested-rescue calls."""
    if int(metrics.get("n_called", 0)) < int(min_called_variants):
        return False

    median_abs_diff = float(metrics.get("median_abs_diff", float("nan")))
    if not np.isfinite(median_abs_diff) or median_abs_diff > float(max_median_abs_diff):
        return False

    r = float(metrics.get("r", float("nan")))
    observed_range = float(metrics.get("observed_range", float("nan")))
    reference_range = float(metrics.get("reference_range", float("nan")))
    min_dynamic_range = float(min_profile_dynamic_range_for_corr)

    # On short, near-constant LOH blocks correlation can be non-informative.
    # In that setting AF distance and matching extreme direction remain strict.
    correlation_informative = (
        np.isfinite(observed_range)
        and np.isfinite(reference_range)
        and observed_range >= min_dynamic_range
        and reference_range >= min_dynamic_range
    )
    if correlation_informative and np.isfinite(r) and r < float(min_r):
        return False

    extreme_match = float(metrics.get("extreme_match", float("nan")))
    if np.isfinite(extreme_match) and extreme_match < float(min_extreme_match):
        return False

    return True


def _nested_arm_variants(arm: str, state: "PipelineState") -> list[str]:
    """Return ordered, available informative variants for one chromosome arm."""
    from .loh_config import arm_variants

    return list(dict.fromkeys(
        str(v)
        for v in arm_variants(str(arm), state)
        if str(v) in state.df_numeric.columns
    ))


def _nested_child_block_for_arm(
    *,
    child_label: str,
    child_cells: list,
    arm: str,
    state: "PipelineState",
    cfg: "LOHConfig",
) -> tuple[list[str], pd.DataFrame]:
    """Return a persisted or inferred dominant-extreme child block for one arm."""
    arm_vars = _nested_arm_variants(arm, state)
    if not arm_vars:
        raise RuntimeError(f"no informative variants available for {arm}")

    arm_set = set(arm_vars)
    block_map = getattr(state, "clone_block_variants", {}) or {}
    persisted = list(dict.fromkeys(
        str(v)
        for v in block_map.get(str(child_label), [])
        if str(v) in arm_set
    ))

    child_cells = [c for c in child_cells if c in state.df_numeric.index]
    if not child_cells:
        raise RuntimeError("child has no cells in the AF matrix")

    child_af = state.df_numeric.loc[child_cells, arm_vars].apply(
        pd.to_numeric,
        errors="coerce",
    )
    low_fraction = (child_af <= float(cfg.low_allele_threshold)).mean(axis=0)
    high_fraction = (child_af >= float(cfg.high_allele_threshold)).mean(axis=0)
    dominant_fraction = pd.concat([low_fraction, high_fraction], axis=1).max(axis=1)

    qc = pd.DataFrame({
        "variant": arm_vars,
        "low_fraction": low_fraction.reindex(arm_vars).to_numpy(),
        "high_fraction": high_fraction.reindex(arm_vars).to_numpy(),
        "dominant_fraction": dominant_fraction.reindex(arm_vars).to_numpy(),
    })
    qc["passes"] = (
        qc["dominant_fraction"]
        >= float(cfg.phase3_nested_loh_rescue_min_child_extreme_fraction)
    )
    qc["block_source"] = "inferred"

    if len(persisted) >= int(cfg.phase3_nested_loh_rescue_min_block_variants):
        qc.loc[qc["variant"].isin(persisted), "block_source"] = "phase1_persisted"
        return persisted, qc

    best_start = None
    best_end = None
    run_start = None
    for i, passed in enumerate(qc["passes"].to_numpy(dtype=bool)):
        if passed and run_start is None:
            run_start = i
        if (not passed) and run_start is not None:
            run_end = i - 1
            if (
                best_start is None
                or (run_end - run_start) > (best_end - best_start)
            ):
                best_start, best_end = run_start, run_end
            run_start = None

    if run_start is not None:
        run_end = len(qc) - 1
        if (
            best_start is None
            or (run_end - run_start) > (best_end - best_start)
        ):
            best_start, best_end = run_start, run_end

    if best_start is None:
        raise RuntimeError("no dominant-extreme child block detected")

    inferred = qc.iloc[best_start:best_end + 1]["variant"].astype(str).tolist()
    if len(inferred) < int(cfg.phase3_nested_loh_rescue_min_block_variants):
        raise RuntimeError(
            f"inferred block contains {len(inferred)} variants; requires "
            f"{cfg.phase3_nested_loh_rescue_min_block_variants}"
        )

    return inferred, qc


def _fp_parent_label_info(fp_label: str) -> tuple[str, str, tuple[str, ...]]:
    """Recover base genotype, active parent label, and arm set from an FP label."""
    from .loh_config import extract_chr_arms, sort_chromosome_arms

    tail = re.sub(r"^\s*FP:\s*", "", str(fp_label)).strip()
    parent_label = re.sub(
        r"\bC(\d+)\s+(chr(?:\d+|X|Y)[pq](?::[^\s]+)?)",
        r"LOH-C\1 \2",
        tail,
    )
    base = re.sub(
        r"\bC\d+\s+chr(?:\d+|X|Y)[pq](?::[^\s]+)?",
        "",
        tail,
    )
    base = re.sub(r"\s+", " ", base).strip()
    arms = tuple(sort_chromosome_arms(list(dict.fromkeys(extract_chr_arms(tail)))))
    return base, parent_label, arms


def _active_loh_label_info(label: str) -> tuple[str, tuple[str, ...]]:
    """Return base genotype and sorted active LOH arm set from a child label."""
    from .loh_config import extract_chr_arms, get_base_clone, sort_chromosome_arms

    base = get_base_clone(str(label))
    arms = tuple(sort_chromosome_arms(list(dict.fromkeys(extract_chr_arms(label)))))
    return base, arms


def _automatic_nested_candidates(
    state: "PipelineState",
    cfg: "LOHConfig",
) -> tuple[list[dict], list[dict]]:
    """Discover unambiguous same-base FP-parent / active-child relationships."""
    from .loh_config import is_fp_label, is_loh_label

    current_counts = state.current_clones.value_counts()
    min_child_cells = int(cfg.min_final_clone_size)

    children = []
    for label, cells in state.clone_to_cells.items():
        if not is_loh_label(label) or len(cells) < min_child_cells:
            continue
        base, arms = _active_loh_label_info(str(label))
        if arms:
            children.append({
                "child_label": str(label),
                "child_base": base,
                "child_arms": arms,
                "child_cells": [c for c in cells if c in state.df_numeric.index],
            })

    candidates: list[dict] = []
    audit: list[dict] = []

    for source_label, source_cells_all in state.clone_to_cells.items():
        if not is_fp_label(source_label):
            continue

        source_cells = [c for c in source_cells_all if c in state.df_numeric.index]
        source_base, parent_label, parent_arms = _fp_parent_label_info(str(source_label))
        parent_set = set(parent_arms)

        if not parent_arms or not source_base or not source_cells:
            audit.append({
                "source_label": str(source_label),
                "status": "skipped_unparseable_fp_parent",
                "n_source": len(source_cells),
            })
            continue

        eligible = []
        for child in children:
            child_set = set(child["child_arms"])
            if child["child_base"] != source_base or not parent_set < child_set:
                continue

            extra_arms = tuple(a for a in child["child_arms"] if a not in parent_set)
            if (
                bool(cfg.phase3_nested_loh_rescue_require_single_child_only_arm)
                and len(extra_arms) != 1
            ):
                continue

            eligible.append({
                "source_label": str(source_label),
                "source_base": source_base,
                "source_cells": source_cells,
                "parent_label": parent_label,
                "parent_arms": parent_arms,
                "child_label": child["child_label"],
                "child_cells": child["child_cells"],
                "child_arms": child["child_arms"],
                "shared_arms": tuple(a for a in parent_arms if a in child_set),
                "child_only_arms": extra_arms,
            })

        if not eligible:
            audit.append({
                "source_label": str(source_label),
                "status": "skipped_no_same_base_strict_superset_child",
                "n_source": len(source_cells),
                "parent_label": parent_label,
                "parent_arms": ";".join(parent_arms),
            })
            continue

        candidates.extend(eligible)

    return candidates, audit


def _apply_nested_parent_child_rescues(
    state: "PipelineState",
    cfg: "LOHConfig",
) -> pd.DataFrame:
    """
    Automatically rescue unambiguous nested LOH lineages.

    No labels or chromosome arms are configured. An FP population is rescued
    only when a same-base active LOH child has a strict superset of LOH arms,
    the shared-arm profile matches, and one child remains unambiguous after
    profile validation. The child-only arm(s) determine per-cell promotion.
    """
    from .loh_config import vprint

    if not bool(getattr(cfg, "phase3_nested_loh_rescue_enabled", False)):
        return pd.DataFrame()

    state.refresh()
    current = state.current_clones.copy()
    block_map = getattr(state, "clone_block_variants", None)
    if block_map is None:
        state.clone_block_variants = {}
        block_map = state.clone_block_variants

    raw_candidates, discovery_audit = _automatic_nested_candidates(state, cfg)
    audit_rows: list[dict] = list(discovery_audit)
    lineage_records: list[dict] = []
    any_change = False

    # Fully validate every label-derived child candidate first. A source is
    # rescued only when exactly one child survives all shared-arm checks.
    valid_by_source: dict[str, list[dict]] = defaultdict(list)
    for candidate in raw_candidates:
        source_label = candidate["source_label"]
        child_label = candidate["child_label"]
        try:
            shared_blocks = []
            child_only_blocks = []
            shared_qc = []
            child_only_qc = []

            for arm in candidate["shared_arms"]:
                block, qc = _nested_child_block_for_arm(
                    child_label=child_label,
                    child_cells=candidate["child_cells"],
                    arm=arm,
                    state=state,
                    cfg=cfg,
                )
                shared_blocks.extend(block)
                shared_qc.append(qc.assign(arm=arm))

            for arm in candidate["child_only_arms"]:
                block, qc = _nested_child_block_for_arm(
                    child_label=child_label,
                    child_cells=candidate["child_cells"],
                    arm=arm,
                    state=state,
                    cfg=cfg,
                )
                child_only_blocks.extend(block)
                child_only_qc.append(qc.assign(arm=arm))

            shared_block = list(dict.fromkeys(shared_blocks))
            child_only_block = list(dict.fromkeys(child_only_blocks))
            if not shared_block or not child_only_block:
                raise RuntimeError("empty shared or child-only block")

            shared_reference = (
                state.df_numeric.loc[candidate["child_cells"], shared_block]
                .apply(pd.to_numeric, errors="coerce")
                .median(axis=0)
            )
            source_centroid = (
                state.df_numeric.loc[candidate["source_cells"], shared_block]
                .apply(pd.to_numeric, errors="coerce")
                .median(axis=0)
            )
            source_shared_metrics = _nested_profile_metrics(
                source_centroid,
                shared_reference,
                low_af=cfg.low_allele_threshold,
                high_af=cfg.high_allele_threshold,
            )
            source_passes = _nested_profile_passes(
                source_shared_metrics,
                min_r=cfg.phase3_nested_loh_rescue_min_shared_arm_r,
                max_median_abs_diff=cfg.phase3_nested_loh_rescue_max_shared_arm_median_abs_diff,
                min_extreme_match=cfg.phase3_nested_loh_rescue_min_shared_arm_extreme_match,
                min_called_variants=cfg.phase3_nested_loh_rescue_min_block_variants,
                min_profile_dynamic_range_for_corr=cfg.phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr,
            )
            if not source_passes:
                audit_rows.append({
                    "source_label": source_label,
                    "candidate_child_label": child_label,
                    "status": "rejected_shared_arm_centroid_nonconcordant",
                    "n_source": len(candidate["source_cells"]),
                    "parent_label": candidate["parent_label"],
                    "shared_arms": ";".join(candidate["shared_arms"]),
                    "child_only_arms": ";".join(candidate["child_only_arms"]),
                    "shared_arm_r": source_shared_metrics["r"],
                    "shared_arm_median_abs_diff": source_shared_metrics["median_abs_diff"],
                    "shared_arm_extreme_match": source_shared_metrics["extreme_match"],
                })
                continue

            candidate.update({
                "shared_block": shared_block,
                "child_only_block": child_only_block,
                "shared_reference": shared_reference,
                "child_only_reference": (
                    state.df_numeric.loc[candidate["child_cells"], child_only_block]
                    .apply(pd.to_numeric, errors="coerce")
                    .median(axis=0)
                ),
                "source_shared_metrics": source_shared_metrics,
                "shared_qc": pd.concat(shared_qc, ignore_index=True),
                "child_only_qc": pd.concat(child_only_qc, ignore_index=True),
            })
            valid_by_source[source_label].append(candidate)

        except Exception as exc:
            audit_rows.append({
                "source_label": source_label,
                "candidate_child_label": child_label,
                "status": "rejected_candidate_block_error",
                "reason": str(exc),
                "n_source": len(candidate["source_cells"]),
                "parent_label": candidate["parent_label"],
                "shared_arms": ";".join(candidate["shared_arms"]),
                "child_only_arms": ";".join(candidate["child_only_arms"]),
            })

    for source_label, valid_candidates in valid_by_source.items():
        if len(valid_candidates) != 1:
            audit_rows.append({
                "source_label": source_label,
                "status": "skipped_ambiguous_multiple_valid_children",
                "n_valid_children": len(valid_candidates),
                "candidate_child_labels": ";".join(
                    c["child_label"] for c in valid_candidates
                ),
            })
            vprint(
                f"  Nested rescue skipped for '{source_label}': "
                f"{len(valid_candidates)} valid same-base children remain ambiguous.",
                cfg,
            )
            continue

        candidate = valid_candidates[0]
        parent_label = candidate["parent_label"]
        child_label = candidate["child_label"]
        promoted_to_child: list = []
        retained_as_parent: list = []
        retained_as_fp: list = []

        for cell in candidate["source_cells"]:
            shared_metrics = _nested_profile_metrics(
                state.df_numeric.loc[cell, candidate["shared_block"]],
                candidate["shared_reference"],
                low_af=cfg.low_allele_threshold,
                high_af=cfg.high_allele_threshold,
            )
            child_only_metrics = _nested_profile_metrics(
                state.df_numeric.loc[cell, candidate["child_only_block"]],
                candidate["child_only_reference"],
                low_af=cfg.low_allele_threshold,
                high_af=cfg.high_allele_threshold,
            )

            shared_pass = _nested_profile_passes(
                shared_metrics,
                min_r=cfg.phase3_nested_loh_rescue_min_shared_arm_r,
                max_median_abs_diff=cfg.phase3_nested_loh_rescue_max_shared_arm_median_abs_diff,
                min_extreme_match=cfg.phase3_nested_loh_rescue_min_shared_arm_extreme_match,
                min_called_variants=cfg.phase3_nested_loh_rescue_min_block_variants,
                min_profile_dynamic_range_for_corr=cfg.phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr,
            )
            child_only_pass = _nested_profile_passes(
                child_only_metrics,
                min_r=cfg.phase3_nested_loh_rescue_min_child_only_arm_r,
                max_median_abs_diff=cfg.phase3_nested_loh_rescue_max_child_only_arm_median_abs_diff,
                min_extreme_match=cfg.phase3_nested_loh_rescue_min_child_only_arm_extreme_match,
                min_called_variants=cfg.phase3_nested_loh_rescue_min_child_only_arm_called_variants,
                min_profile_dynamic_range_for_corr=cfg.phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr,
            )

            if not shared_pass:
                assignment = "remain_fp_shared_arm_nonconcordant"
                new_label = source_label
                retained_as_fp.append(cell)
            elif child_only_pass:
                assignment = "promote_to_child"
                new_label = child_label
                promoted_to_child.append(cell)
            else:
                assignment = "retain_shared_arm_parent"
                new_label = parent_label
                retained_as_parent.append(cell)

            audit_rows.append({
                "source_label": source_label,
                "parent_label": parent_label,
                "child_label": child_label,
                "status": "assigned",
                "barcode": str(state.cell_to_barcode.get(cell, str(cell))),
                "cell_id": str(cell),
                "assignment": assignment,
                "new_label": new_label,
                "shared_arms": ";".join(candidate["shared_arms"]),
                "shared_block_variants": ";".join(candidate["shared_block"]),
                "shared_arm_r": shared_metrics["r"],
                "shared_arm_median_abs_diff": shared_metrics["median_abs_diff"],
                "shared_arm_extreme_match": shared_metrics["extreme_match"],
                "shared_arm_n_called": shared_metrics["n_called"],
                "child_only_arms": ";".join(candidate["child_only_arms"]),
                "child_only_block_variants": ";".join(candidate["child_only_block"]),
                "child_only_arm_r": child_only_metrics["r"],
                "child_only_arm_median_abs_diff": child_only_metrics["median_abs_diff"],
                "child_only_arm_extreme_match": child_only_metrics["extreme_match"],
                "child_only_arm_n_called": child_only_metrics["n_called"],
                "source_shared_centroid_r": candidate["source_shared_metrics"]["r"],
                "source_shared_centroid_median_abs_diff": candidate["source_shared_metrics"]["median_abs_diff"],
                "source_shared_centroid_extreme_match": candidate["source_shared_metrics"]["extreme_match"],
            })

        min_parent_cells = cfg.phase3_nested_loh_rescue_min_retained_parent_cells
        if min_parent_cells is None:
            min_parent_cells = cfg.min_final_clone_size
        if len(retained_as_parent) < int(min_parent_cells):
            for cell in retained_as_parent:
                retained_as_fp.append(cell)
                for row in audit_rows:
                    if row.get("cell_id") == str(cell) and row.get("assignment") == "retain_shared_arm_parent":
                        row["assignment"] = "remain_fp_parent_too_small"
                        row["new_label"] = source_label
                        break
            retained_as_parent = []

        for cell in retained_as_parent:
            bc = state.cell_to_barcode.get(cell, str(cell))
            idx = state.sample_idx_for_barcode(str(bc))
            if idx is not None and str(current.iloc[idx]) != parent_label:
                current.iloc[idx] = parent_label
                any_change = True

        for cell in promoted_to_child:
            bc = state.cell_to_barcode.get(cell, str(cell))
            idx = state.sample_idx_for_barcode(str(bc))
            if idx is not None and str(current.iloc[idx]) != child_label:
                current.iloc[idx] = child_label
                any_change = True

        if retained_as_parent:
            block_map[parent_label] = list(candidate["shared_block"])
            existing_child_block = list(block_map.get(child_label, []))
            block_map[child_label] = list(dict.fromkeys(
                existing_child_block
                + list(candidate["shared_block"])
                + list(candidate["child_only_block"])
            ))
            lineage_records.append({
                "parent_label": parent_label,
                "child_label": child_label,
                "shared_arms": list(candidate["shared_arms"]),
                "child_only_arms": list(candidate["child_only_arms"]),
                "n_source": len(candidate["source_cells"]),
                "n_parent": len(retained_as_parent),
                "n_child": len(promoted_to_child),
                "n_fp": len(retained_as_fp),
                "shared_block_variants": list(candidate["shared_block"]),
                "child_only_block_variants": list(candidate["child_only_block"]),
                "discovery": "automatic_same_base_strict_arm_subset",
            })

        vprint(
            f"  Nested rescue: '{source_label}' → parent={len(retained_as_parent)}, "
            f"child={len(promoted_to_child)}, FP={len(retained_as_fp)} "
            f"(shared={','.join(candidate['shared_arms'])}, "
            f"child-only={','.join(candidate['child_only_arms'])}).",
            cfg,
        )

    if any_change:
        state.set_clone_labels(current)

    if lineage_records:
        existing = list(getattr(state, "nested_loh_lineages", []) or [])
        existing_pairs = {
            (str(x.get("parent_label")), str(x.get("child_label")))
            for x in existing
            if isinstance(x, dict)
        }
        for record in lineage_records:
            pair = (record["parent_label"], record["child_label"])
            if pair not in existing_pairs:
                existing.append(record)
                existing_pairs.add(pair)
        state.nested_loh_lineages = existing

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        state.results["phase3_nested_loh_rescue"] = audit_df
        state.nested_loh_rescue_audit = audit_df

    return audit_df


# ═══════════════════════════════════════════════════════════════════
# Step 4: Discard small LOH clones
# ═══════════════════════════════════════════════════════════════════

def _discard_small_loh_clones(state: "PipelineState", cfg: "LOHConfig") -> int:
    """Revert LOH clones with fewer than min_final_clone_size cells."""
    from .loh_config import vprint, get_base_clone, is_loh_label

    min_cells = int(cfg.min_final_clone_size)
    counts = state.current_clones.value_counts()
    small_labels = [
        lbl for lbl, n in counts.items()
        if is_loh_label(lbl) and int(n) < min_cells
    ]

    if not small_labels:
        return 0

    for lbl in small_labels:
        vprint(f"  {_RED}{lbl}  (n={int(counts[lbl])}) ✗{_RESET}", cfg)

    current = state.current_clones.copy()
    small_set = set(map(str, small_labels))
    n_changed = 0
    for i in range(len(current)):
        if str(current.iloc[i]) in small_set:
            current.iloc[i] = get_base_clone(current.iloc[i])
            n_changed += 1
    state.set_clone_labels(current)
    return n_changed


# ═══════════════════════════════════════════════════════════════════
# Step 4b: Flag reverted LOH outliers from non-LOH clones
# ═══════════════════════════════════════════════════════════════════

def _remove_reverted_loh_outliers(
    state: "PipelineState",
    cfg: "LOHConfig",
    loh_label_before_by_barcode: dict[str, str],
) -> int:
    """
    Flag reverted-but-still-extreme cells without deleting them from matrices.

    The cell is retained as a separate ``FP: <assigned LOH clone>`` label.
    Existing FP labels are intentionally left untouched.
    """
    from .loh_config import (
        vprint, arm_variants, extract_chr_arms,
        is_loh_label, is_fp_label, is_sub_label, make_fp_label,
    )

    low = cfg.low_allele_threshold
    high = cfg.high_allele_threshold
    min_extreme = cfg.min_extreme_fraction

    all_loh_arms: list[str] = []
    for arm_key in list(state.results.keys()):
        if str(arm_key).startswith("phase"):
            continue
        all_loh_arms.extend(re.findall(r"chr(?:\d+|X|Y)[pq]", str(arm_key)))
    for prior_label in loh_label_before_by_barcode.values():
        all_loh_arms.extend(extract_chr_arms(str(prior_label)))
    all_loh_arms = list(dict.fromkeys(all_loh_arms))
    if not all_loh_arms:
        return 0

    fp_assignments: dict[str, str] = {}
    for bc, prior_label in loh_label_before_by_barcode.items():
        current_label = state.barcode_to_clone.get(str(bc), "")
        if is_loh_label(current_label) or is_fp_label(current_label) or is_sub_label(current_label):
            continue
        if str(bc) not in state.df.index:
            continue

        for arm in all_loh_arms:
            arm_vars = arm_variants(arm, state)
            if len(arm_vars) < cfg.min_contiguous_variants_initial:
                continue
            vals = state.df.loc[str(bc), arm_vars].values.astype(float)
            valid = vals[np.isfinite(vals)]
            if len(valid) < cfg.min_contiguous_variants_initial:
                continue
            ef = float(np.mean((valid <= low) | (valid >= high)))
            if ef >= min_extreme:
                fp_assignments[str(bc)] = make_fp_label(prior_label)
                vprint(
                    f"    {bc}: extreme={ef:.0%} on {arm} → retained as '{fp_assignments[str(bc)]}'",
                    cfg,
                )
                break

    if not fp_assignments:
        return 0

    current = state.current_clones.copy()
    for bc, fp_label in fp_assignments.items():
        idx = state.sample_idx_for_barcode(bc)
        if idx is not None:
            current.iloc[idx] = fp_label
    state.set_clone_labels(current)

    n_fp_labels = len(set(fp_assignments.values()))
    vprint(
        f"    Retained {len(fp_assignments)} reverted-LOH outlier(s) in {n_fp_labels} "
        f"per-clone FP label(s); no rows removed from state matrices.",
        cfg,
    )
    return len(fp_assignments)


# ═══════════════════════════════════════════════════════════════════
# Step 5: Results reconciliation
# ═══════════════════════════════════════════════════════════════════

def _reconcile_results(state: "PipelineState", cfg: "LOHConfig") -> None:
    """Prune state.results to match surviving labels."""
    from .loh_config import vprint, LOH_TAG, is_loh_label

    results = state.results

    for key in list(results.keys()):
        if str(key).startswith("phase"):
            continue

        df_segments, df_loh, df_corr, df_homog = results[key]

        if len(df_loh) == 0:
            del results[key]
            continue

        new_labels = []
        for cell_id in df_loh.index:
            bc = state.cell_to_barcode.get(cell_id, str(cell_id))
            new_labels.append(state.barcode_to_clone.get(str(bc), ""))
        df_loh["label"] = new_labels

        key_arms = re.findall(r"chr(?:\d+|X|Y)[pq]", key)

        def _label_matches(lbl):
            if not is_loh_label(lbl):
                return False
            return any(arm in str(lbl) for arm in key_arms)

        keep_mask = df_loh["label"].apply(_label_matches)
        df_loh = df_loh.loc[keep_mask].copy()

        if len(df_segments) > 0 and "cell" in df_segments.columns:
            df_segments = df_segments.loc[
                df_segments["cell"].isin(set(df_loh.index))
            ].copy()

        if len(df_loh) == 0:
            vprint(f"  {key}: removed (no cells retain LOH).", cfg)
            del results[key]
        else:
            old_n = len(results[key][1])
            if len(df_loh) < old_n:
                vprint(f"  {key}: {old_n} → {len(df_loh)} cells retained.", cfg)
            results[key] = (df_segments, df_loh, df_corr, df_homog)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_phase3(
    state: "PipelineState",
    cfg: "LOHConfig | None" = None,
) -> dict:
    """Phase 3: LOH clone QC."""
    from .loh_config import vprint

    if cfg is None:
        cfg = state.config

    # Reproducibility.
    import random
    np.random.seed(123)
    random.seed(123)

    results = state.results

    # ── Skip path ──
    if cfg.skip_homogeneity_check:
        vprint("\n" + "=" * 60, cfg)
        vprint("PHASE 3: LOH Clone QC — SKIPPED  (skip_homogeneity_check=True)", cfg)
        vprint("=" * 60, cfg)
        vprint("\n  Reconciling results with current labels …", cfg)
        _reconcile_results(state, cfg)
        vprint(f"  Result keys: {sorted(results.keys())}", cfg)
        vprint(f"  LOH labels:  {state.loh_clone_labels()}", cfg)
        vprint(f"  LOH cells:   {state.loh_cell_count()}", cfg)
        return results

    # ── Full path ──
    vprint("\n" + "=" * 60, cfg)
    vprint("PHASE 3: LOH Clone QC", cfg)
    vprint("=" * 60, cfg)
    vprint(
        f"  Step 1  Per-cell outlier removal: mean − "
        f"{getattr(cfg, 'phase3_outlier_k', 2.0):.1f}×std",
        cfg,
    )
    vprint(
        f"  Step 2  Rescue: small LOH → compatible LOH "
        f"(r ≥ {cfg.post_merge_correlation_threshold:.2f})",
        cfg,
    )

    ado_dir_msg = ""
    if bool(getattr(cfg, "phase3_allow_single_ado_amplicon_direction_fail", False)):
        ado_dir_msg = (
            f"; allow ≤{getattr(cfg, 'phase3_direction_ado_max_bad_amplicons', 1)} "
            f"bad amplicon if block has ≥"
            f"{getattr(cfg, 'phase3_direction_ado_min_block_variants', 6)} "
            f"evaluated variants"
        )

    vprint(
        f"  Step 3  Clone homogeneity: each single-arm LOH block uses mean cell-to-mean r "
        f"≥ {cfg.phase3_min_mean_corr_to_mean:.2f}; multi-arm clones require every arm-specific "
        f"Phase-1 block (≥4 contiguous variants) to pass independently; "
        f"single-arm directional failures split before per-clone FP labelling{ado_dir_msg}",
        cfg,
    )
    if bool(getattr(cfg, "phase3_nested_loh_rescue_enabled", False)):
        vprint(
            "  Step 3b Automatic nested parent-child rescue: discover same-base "
            "FP parents with strict-subset LOH arms and partition them by child-only arms.",
            cfg,
        )
    vprint(
        f"  Step 4  Discard small LOH clones "
        f"(< {cfg.min_final_clone_size} cells)",
        cfg,
    )
    vprint("  Step 5  Results reconciliation", cfg)

    loh_before = state.loh_cell_count()
    labels_before = len(state.loh_clone_labels())
    vprint(f"\n  LOH clones before: {labels_before}  ({loh_before} cells)", cfg)

    # Snapshot each original active LOH assignment before any QC relabelling.
    from .loh_config import is_loh_label as _is_loh_label
    _loh_label_before_by_barcode: dict[str, str] = {}
    for cl, cells in state.clone_to_cells.items():
        if _is_loh_label(cl):
            for cid in cells:
                bc = state.cell_to_barcode.get(cid, str(cid))
                _loh_label_before_by_barcode[str(bc)] = str(cl)

    # Step 1.
    vprint("\n  ── Step 1: Per-cell outlier removal ──", cfg)
    n_outliers = _per_cell_outlier_removal(state, cfg)
    if n_outliers == 0:
        vprint("  No per-cell outliers detected.", cfg)
    else:
        vprint(f"  Stripped LOH from {n_outliers} outlier cell(s).", cfg)

    # Step 2.
    vprint("\n  ── Step 2: Rescue merges ──", cfg)
    n_rescued = _rescue_merges(state, cfg)
    if n_rescued == 0:
        vprint("  No rescue merges needed.", cfg)

    # Step 3.
    vprint("\n  ── Step 3: Clone-level homogeneity ──", cfg)
    n_homo = _clone_homogeneity_check(state, cfg)
    if n_homo == 0:
        vprint("  All LOH clones passed homogeneity check.", cfg)

    # Step 3b.
    vprint("\n  ── Step 3b: Nested parent-child LOH rescue ──", cfg)
    nested_rescue_audit = _apply_nested_parent_child_rescues(state, cfg)
    if nested_rescue_audit.empty:
        vprint("  No automatic nested parent-child rescues applied.", cfg)

    # Step 4.
    vprint("\n  ── Step 4: Discard small LOH clones ──", cfg)
    n_small = _discard_small_loh_clones(state, cfg)
    if n_small == 0:
        vprint(
            f"  ✓ No LOH clones below size threshold ({cfg.min_final_clone_size}).",
            cfg,
        )

    # Step 4b.
    vprint("\n  ── Step 4b: Flag reverted LOH outliers ──", cfg)
    n_removed = _remove_reverted_loh_outliers(state, cfg, _loh_label_before_by_barcode)
    if n_removed == 0:
        vprint("  ✓ No reverted LOH outliers to flag as possible-LOH.", cfg)
    else:
        vprint(f"  Retained and FP-labelled {n_removed} reverted LOH-pattern cell(s).", cfg)

    # Step 5.
    vprint("\n  ── Step 5: Results reconciliation ──", cfg)
    _reconcile_results(state, cfg)

    # Summary.
    loh_after = state.loh_cell_count()
    labels_after = len(state.loh_clone_labels())
    from .loh_config import is_fp_label as _is_fp_label, is_sub_label as _is_sub_label
    _fp_counts = state.current_clones.astype(str).loc[
        state.current_clones.astype(str).apply(_is_fp_label)
    ].value_counts()
    _sub_counts = state.current_clones.astype(str).loc[
        state.current_clones.astype(str).apply(_is_sub_label)
    ].value_counts()

    vprint(
        f"\n  Active LOH clones after:  {labels_after}  ({loh_after} cells)  "
        f"[Δ = {loh_after - loh_before:+d} cells]",
        cfg,
    )
    vprint(
        f"  Unresolved SUB clones: {len(_sub_counts)}  ({int(_sub_counts.sum())} cells)",
        cfg,
    )
    if len(_sub_counts):
        for _label, _n in _sub_counts.items():
            vprint(f"    • {_label}  (n={int(_n)})", cfg)
    vprint(
        f"  Possible-LOH (FP) clones: {len(_fp_counts)}  ({int(_fp_counts.sum())} cells)",
        cfg,
    )
    if len(_fp_counts):
        for _label, _n in _fp_counts.items():
            vprint(f"    • {_label}  (n={int(_n)})", cfg)
    vprint(f"  Result keys: {sorted(results.keys())}", cfg)
    vprint(f"  Active LOH labels:  {state.loh_clone_labels()}", cfg)
    vprint("=" * 60 + "\n", cfg)

    return results
