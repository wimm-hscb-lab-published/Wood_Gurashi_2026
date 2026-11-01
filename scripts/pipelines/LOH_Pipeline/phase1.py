"""
LOH Pipeline – Phase 1: per-arm LOH detection and clone-level QC.

Entry point
-----------
``run_phase1(state, detect_loh_fn)``
    Calls ``detect_loh_fn`` (normally
    :func:`LOH_Pipeline.detect_loh.detect_loh_and_update_clones`) once per
    chromosome arm with size filtering and homogeneity checks *deferred*
    (``skip_size_filtering=True``, ``skip_homogeneity_check=True``), then runs
    Phase-1 clone QC on the resulting ``LOH-C<n> <arm>`` labels:

    1.  Pass 1 – detection on every arm; collects the extreme-AF dispersion
        (``sd_med``) of every clone with a valid LOH block to derive a global
        dynamic SD threshold.
    2.  Pass 2 – per-arm clone QC (:func:`_show_clones_for_arm`): size gate,
        arm-wide or localised (cytoband) extreme check, similarity and SD
        gates, directional consistency, NGT/no-call exclusion.  Failing
        clones have the arm's LOH token stripped.
    3.  Pool-and-reinstate for arms whose clones all failed.
    4.  Clone-level amplicon validation (if ``cfg.validate_segment_length``).
    5.  Cytoband annotation of surviving labels and a summary table.

After this phase ``state.results`` maps each arm that retained LOH to its
``(df_segments, df_loh, df_corr, df_homog)`` tuple, ``state.Sample`` clone
labels are updated in place, and ``state`` mappings are refreshed.
"""
from __future__ import annotations

import re
import traceback
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
# ``ndimage_label`` and ``LOH_TAG`` are not used here but are kept as
# module-level names for backward compatibility.
from scipy.ndimage import label as ndimage_label

from .loh_config import (
    LOH_TAG,
    LOHConfig,
    PipelineState,
    arm_variants,
    build_cytoband_maps,
    check_directional_consistency,
    edge_tolerant_extreme_runs,
    extract_chr_arms,
    fmt_float,
    get_base_clone,
    is_loh_label,
    vprint,
)


# ANSI escape codes for Jupyter/terminal coloured text
_RED   = "\033[91m"
_GREEN = "\033[92m"
_GREY  = "\033[90m"
_RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────
# Amplicon-validation helpers
# ─────────────────────────────────────────────────────────────────────

def _build_variant_to_amplicon(state: PipelineState) -> dict[str, str]:
    """Return ``{variant_id: amplicon_id}`` from Sample.dna.col_attrs."""
    try:
        col_attrs = state.Sample.dna.col_attrs
        ids = list(col_attrs.get("id", []))
        amps = list(col_attrs.get("amplicon", []))
        if len(ids) == len(amps) and len(ids) > 0:
            return {str(i): str(a) for i, a in zip(ids, amps)}
    except Exception:
        pass
    return {}


def _variant_pos(v: str) -> int:
    """Genomic position from a variant string like 'chr5:12345:A/G'."""
    try:
        return int(str(v).split(":")[1])
    except Exception:
        return 10**18


def _informative_variant_set(cfg: LOHConfig) -> set[str] | None:
    """Return the germline/informative variant set, or None if all variants count."""
    vals = getattr(cfg, "informative_variants", None)
    if vals is None:
        return None
    try:
        out = {str(v) for v in vals}
    except Exception:
        return None
    return out if out else None


def _block_somatic_flanking_ok(
    vars_sorted: list[str],
    block_indices: list[int],
    cfg: LOHConfig,
) -> tuple[bool, str]:
    """
    Validate blocks that contain somatic/non-informative variants.

    If cfg.informative_variants is provided, variants outside that set are
    treated as somatic/context variants.  Such variants may remain in the
    matrix, but they do not count as germline LOH support.  A block containing
    them passes only if enough informative variants remain in the block and,
    when requested, informative variants occur on both sides of the somatic span.
    
    The caller only supplies contiguous extreme blocks, so somatic positions in
    block_indices are already extreme by construction.
    """
    if not bool(getattr(cfg, "enforce_somatic_block_flanking", True)):
        return True, ""

    info_set = _informative_variant_set(cfg)
    if info_set is None:
        return True, ""

    if not block_indices:
        return False, "empty_block"

    block_vars = [vars_sorted[i] for i in block_indices]
    is_info = [str(v) in info_set for v in block_vars]

    if all(is_info):
        return True, ""

    n_info = int(sum(is_info))
    min_info = int(getattr(cfg, "min_informative_variants_flanking_somatic", 3))
    if n_info < min_info:
        return False, f"somatic_block:n_informative<{min_info}"

    if bool(getattr(cfg, "require_informative_on_both_sides_of_somatic", True)):
        som_idx = [i for i, ok in enumerate(is_info) if not ok]
        first_som = min(som_idx)
        last_som = max(som_idx)
        has_left = any(is_info[:first_som])
        has_right = any(is_info[last_som + 1:])
        if not (has_left and has_right):
            return False, "somatic_block:not_flanked_by_informative_variants"

    return True, ""


def _informative_amplicons_in_block(
    run_vars: list[str],
    v2a: dict[str, str],
    cfg: LOHConfig,
) -> set[str]:
    """Amplicons contributed by germline/informative variants only."""
    info_set = _informative_variant_set(cfg)
    out = set()
    for v in run_vars:
        if info_set is not None and str(v) not in info_set:
            continue
        amp = v2a.get(v)
        if amp not in (None, "", "nan", "None"):
            out.add(str(amp))
    return out


def _strip_specific_loh_token_from_label(label: str, chr_arm: str, clone_label: str) -> str:
    """Remove exactly this clone's LOH token for one arm from a label."""
    label = str(label)
    m = re.search(r"LOH-C\d+", str(clone_label))
    if not m:
        return label
    loh_id = re.escape(m.group(0))
    arm = re.escape(str(chr_arm))
    # Handles both unannotated tokens (" LOH-C1 chr17p") and cytoband-annotated
    # tokens (" LOH-C1 chr17p:p13.1-p13.3").
    pat = rf"\s+{loh_id}\s+{arm}(?::\S+)?"
    out = re.sub(pat, "", label).strip()
    return re.sub(r"\s+", " ", out)


def _get_ngt_matrix(state: PipelineState, cfg: LOHConfig) -> pd.DataFrame | None:
    """
    Return cells/barcodes x variants NGT matrix if available.

    MissionBio/Tapestri objects usually expose this through
    Sample.dna.get_attribute('NGT', constraint='row+col').  The helper is
    deliberately defensive because row/column labels differ across versions.
    """
    cached = getattr(state, "_phase1_ngt_matrix", None)
    if cached is not None:
        return cached

    try:
        ngt = state.Sample.dna.get_attribute("NGT", constraint="row+col")
    except Exception as e:
        if bool(getattr(cfg, "verbose", True)):
            vprint(f"    NGT exclusion skipped: could not read NGT attribute ({type(e).__name__}: {e})", cfg)
        state._phase1_ngt_matrix = None
        return None

    try:
        ngt = pd.DataFrame(ngt).copy()
    except Exception:
        state._phase1_ngt_matrix = None
        return None

    # If row index does not match known barcodes/cells but shape does, assign barcodes.
    try:
        barcodes = pd.Index(state.barcodes.astype(str))
        if ngt.shape[0] == len(barcodes):
            idx_str = pd.Index(ngt.index.map(str))
            if len(idx_str.intersection(barcodes)) == 0:
                ngt.index = barcodes
            else:
                ngt.index = idx_str
    except Exception:
        pass

    # If columns do not look like variant IDs but shape matches, assign col_attrs['id'].
    try:
        ids = list(state.Sample.dna.col_attrs.get("id", []))
        if len(ids) == ngt.shape[1]:
            col_str = pd.Index(ngt.columns.map(str))
            id_str = pd.Index([str(x) for x in ids])
            if len(col_str.intersection(pd.Index(state.variant_cols))) == 0:
                ngt.columns = id_str
            else:
                ngt.columns = col_str
    except Exception:
        ngt.columns = pd.Index(ngt.columns.map(str))

    ngt.index = pd.Index(ngt.index.map(str))
    ngt.columns = pd.Index(ngt.columns.map(str))
    state._phase1_ngt_matrix = ngt
    return ngt


def _ngt_no_call_mask(values: pd.DataFrame, cfg: LOHConfig) -> pd.DataFrame:
    """Boolean mask: True where an NGT value is a no-call/NGT state."""
    no_call_values = {str(x).strip().upper() for x in getattr(cfg, "phase1_ngt_no_call_values", ())}
    numeric_no_call_values = set(getattr(cfg, "phase1_ngt_numeric_no_call_values", ()))
    treat_nan = bool(getattr(cfg, "phase1_ngt_treat_nan_as_ngt", True))

    out = pd.DataFrame(False, index=values.index, columns=values.columns)

    if treat_nan:
        out |= values.isna()

    # Numeric no-call states, e.g. 3 in many NGT encodings.
    if numeric_no_call_values:
        num = values.apply(pd.to_numeric, errors="coerce")
        for x in numeric_no_call_values:
            try:
                out |= num.eq(float(x))
            except Exception:
                pass

    # String no-call states, e.g. "NGT".
    if no_call_values:
        sval = values.astype(str).apply(lambda col: col.str.strip().str.upper())
        out |= sval.isin(no_call_values)

    return out


def _exclude_ngt_cells_from_clone_block(
    state: PipelineState,
    chr_arm: str,
    clone_label: str,
    cells_in_df: list,
    block_variants: list[str],
    cfg: LOHConfig,
) -> tuple[list, list, float]:
    """
    Strip this arm's LOH token from cells that are NGT/no-call in the LOH block.

    Returns (kept_cells, dropped_cells, max_ngt_fraction_observed).
    """
    if not bool(getattr(cfg, "phase1_exclude_ngt_cells_in_loh_block", True)):
        return list(cells_in_df), [], np.nan

    if not cells_in_df or not block_variants:
        return list(cells_in_df), [], np.nan

    ngt = _get_ngt_matrix(state, cfg)
    if ngt is None or ngt.empty:
        return list(cells_in_df), [], np.nan

    block_vars = [str(v) for v in block_variants if str(v) in ngt.columns]
    if not block_vars:
        return list(cells_in_df), [], np.nan

    row_keys = []
    key_to_cell = {}
    for cell in cells_in_df:
        ckey = str(cell)
        if ckey in ngt.index:
            row_key = ckey
        else:
            bc = state.cell_to_barcode.get(cell)
            row_key = str(bc) if bc is not None and str(bc) in ngt.index else None
        if row_key is not None:
            row_keys.append(row_key)
            key_to_cell[row_key] = cell

    if not row_keys:
        return list(cells_in_df), [], np.nan

    sub = ngt.loc[row_keys, block_vars]
    no_call = _ngt_no_call_mask(sub, cfg)
    ngt_frac = no_call.sum(axis=1) / max(1, len(block_vars))
    max_allowed = float(getattr(cfg, "phase1_ngt_max_fraction_in_block", 0.0))
    drop_row_keys = ngt_frac.index[ngt_frac > max_allowed].tolist()
    dropped_cells = [key_to_cell[k] for k in drop_row_keys]

    if not dropped_cells:
        return list(cells_in_df), [], float(ngt_frac.max()) if len(ngt_frac) else np.nan

    dropped_set = set(dropped_cells)
    kept_cells = [c for c in cells_in_df if c not in dropped_set]

    current = pd.Series(state.Sample.dna.row_attrs["Clone"]).astype(str).copy()
    changed = 0
    records = []
    for cell in dropped_cells:
        bc = state.cell_to_barcode.get(cell)
        idx = state.sample_idx_for_barcode(str(bc)) if bc is not None else None
        if idx is None:
            continue
        old_label = str(current.iloc[idx])
        new_label = _strip_specific_loh_token_from_label(old_label, chr_arm, clone_label)
        if new_label != old_label:
            current.iloc[idx] = new_label
            changed += 1
        row_key = str(cell) if str(cell) in ngt_frac.index else str(bc)
        records.append({
            "chr_arm": chr_arm,
            "clone_label": clone_label,
            "cell_id": cell,
            "barcode": bc,
            "n_block_variants_with_ngt": int(no_call.loc[row_key].sum()) if row_key in no_call.index else np.nan,
            "n_block_variants_checked": len(block_vars),
            "ngt_fraction_in_block": float(ngt_frac.loc[row_key]) if row_key in ngt_frac.index else np.nan,
        })

    if changed:
        state.Sample.dna.row_attrs["Clone"] = current.values
        state.refresh()

    if records:
        if not hasattr(state, "phase1_ngt_exclusions"):
            state.phase1_ngt_exclusions = []
        state.phase1_ngt_exclusions.extend(records)

    return kept_cells, dropped_cells, float(ngt_frac.max()) if len(ngt_frac) else np.nan


def _amplicon_validate_clone(
    state: PipelineState,
    chr_arm: str,
    clone_label: str,
    cfg: LOHConfig,
    v2a: dict[str, str],
) -> bool:
    """
    Clone-level amplicon validation for one LOH clone on one arm.

    Computes the clone's mean AF profile on the arm's variants (sorted
    by position), marks variants as extreme (≤ low or ≥ high threshold),
    finds contiguous runs of extreme variants, and checks whether any
    run has ≥ ``min_contiguous_variants_final`` variants spanning
    ≥ ``min_different_amplicons`` distinct amplicons.

    Returns True if the clone PASSES validation.
    """
    min_len = cfg.min_contiguous_variants_final
    min_amps = cfg.min_different_amplicons
    low = cfg.low_allele_threshold
    high = cfg.high_allele_threshold

    vars_arm = arm_variants(chr_arm, state)
    vars_sorted = sorted(vars_arm, key=_variant_pos)
    if len(vars_sorted) < min_len:
        return True   # not enough variants to validate — pass by default

    cells = state.clone_to_cells.get(clone_label, [])
    cells_in_df = [c for c in cells if c in state.df_numeric.index]
    if not cells_in_df:
        return True

    # Clone mean AF per variant
    sub = state.df_numeric.loc[cells_in_df, vars_sorted]
    mu = sub.mean(axis=0, skipna=True).values.astype(float)

    # Binary extreme mask
    extreme = ((mu <= low) | (mu >= high)).astype(int)

    # Strict extreme support defines the block core.  One (configurable)
    # terminal non-extreme variant can count toward contiguous *span* length,
    # but cannot inflate support, purity, or amplicon counts.
    edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
    for run in edge_tolerant_extreme_runs(extreme, edge_tol):
        run_idx_list = list(run["core_indices"])
        span_idx_list = list(run["span_indices"])
        if len(span_idx_list) < min_len:
            continue

        ok, _reason = _block_somatic_flanking_ok(vars_sorted, run_idx_list, cfg)
        if not ok:
            continue

        run_vars = [vars_sorted[i] for i in run_idx_list]
        # Somatic/non-informative variants do not count toward independent
        # amplicon support when cfg.informative_variants is supplied.
        run_amps = _informative_amplicons_in_block(run_vars, v2a, cfg)
        if not run_amps:
            run_amps = {v2a.get(v) for v in run_vars
                        if v2a.get(v) not in (None, "", "nan", "None")}

        if len(run_amps) >= min_amps:
            return True

    return False


def _amplicon_validation_for_arm(
    state: PipelineState,
    chr_arm: str,
    cfg: LOHConfig,
) -> tuple[list[str], int]:
    """
    Run clone-level amplicon validation for all LOH clones on *chr_arm*.

    For each LOH clone label that mentions *chr_arm*, checks whether its
    mean profile has a contiguous extreme segment spanning multiple
    amplicons.  Clones that fail have the LOH token for this arm stripped.

    Returns (failing_clone_labels, n_cells_stripped).
    """
    if not cfg.validate_segment_length:
        return [], 0

    v2a = _build_variant_to_amplicon(state)
    if not v2a:
        vprint(f"    Amplicon validation skipped (no variant→amplicon mapping).", cfg)
        return [], 0

    # Find all LOH clone labels mentioning this arm
    failing: list[str] = []
    for clone_label in list(state.clone_to_cells.keys()):
        if not is_loh_label(clone_label):
            continue
        if chr_arm not in extract_chr_arms(clone_label):
            continue

        if not _amplicon_validate_clone(state, chr_arm, clone_label, cfg, v2a):
            failing.append(clone_label)

    if not failing:
        return [], 0

    # Strip the LOH token for this arm from failing clones
    n_stripped = _strip_arm_from_failing_clones(state, chr_arm, failing, cfg)
    return failing, n_stripped


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────
def run_phase1(
    state: PipelineState,
    detect_loh_fn: Callable[..., tuple],
) -> dict:
    """
    Phase 1: per-arm LOH detection followed by strict Phase-1 clone QC.

    Parameters
    ----------
    state : PipelineState
        Shared mutable state.  ``state.Sample.dna.row_attrs["Clone"]`` is
        modified in place, both by ``detect_loh_fn`` (adds
        ``LOH-C<n> <arm>`` tokens) and by the QC steps here (strips tokens
        from failing clones, reinstates pooled clones, appends cytoband
        ranges).  ``state.config`` supplies all thresholds.
    detect_loh_fn : callable
        Normally ``detect_loh_and_update_clones``.  Called once per arm in
        ``state.chr_arms`` with keyword arguments only and expected to return
        ``(df_segments, df_loh, df_corr, df_homog)``.

    Returns
    -------
    dict
        ``{chr_arm: (df_segments, df_loh, df_corr, df_homog)}`` for every arm
        that still carries LOH after QC.  The same object is stored on
        ``state.results``.

    Notes
    -----
    * Exceptions raised by ``detect_loh_fn`` for one arm are caught, printed
      (with traceback when ``cfg.verbose``) and the arm is recorded as
      ``ERROR``; processing continues with the next arm.
    * Pass 1 computes, for every LOH clone with ≥
      ``phase1_extreme_sd_qc_min_cells`` (default 5) cells, the median SD of
      folded extreme AFs on its longest edge-tolerant extreme block (span ≥
      ``min_contiguous_variants_initial``).  The global dynamic SD threshold
      is ``max(sd_med) × phase1_sd_dynamic_factor`` (default 1.2), or ``None``
      when no clone qualified.
    * Side effects on ``state``: ``results``, ``clone_block_variants``
      (accepted block variants per passing clone), ``phase1_ngt_exclusions``,
      ``_phase1_ngt_matrix`` (cache) and ``loh_cytoband_ranges``.
    * All progress is printed through ``vprint`` (silent when
      ``cfg.verbose`` is False).
    """
    cfg = state.config

    vprint("\n" + "=" * 60, cfg)
    vprint("PHASE 1: Per-arm LOH detection", cfg)
    vprint("=" * 60, cfg)
    vprint(
        f"  What: scan each chromosome arm independently for cells with LOH-like "
        f"allele-frequency extremes "
        f"(LOW≤{getattr(cfg, 'low_allele_threshold', 10):g}%, "
        f"HIGH≥{getattr(cfg, 'high_allele_threshold', 90):g}%).",
        cfg,
    )
    vprint(
        f"  Detection rule: require ≥{getattr(cfg, 'min_contiguous_variants_initial', 4)} "
        f"contiguous informative variants with block extreme fraction "
        f"≥{getattr(cfg, 'min_extreme_fraction', 0.85):.0%}.",
        cfg,
    )
    vprint(
        f"  Clustering rule: putative LOH cells are grouped by shared extreme variants "
        f"(min_shared_variants={getattr(cfg, 'min_shared_variants', 4)}, "
        f"min_cluster_size={getattr(cfg, 'min_cluster_size', 5)}, "
        f"min_final_clone_size={getattr(cfg, 'min_final_clone_size', 5)}).",
        cfg,
    )
    vprint(
        f"  Validation rule: candidate blocks are checked for clone size, localized "
        f"homogeneity, directional consistency, NGT/no-call burden, and independent "
        f"amplicon support "
        f"(min_different_amplicons={getattr(cfg, 'min_different_amplicons', 3)}).",
        cfg,
    )

    # Base localized QC
    local_min_cells = int(getattr(cfg, "phase1_localized_min_cells", 5))
    local_min_mean_r = float(getattr(cfg, "phase1_localized_min_mean_r", 0.80))
    local_min_good_frac = float(getattr(cfg, "phase1_localized_min_good_frac", 0.80))

    # Hard minimum clone size for all Phase 1 calls (arm-wide + localised).
    phase1_min_clone_cells_all = int(getattr(cfg, "phase1_min_clone_cells_all", 5))

    use_sd_qc = bool(getattr(cfg, "phase1_extreme_sd_qc_enabled", True))
    sd_min_cells = int(getattr(cfg, "phase1_extreme_sd_qc_min_cells", 5))
    sd_med = float(getattr(cfg, "phase1_extreme_sd_qc_max_median_sd", 12.0))
    sd_p90 = float(getattr(cfg, "phase1_extreme_sd_qc_max_p90_sd", 18.0))
    sd_factor = float(getattr(cfg, "phase1_sd_dynamic_factor", 1.2))

    vprint("  Arm/cytoband/amplicon QC in Phase 1: STRICT (strip/drop failing clones).", cfg)
    vprint(
        f"  Phase 1 clone-size gate (arm-wide + localised): n≥{phase1_min_clone_cells_all}",
        cfg,
    )
    vprint(
        f"  Localized cytoband QC (if arm-wide fails): "
        f"n≥{local_min_cells}, mean cell-to-mean r≥{local_min_mean_r:.2f}, "
        f"good-cell fraction≥{local_min_good_frac:.0%}",
        cfg,
    )

    if use_sd_qc:
        vprint(
            f"  LOH-block extreme-AF variance QC: n≥{sd_min_cells}, "
            f"arm-wide: median SD≤{sd_med:.1f}, p90 SD≤{sd_p90:.1f}; "
            f"localised: dynamic SD≤max(valid-block SD)×{sd_factor}",
            cfg,
        )
    if cfg.phase1_show_top_clones:
        vprint("  Top-clone AF summaries: ON", cfg)

    results: dict = {}
    n_arms = len(state.chr_arms)
    arm_summaries: list[tuple[str, int, str]] = []   # (arm, n_cells, status)

    # ══════════════════════════════════════════════════════════════
    # Pass 1: Run detection on all arms + collect SD from clones
    # with valid LOH blocks to compute a global dynamic SD threshold.
    # ══════════════════════════════════════════════════════════════
    sd_dynamic_factor = float(getattr(cfg, "phase1_sd_dynamic_factor", 1.2))
    sd_max_median_static = float(getattr(cfg, "phase1_extreme_sd_qc_max_median_sd", 12.0))
    use_sd_qc = bool(getattr(cfg, "phase1_extreme_sd_qc_enabled", True))
    sd_min_cells_cfg = int(getattr(cfg, "phase1_extreme_sd_qc_min_cells", 5))

    # Collect clone sd_med values (all valid LOH blocks, not just arm-wide passers)
    global_sd_meds: list[float] = []

    for idx, chr_arm in enumerate(state.chr_arms, 1):
        vprint(f"\n── [{idx}/{n_arms}] {chr_arm} " + "─" * (44 - len(chr_arm)), cfg)

        try:
            df_segments, df_loh, df_corr, df_homog = detect_loh_fn(
                Sample=state.Sample,
                df=state.df,
                grouped_by_chr_arm=state.grouped_by_chr_arm,
                target_chr=chr_arm,
                # ── detection thresholds ──
                low_allele_threshold=cfg.low_allele_threshold,
                high_allele_threshold=cfg.high_allele_threshold,
                min_segment_length=cfg.min_contiguous_variants_initial,
                min_extreme_fraction=cfg.min_extreme_fraction,
                min_shared_variants=cfg.min_shared_variants,
                min_cluster_size=cfg.min_cluster_size,
                min_final_clone_size=cfg.min_final_clone_size,
                correlation_threshold=cfg.correlation_threshold,
                # ── controls: always False here (backup is caller's responsibility) ──
                restore_from_backup=False,
                skip_homogeneity_check=True,         # deferred to Phase 3
                min_homogeneity_fraction=cfg.min_homogeneity_fraction,
                skip_size_filtering=True,            # deferred to Phase 3 / 6
                cytobands_df=getattr(state, 'cytobands_df', None),
                keep_incoherent_clusters=cfg.plot_dropped_clusters,
                verbose=cfg.detect_loh_verbose,
            )

            n_loh = len(df_loh)
            if n_loh > 0:
                results[chr_arm] = (df_segments, df_loh, df_corr, df_homog)

                # Collect SD from clones on this arm (all valid LOH blocks)
                if use_sd_qc:
                    state.refresh()
                    vars_arm = arm_variants(chr_arm, state)
                    vars_arm_sorted = sorted(
                        vars_arm,
                        key=lambda v: int(v.split(":")[1].split("-")[0]),
                    )

                    for clone_label, cells in state.clone_to_cells.items():
                        if not is_loh_label(clone_label):
                            continue
                        if chr_arm not in extract_chr_arms(clone_label):
                            continue
                        cells_in_df = [c for c in cells if c in state.df_numeric.index]
                        n = len(cells_in_df)
                        if n < sd_min_cells_cfg:
                            continue

                        af_matrix = state.df_numeric.loc[
                            cells_in_df, vars_arm
                        ].values.astype(float)
                        fin = np.isfinite(af_matrix)
                        n_fin = int(fin.sum())
                        if n_fin == 0:
                            continue

                        # Compute SD on best contiguous LOH-like block (clone mean profile)
                        af_sorted = state.df_numeric.loc[
                            cells_in_df, vars_arm_sorted
                        ].values.astype(float)
                        mu = np.nanmean(af_sorted, axis=0)
                        ext_1d = (
                            (mu <= cfg.low_allele_threshold)
                            | (mu >= cfg.high_allele_threshold)
                        ).astype(int)
                        edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
                        best_idx: list[int] = []
                        best_span_len = 0
                        for run in edge_tolerant_extreme_runs(ext_1d, edge_tol):
                            ridx = list(run["core_indices"])
                            span_len = len(run["span_indices"])
                            if span_len > best_span_len:
                                best_idx = ridx
                                best_span_len = span_len

                        if best_span_len >= cfg.min_contiguous_variants_initial:
                            sd_med, _, _ = _extreme_block_dispersion_stats(
                                af_sorted, best_idx,
                                cfg.low_allele_threshold,
                                cfg.high_allele_threshold,
                            )
                            if np.isfinite(sd_med):
                                global_sd_meds.append(sd_med)

                vprint(f"  Detection: {n_loh} putative LOH cells found", cfg)
            else:
                vprint("  ⚠️  No LOH detected", cfg)
                arm_summaries.append((chr_arm, 0, "—"))

        except Exception as e:
            vprint(f"  ❌ Error: {type(e).__name__}: {e}", cfg)
            if cfg.verbose:
                vprint(traceback.format_exc(), cfg)
            arm_summaries.append((chr_arm, 0, "ERROR"))
            continue

    # ── Compute global dynamic SD threshold ───────────────────────
    if global_sd_meds:
        global_dynamic_sd = max(global_sd_meds) * sd_dynamic_factor
        vprint(
            f"\n  Global dynamic SD threshold: {global_dynamic_sd:.1f} "
            f"(max sd_med={max(global_sd_meds):.1f} "
            f"× {sd_dynamic_factor}, from {len(global_sd_meds)} "
            f"clone(s) with valid LOH block(s))",
            cfg,
        )
    else:
        global_dynamic_sd = None
        vprint(
            f"\n  Global dynamic SD threshold: not available "
            f"(no clones with valid LOH blocks and SD ≥ {sd_min_cells_cfg} cells)",
            cfg,
        )

    # ══════════════════════════════════════════════════════════════
    # Pass 2: Evaluate clones per arm with global SD threshold,
    # strip failing clones, and print results.
    # ══════════════════════════════════════════════════════════════
    vprint("\n" + "─" * 60, cfg)
    vprint("  Clone QC evaluation (with global SD threshold)", cfg)
    vprint("─" * 60, cfg)

    for chr_arm in list(results.keys()):
        vprint(f"\n── {chr_arm} " + "─" * (50 - len(chr_arm)), cfg)

        any_pass, n_clones, n_pass, failing = _show_clones_for_arm(
            state, chr_arm, cfg, global_dynamic_sd=global_dynamic_sd
        )

        if failing:
            n_stripped = _strip_arm_from_failing_clones(state, chr_arm, failing, cfg)
            vprint(
                f"  {_RED}Removed {chr_arm} LOH from {n_stripped} cells "
                f"({len(failing)} clone{'s' if len(failing) != 1 else ''} "
                f"failed Phase 1 clone QC){_RESET}",
                cfg,
            )

        if any_pass:
            state.refresh()
            remaining = sum(
                len(cells)
                for lbl, cells in state.clone_to_cells.items()
                if is_loh_label(lbl) and chr_arm in extract_chr_arms(lbl)
            )
            vprint(
                f"  {chr_arm} summary: {remaining} cells with LOH "
                f"across {n_pass} passing clone{'s' if n_pass != 1 else ''} "
                f"(of {n_clones} total)",
                cfg,
            )
            arm_summaries.append((chr_arm, remaining, "LOH"))
        else:
            if chr_arm in results:
                del results[chr_arm]
            vprint(
                f"  {_RED}{chr_arm} summary: all {n_clones} putative LOH clone"
                f"{'s' if n_clones != 1 else ''} failed Phase 1 clone QC "
                f"— {chr_arm} LOH dropped{_RESET}",
                cfg,
            )
            arm_summaries.append((chr_arm, 0, "DROPPED"))

    # Add "—" entries for arms that had no LOH detected
    arms_in_summaries = {a for a, _, _ in arm_summaries}
    for chr_arm in state.chr_arms:
        if chr_arm not in arms_in_summaries:
            arm_summaries.append((chr_arm, 0, "—"))

    # Persist on state + refresh mappings (detect_loh_fn mutates Sample)
    state.results = results
    state.refresh()

    # Sync df_loh label columns with final (post-strip) clone labels
    _sync_results_labels(state, results)

    # ── Phase 1b-pre: Pool-and-reinstate ──────────────────────────────
    # Cells sharing one real arm LOH can be fragmented into many tiny
    # sub-clones by co-occurring secondary LOH tokens, so every clone on the
    # arm fails the size gate.  For each DROPPED arm, pool the cells that are
    # extreme on that arm by base genotype and reinstate a consolidated clone
    # when the pool passes the size and extreme checks (the size threshold
    # itself is not lowered).
    _pool_and_reinstate_dropped_arms(state, results, arm_summaries, cfg)

    # ── Phase 1b: Clone-level amplicon validation ──────────────────────
    if cfg.validate_segment_length:
        amp_msgs: list[str] = []
        amp_dropped: set[str] = set()
        for chr_arm in sorted(results.keys()):
            failing, n_stripped = _amplicon_validation_for_arm(state, chr_arm, cfg)
            if failing:
                amp_msgs.append(
                    f"  {chr_arm}: removed LOH from {len(failing)} clone(s) "
                    f"({n_stripped} cells) — failed amplicon validation "
                    f"(min_len={cfg.min_contiguous_variants_final}, "
                    f"min_amps={cfg.min_different_amplicons})"
                )
                state.refresh()
                remaining = sum(
                    len(cells)
                    for lbl, cells in state.clone_to_cells.items()
                    if is_loh_label(lbl) and chr_arm in extract_chr_arms(lbl)
                )
                if remaining == 0 and chr_arm in results:
                    del results[chr_arm]
                    amp_msgs.append(f"  {chr_arm}: all clones failed → arm dropped")
                    amp_dropped.add(chr_arm)

        if amp_msgs:
            vprint("\n  ── Amplicon validation ──", cfg)
            for msg in amp_msgs:
                vprint(msg, cfg)
            _sync_results_labels(state, results)
        else:
            vprint("\n  ── Amplicon validation: all clones passed ──", cfg)

        if amp_dropped:
            arm_summaries = [
                (arm, n, "DROPPED") if arm in amp_dropped else (arm, n, status)
                for arm, n, status in arm_summaries
            ]

    # ── Phase 1c: Annotate surviving LOH clone labels with cytobands ──
    n_annotated = _annotate_loh_clones_with_cytobands(state, results, cfg)
    if n_annotated > 0:
        vprint(
            f"\n  ── Cytoband annotation: {n_annotated} cell(s) updated ──",
            cfg,
        )
        _sync_results_labels(state, results)

    _print_phase1_summary(state, results, arm_summaries, cfg)
    return results


# ─────────────────────────────────────────────────────────────────────
# Phase-1 clone QC helpers
# ─────────────────────────────────────────────────────────────────────

def _extreme_block_dispersion_stats(
    af_sorted: np.ndarray,
    block_indices: list[int],
    low_thr: float,
    high_thr: float,
) -> tuple[float, float, int]:
    """
    Dispersion of extreme AF values within a clone's LOH block.

    AF values are "folded" to distance-from-nearest-extreme:
      folded = min(AF, 100-AF)

    This treats LOW and HIGH LOH symmetrically.
    Returns (median_sd, p90_sd, n_variants_used).
    """
    if af_sorted.ndim != 2 or len(block_indices) == 0:
        return np.nan, np.nan, 0

    per_var_sd: list[float] = []

    for vi in block_indices:
        vals = af_sorted[:, vi]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue

        ext_mask = (vals <= low_thr) | (vals >= high_thr)
        ext_vals = vals[ext_mask]
        if len(ext_vals) < 2:
            continue

        folded = np.minimum(ext_vals, 100.0 - ext_vals)
        sd = float(np.std(folded, ddof=1)) if len(folded) > 1 else 0.0
        per_var_sd.append(sd)

    if not per_var_sd:
        return np.nan, np.nan, 0

    arr = np.array(per_var_sd, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 90)), int(len(arr))


# ─────────────────────────────────────────────────────────────────────
# Clone AF summary per arm (all clones, with pass/fail)
# ─────────────────────────────────────────────────────────────────────

def _show_clones_for_arm(
    state: PipelineState, chr_arm: str, cfg: LOHConfig,
    global_dynamic_sd: float | None = None,
) -> tuple[bool, int, int, list[str]]:
    """
    Run Phase-1 clone QC for every active LOH clone on *chr_arm*.

    For each LOH clone mentioning the arm (largest first), the clone mean AF
    profile over position-sorted arm variants is scanned for edge-tolerant
    extreme runs (span ≥ 4, passing the somatic-flanking rule); the longest
    span (ties: higher block extreme fraction) is the clone's LOH block.  A
    clone then passes if, in order:

    1. ``n ≥ phase1_min_clone_cells_all`` (default 5);
    2. **arm-wide**: extreme fraction over all arm AFs ≥
       ``min_extreme_fraction``, and (when SD QC applies) block
       ``sd_median ≤ phase1_extreme_sd_qc_max_median_sd`` (12.0) and
       ``sd_p90 ≤ phase1_extreme_sd_qc_max_p90_sd`` (18.0); **or**
       **localised** (cytoband mode only): the best run of consecutive block
       cytobands each with extreme fraction ≥ ``min_extreme_fraction`` holds ≥
       ``phase1_localized_min_extreme_variants_passing_bands`` (default 5)
       variants, cell-to-mean Pearson r on that block has mean ≥
       ``phase1_localized_min_mean_r`` (0.80) and fraction of cells with
       r ≥ ``phase1_localized_good_cell_r`` (0.80) ≥
       ``phase1_localized_min_good_frac`` (0.80) — zero-variance pairs count
       as r = 1.0 — and block ``sd_median`` ≤ the dynamic SD threshold;
    3. directional consistency on the accepted block
       (``check_directional_consistency``);
    4. after stripping cells that are NGT/no-call in the block, the clone
       still has ≥ ``phase1_min_clone_cells_all`` cells.

    Passing clones have their block variants stored in
    ``state.clone_block_variants[clone_label]``.  When
    ``cfg.phase1_show_top_clones`` is set, one line per clone is printed.

    Parameters
    ----------
    global_dynamic_sd : float or None
        SD threshold for localised clones; falls back to
        ``phase1_extreme_sd_qc_max_median_sd`` when None.

    Returns
    -------
    tuple
        ``(any_pass, n_clones, n_pass, failing_clone_labels)``.
    """
    state.refresh()

    vars_arm = arm_variants(chr_arm, state)
    if not vars_arm:
        return False, 0, 0, []

    vars_arm_sorted = sorted(vars_arm, key=_variant_pos)

    _var_idx_to_band, _band_to_var_indices = build_cytoband_maps(
        vars_arm_sorted, getattr(state, "cytobands_df", None), chr_arm
    )
    _cytoband_mode = _var_idx_to_band is not None and _band_to_var_indices is not None

    loh_clones_for_arm: dict[str, list] = defaultdict(list)
    for clone_label, cells in state.clone_to_cells.items():
        if not is_loh_label(clone_label):
            continue
        if chr_arm in extract_chr_arms(clone_label):
            loh_clones_for_arm[clone_label] = cells

    if not loh_clones_for_arm:
        return False, 0, 0, []

    ranked = sorted(loh_clones_for_arm.items(), key=lambda kv: len(kv[1]), reverse=True)

    thr = float(cfg.min_extreme_fraction)
    n_clones = len(ranked)

    local_min_cells = int(getattr(cfg, "phase1_localized_min_cells", 5))
    local_min_mean_r = float(getattr(cfg, "phase1_localized_min_mean_r", 0.80))
    local_min_good_frac = float(getattr(cfg, "phase1_localized_min_good_frac", 0.80))
    local_good_cell_r = float(getattr(cfg, "phase1_localized_good_cell_r", 0.80))

    phase1_min_clone_cells_all = int(
        getattr(cfg, "phase1_min_clone_cells_all", local_min_cells)
    )

    # Phase 1 minimum LOH block span is fixed at 4 variants.
    min_block_len = 4

    use_extreme_sd_qc = bool(getattr(cfg, "phase1_extreme_sd_qc_enabled", True))
    sd_min_cells = int(getattr(cfg, "phase1_extreme_sd_qc_min_cells", 5))
    sd_max_median = float(getattr(cfg, "phase1_extreme_sd_qc_max_median_sd", 12.0))
    sd_max_p90 = float(getattr(cfg, "phase1_extreme_sd_qc_max_p90_sd", 18.0))
    sd_dynamic_factor = float(getattr(cfg, "phase1_sd_dynamic_factor", 1.2))

    clone_metrics: list[dict] = []

    for clone_label, cells in ranked:
        cells_in_df = [c for c in cells if c in state.df_numeric.index]
        n = len(cells_in_df)
        if n == 0:
            continue

        af_matrix = state.df_numeric.loc[cells_in_df, vars_arm].values.astype(float)
        finite_mask = np.isfinite(af_matrix)
        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            continue

        vals = af_matrix[finite_mask]
        low_frac = float(np.sum(vals <= cfg.low_allele_threshold)) / n_finite
        high_frac = float(np.sum(vals >= cfg.high_allele_threshold)) / n_finite
        extreme = low_frac + high_frac
        mid_frac = 1.0 - extreme
        mean_af = float(np.nanmean(af_matrix))
        median_af = float(np.nanmedian(af_matrix))

        af_sorted = state.df_numeric.loc[cells_in_df, vars_arm_sorted].values.astype(float)

        clone_mean = np.nanmean(af_sorted, axis=0)
        ext_1d = (
            (clone_mean <= cfg.low_allele_threshold)
            | (clone_mean >= cfg.high_allele_threshold)
        ).astype(int)

        edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
        best_block_idx: list[int] = []  # strict extreme support
        best_block_span_idx: list[int] = []  # support + tolerated terminal edge(s)
        best_block_extreme = np.nan
        best_block_edge_padded = 0

        for run in edge_tolerant_extreme_runs(ext_1d, edge_tol):
            ridx_list = list(run["core_indices"])
            span_idx_list = list(run["span_indices"])
            if len(span_idx_list) < min_block_len:
                continue
            som_ok, _som_reason = _block_somatic_flanking_ok(vars_arm_sorted, ridx_list, cfg)
            if not som_ok:
                continue
            block_vals = af_sorted[:, ridx_list]
            block_fin = np.isfinite(block_vals)
            block_n = int(block_fin.sum())
            if block_n == 0:
                continue
            block_extreme = float(
                np.sum(block_vals[block_fin] <= cfg.low_allele_threshold)
                + np.sum(block_vals[block_fin] >= cfg.high_allele_threshold)
            ) / block_n

            if (
                len(span_idx_list) > len(best_block_span_idx)
                or (len(span_idx_list) == len(best_block_span_idx)
                    and (not np.isfinite(best_block_extreme)
                         or block_extreme > best_block_extreme))
            ):
                best_block_idx = ridx_list
                best_block_span_idx = span_idx_list
                best_block_extreme = float(block_extreme)
                best_block_edge_padded = int(run["n_edge_padded"])

        max_block = len(best_block_span_idx)

        block_label = "NA"
        spanned_bands: list[str] = []
        if best_block_idx and _cytoband_mode:
            for vi in best_block_idx:
                b = _var_idx_to_band.get(vi)
                if b and (not spanned_bands or spanned_bands[-1] != b):
                    spanned_bands.append(b)
            if spanned_bands:
                block_label = (
                    spanned_bands[0] if len(spanned_bands) == 1
                    else f"{spanned_bands[0]}-{spanned_bands[-1]}"
                )
            else:
                block_label = f"v{best_block_idx[0]+1}-v{best_block_idx[-1]+1}"
        elif best_block_idx:
            block_label = f"v{best_block_idx[0]+1}-v{best_block_idx[-1]+1}"

        sd_median_val = np.nan
        sd_p90_val = np.nan
        sd_nvars = 0
        if use_extreme_sd_qc and best_block_idx and n >= sd_min_cells:
            sd_median_val, sd_p90_val, sd_nvars = _extreme_block_dispersion_stats(
                af_sorted=af_sorted,
                block_indices=best_block_idx,
                low_thr=cfg.low_allele_threshold,
                high_thr=cfg.high_allele_threshold,
            )

        clone_metrics.append({
            "clone_label": clone_label,
            "cells": cells,
            "cells_in_df": cells_in_df,
            "n": n,
            "af_sorted": af_sorted,
            "extreme": extreme,
            "low_frac": low_frac,
            "high_frac": high_frac,
            "mid_frac": mid_frac,
            "mean_af": mean_af,
            "median_af": median_af,
            "max_block": max_block,
            "best_block_idx": best_block_idx,
            "best_block_span_idx": best_block_span_idx,
            "best_block_edge_padded": best_block_edge_padded,
            "best_block_extreme": best_block_extreme,
            "block_label": block_label,
            "spanned_bands": spanned_bands,
            "sd_median": sd_median_val,
            "sd_p90": sd_p90_val,
            "sd_nvars": sd_nvars,
        })

    if not clone_metrics:
        return False, 0, 0, []

    if global_dynamic_sd is not None:
        dynamic_sd_threshold = global_dynamic_sd
        _sd_source = f"global: valid-block max_sd×{sd_dynamic_factor}"
    else:
        dynamic_sd_threshold = sd_max_median
        _sd_source = "static (no global reference)"

    n_pass = 0
    failing: list[str] = []

    if cfg.phase1_show_top_clones:
        arm_thr_hdr = f"{thr:.0%}"
        local_sd_hdr = f"{dynamic_sd_threshold:.1f} [{_sd_source}]"

        vprint(
            f"  ┌─ LOH clones on {chr_arm}  "
            f"({len(vars_arm)} variants, arm-wide extreme threshold: {arm_thr_hdr}, "
            f"localised SD≤{local_sd_hdr})",
            cfg,
        )

    show_failed = bool(getattr(cfg, "phase1_show_failed_clones", True))

    for m in clone_metrics:
        clone_label = m["clone_label"]
        n = m["n"]
        af_sorted = m["af_sorted"]
        best_block_idx = m["best_block_idx"]
        best_block_span_idx = m.get("best_block_span_idx", best_block_idx)
        best_block_edge_padded = int(m.get("best_block_edge_padded", 0))
        spanned_bands = m["spanned_bands"]

        # Use strict support for purity/direction/NGT; span only relaxes the
        # terminal block-length rule.
        block_idx_used = list(best_block_idx) if best_block_idx else []
        block_label_used = m["block_label"]
        block_extreme_used = m["best_block_extreme"]
        sd_med_used = m["sd_median"]
        sd_p90_used = m["sd_p90"]
        sd_nvars_used = m["sd_nvars"]

        clone_min_cells_req = phase1_min_clone_cells_all
        arm_extreme_thr_req = thr
        local_mean_r_req = local_min_mean_r
        local_good_frac_req = local_min_good_frac
        local_good_cell_r_req = local_good_cell_r

        local_sd_threshold_req = dynamic_sd_threshold

        localized_min_cells_req = max(local_min_cells, clone_min_cells_req)

        arm_wide_pass = m["extreme"] >= arm_extreme_thr_req

        passes = False
        qc_reason = ""

        if n < clone_min_cells_req:
            passes = False
            qc_reason = f"fail:n<{clone_min_cells_req}"

        elif arm_wide_pass:
            passes = True
            qc_reason = "arm-wide"

            if (use_extreme_sd_qc and n >= sd_min_cells and np.isfinite(m["sd_median"])):
                if m["sd_median"] > sd_max_median or m["sd_p90"] > sd_max_p90:
                    passes = False
                    qc_reason = "fail:extreme-AF-variance"

        elif _cytoband_mode and best_block_idx and spanned_bands:
            min_pass_vars = int(
                getattr(
                    cfg,
                    "phase1_localized_min_extreme_variants_passing_bands",
                    max(min_block_len, 5),
                )
            )

            band_to_block_idx: dict[str, list[int]] = defaultdict(list)
            for vi in best_block_idx:
                b = _var_idx_to_band.get(vi)
                if b:
                    band_to_block_idx[str(b)].append(int(vi))

            band_pass: dict[str, bool] = {}
            for b in spanned_bands:
                idxs = band_to_block_idx.get(str(b), [])
                if not idxs:
                    band_pass[str(b)] = False
                    continue
                b_vals = af_sorted[:, idxs]
                b_fin = np.isfinite(b_vals)
                b_n = int(b_fin.sum())
                if b_n == 0:
                    band_pass[str(b)] = False
                    continue
                b_ext = float(
                    np.sum(b_vals[b_fin] <= cfg.low_allele_threshold)
                    + np.sum(b_vals[b_fin] >= cfg.high_allele_threshold)
                ) / b_n
                band_pass[str(b)] = (b_ext >= arm_extreme_thr_req)

            runs: list[list[str]] = []
            cur: list[str] = []
            for b in spanned_bands:
                b = str(b)
                if band_pass.get(b, False):
                    cur.append(b)
                else:
                    if cur:
                        runs.append(cur)
                        cur = []
            if cur:
                runs.append(cur)

            best_run: list[str] = []
            best_run_nvars = 0
            for run in runs:
                idxs = []
                for b in run:
                    idxs.extend(band_to_block_idx.get(b, []))
                nvars = len(idxs)
                if (nvars > best_run_nvars) or (nvars == best_run_nvars and len(run) > len(best_run)):
                    best_run_nvars = nvars
                    best_run = run

            passing_bands = best_run
            effective_block_idx = sorted(
                {vi for b in passing_bands for vi in band_to_block_idx.get(b, [])}
            )

            bands_ok = (len(effective_block_idx) >= min_pass_vars)
            somatic_block_reason = ""
            if bands_ok:
                bands_ok, somatic_block_reason = _block_somatic_flanking_ok(
                    vars_arm_sorted, effective_block_idx, cfg
                )

            if bands_ok:
                block_idx_used = effective_block_idx
                if passing_bands:
                    block_label_used = (
                        passing_bands[0]
                        if len(passing_bands) == 1
                        else f"{passing_bands[0]}-{passing_bands[-1]}"
                    )

                _blk = af_sorted[:, block_idx_used]
                _fin = np.isfinite(_blk)
                _n = int(_fin.sum())
                if _n > 0:
                    block_extreme_used = float(
                        np.sum(_blk[_fin] <= cfg.low_allele_threshold)
                        + np.sum(_blk[_fin] >= cfg.high_allele_threshold)
                    ) / _n
                else:
                    block_extreme_used = np.nan

                if use_extreme_sd_qc and n >= sd_min_cells and block_idx_used:
                    sd_med_used, sd_p90_used, sd_nvars_used = _extreme_block_dispersion_stats(
                        af_sorted=af_sorted,
                        block_indices=block_idx_used,
                        low_thr=cfg.low_allele_threshold,
                        high_thr=cfg.high_allele_threshold,
                    )

            # ── Similarity gate: zero variance is treated as perfect similarity ──
            similarity_ok = False
            sim_mean_r = np.nan
            sim_good_frac = np.nan

            if bands_ok and n >= localized_min_cells_req:
                block_vals = af_sorted[:, block_idx_used]
                mu = np.nanmean(block_vals, axis=0)
                cell_rs: list[float] = []

                for ii in range(block_vals.shape[0]):
                    xi = block_vals[ii, :]
                    mask = np.isfinite(xi) & np.isfinite(mu)
                    if int(mask.sum()) < max(3, min_block_len):
                        continue

                    xi_use = xi[mask]
                    mu_use = mu[mask]

                    # Correlation is undefined for zero variance; count as r = 1.0.
                    if np.nanstd(xi_use) == 0 or np.nanstd(mu_use) == 0:
                        r = 1.0
                    else:
                        r = np.corrcoef(xi_use, mu_use)[0, 1]

                    if np.isfinite(r):
                        cell_rs.append(float(r))

                if len(cell_rs) >= localized_min_cells_req:
                    sim_mean_r = float(np.mean(cell_rs))
                    sim_good_frac = float(np.mean(np.array(cell_rs) >= local_good_cell_r_req))
                    similarity_ok = (sim_mean_r >= local_mean_r_req and sim_good_frac >= local_good_frac_req)

            if bands_ok and n < localized_min_cells_req:
                similarity_ok = False

            sd_ok = True
            if bands_ok and similarity_ok and use_extreme_sd_qc:
                if np.isfinite(sd_med_used):
                    sd_ok = sd_med_used <= local_sd_threshold_req

            if bands_ok and similarity_ok and sd_ok:
                passes = True
                qc_reason = (
                    f"localized:{passing_bands[0]}"
                    if len(passing_bands) == 1
                    else f"localized:{passing_bands[0]}-{passing_bands[-1]}"
                )
            else:
                if not bands_ok:
                    if somatic_block_reason:
                        qc_reason = f"localized_fail:{somatic_block_reason}"
                    else:
                        qc_reason = (
                            f"localized_fail:cytoband(pass_ext_vars={len(effective_block_idx)}"
                            f"<{min_pass_vars})"
                        )
                elif n < localized_min_cells_req:
                    qc_reason = f"localized_fail:n<{localized_min_cells_req}"
                elif not similarity_ok:
                    sim_mean_str = "NA" if not np.isfinite(sim_mean_r) else f"{sim_mean_r:.3f}"
                    sim_good_str = "NA" if not np.isfinite(sim_good_frac) else f"{sim_good_frac:.0%}"
                    qc_reason = f"localized_fail:sim(mean_r={sim_mean_str}, good={sim_good_str})"
                elif similarity_ok and not sd_ok:
                    qc_reason = f"localized_fail:sd_med={sd_med_used:.1f}>{local_sd_threshold_req:.1f}"
                else:
                    qc_reason = "localized_fail:cytoband"

        else:
            qc_reason = "arm-wide_fail"

        # ── Directional consistency gate (overrides passes if mixed) ──
        dir_result = None
        if (
            passes
            and getattr(cfg, "directional_check_enabled", True)
            and len(block_idx_used) > 0
        ):
            block_vals = af_sorted[:, block_idx_used]
            dir_result = check_directional_consistency(
                block_vals,
                low_thr=cfg.low_allele_threshold,
                high_thr=cfg.high_allele_threshold,
                min_per_var_agreement=cfg.directional_min_per_variant_agreement,
                min_block_pass_frac=cfg.directional_min_block_pass_frac,
                min_extreme_cells_per_var=cfg.directional_min_extreme_cells_per_variant,
            )
            if not dir_result["passes"]:
                passes = False
                bp = dir_result["block_pass_frac"]
                bp_str = "NA" if not np.isfinite(bp) else f"{bp:.0%}"
                qc_reason = (
                    f"fail:directional({dir_result['dominant_direction']},"
                    f"pass={dir_result['n_variants_passing']}/"
                    f"{dir_result['n_variants_evaluated']}={bp_str})"
                )

        # ── NGT/no-call exclusion inside the accepted LOH block ──
        # Cells that are NGT/no-call for variants in the accepted block are
        # stripped out of this LOH subclone before Phase 1 finalizes it.
        # If too many cells are stripped and the subclone drops below the
        # uniform Phase 1 clone-size gate, the clone fails QC.
        ngt_excluded = 0
        ngt_max_frac = np.nan
        if passes and block_idx_used:
            block_variants_used = [vars_arm_sorted[i] for i in block_idx_used]
            kept_cells, dropped_cells, ngt_max_frac = _exclude_ngt_cells_from_clone_block(
                state=state,
                chr_arm=chr_arm,
                clone_label=clone_label,
                cells_in_df=m["cells_in_df"],
                block_variants=block_variants_used,
                cfg=cfg,
            )
            ngt_excluded = len(dropped_cells)
            if ngt_excluded > 0:
                m["cells_in_df"] = kept_cells
                n = len(kept_cells)
                if n < clone_min_cells_req:
                    passes = False
                    qc_reason = f"fail:NGT_excluded_n<{clone_min_cells_req}"
                else:
                    qc_reason = f"{qc_reason};NGT_excluded={ngt_excluded}" if qc_reason else f"NGT_excluded={ngt_excluded}"

        band_details: list[str] = []
        if _cytoband_mode:
            band_order = sorted(_band_to_var_indices.keys(), key=lambda b: min(_band_to_var_indices[b]))
            for band in band_order:
                band_indices = _band_to_var_indices[band]
                n_vars_in_band = len(band_indices)
                band_vals = af_sorted[:, band_indices]
                band_finite = np.isfinite(band_vals)
                band_n = int(band_finite.sum())
                if band_n == 0:
                    band_details.append(f"{band} ({n_vars_in_band}v, no data)")
                    continue
                band_extreme_frac = float(
                    np.sum(band_vals[band_finite] <= cfg.low_allele_threshold)
                    + np.sum(band_vals[band_finite] >= cfg.high_allele_threshold)
                ) / band_n
                band_details.append(f"{band} ({n_vars_in_band}v, {band_extreme_frac*100:.0f}% extreme)")

        if passes:
            n_pass += 1
            # Persist LOH-block variant names so Phase 3 can use them
            if not hasattr(state, "clone_block_variants"):
                state.clone_block_variants = {}
            if block_idx_used:
                state.clone_block_variants[clone_label] = [
                    vars_arm_sorted[i] for i in block_idx_used
                ]
        else:
            failing.append(clone_label)

        if cfg.phase1_show_top_clones and (passes or show_failed):
            tag = "✓" if passes else "✗"
            col_on, col_off = (_GREEN, _RESET) if passes else (_RED, _RESET)

            if len(block_idx_used) > 0 and np.isfinite(block_extreme_used):
                edge_str = (
                    f", span={len(best_block_span_idx)}v, edge_tol={best_block_edge_padded}"
                    if best_block_edge_padded > 0 else ""
                )
                block_str = (
                    f"block={block_label_used} (support={len(block_idx_used)}v"
                    f"{edge_str}, {block_extreme_used*100:.1f}% extreme)"
                )
            else:
                block_str = "block=NA (0v)"

            sd_str = ""
            if use_extreme_sd_qc and np.isfinite(sd_med_used):
                _sd_thr_used = sd_max_median if arm_wide_pass else local_sd_threshold_req
                sd_mark = "✓" if sd_med_used <= _sd_thr_used else "✗"
                sd_str = (
                    f"  extreme_sd_med={sd_med_used:.1f} "
                    f"(p90={sd_p90_used:.1f}, nvar={sd_nvars_used}) {sd_mark}"
                )
            elif use_extreme_sd_qc and sd_nvars_used == 0 and len(block_idx_used) > 0:
                sd_str = "  extreme_sd_med=NA"

            dir_str = ""
            if dir_result is not None and dir_result["n_variants_evaluated"] > 0:
                dir_mark = "✓" if dir_result["passes"] else "✗"
                bp = dir_result["block_pass_frac"]
                bp_pct = "NA" if not np.isfinite(bp) else f"{bp*100:.0f}%"
                dir_str = (
                    f"  dir={dir_result['dominant_direction']} "
                    f"({dir_result['n_variants_passing']}/"
                    f"{dir_result['n_variants_evaluated']}={bp_pct}) {dir_mark}"
                )

            ngt_str = ""
            if 'ngt_excluded' in locals() and ngt_excluded > 0:
                ngt_str = f"  NGT_excluded={ngt_excluded}"

            reason_str = f"  [{qc_reason}]" if qc_reason else ""

            vprint(
                f"  • {col_on}{clone_label}  (n={n})  "
                f"n_variants_loh_block={len(block_idx_used)}  {block_str}  "
                f"extreme={fmt_float(m['extreme'] * 100, 1)}%  "
                f"mean={fmt_float(m['mean_af'], 1)}  "
                f"median={fmt_float(m['median_af'], 1)}  "
                f"[LOW≤{cfg.low_allele_threshold:.0f}%: "
                f"{fmt_float(m['low_frac'] * 100, 1)}%  "
                f"HIGH≥{cfg.high_allele_threshold:.0f}%: "
                f"{fmt_float(m['high_frac'] * 100, 1)}%  "
                f"mid: {fmt_float(m['mid_frac'] * 100, 1)}%]  "
                f"{tag}{sd_str}{dir_str}{ngt_str}{reason_str}{col_off}",
                cfg,
            )
            if passes and band_details:
                vprint(
                    f"          {_GREY}cytobands: {', '.join(band_details)}{_RESET}",
                    cfg,
                )

    if cfg.phase1_show_top_clones:
        vprint("  └─", cfg)

    return (n_pass > 0), n_clones, n_pass, failing


# ─────────────────────────────────────────────────────────────────────
# Pool-and-reinstate dropped arms
# ─────────────────────────────────────────────────────────────────────
def _pool_and_reinstate_dropped_arms(
    state: PipelineState,
    results: dict,
    arm_summaries: list[tuple[str, int, str]],
    cfg: LOHConfig,
) -> None:
    """
    Pool-and-reinstate arms dropped because their clones were fragmented.

    When every Phase-1 clone on an arm fails QC, cells sharing one real LOH
    event may have been split into many tiny sub-clones by co-occurring
    secondary LOH tokens (e.g. ``"TET2_... LOH-C0 chr4q LOH-C2 chr5q"``,
    ``"TET2_... LOH-C0 chr4q LOH-C3 chr6p"``, ...).

    For each arm whose ``arm_summaries`` status is ``"DROPPED"`` (and that has
    ≥ 4 variants):

    1. Every cell in ``state.df_numeric`` (not only previously-LOH cells) with
       ≥ 50 % of its finite AFs on the arm extreme becomes a candidate.
    2. Candidates are pooled by ``get_base_clone`` of their current label.
    3. A pool with ≥ ``phase1_min_clone_cells_all`` (default 5) cells is
       accepted if its pooled extreme fraction ≥ ``min_extreme_fraction``, or
       otherwise if its mean profile has an edge-tolerant extreme run of
       span ≥ 4 whose strict-support block (passing the somatic-flanking
       rule) has extreme fraction ≥ ``min_extreme_fraction``.
    4. Accepted cells are relabelled: cells already carrying the arm keep
       their label, other LOH labels get `` LOH-C<k> <arm>`` appended, and
       non-LOH cells become ``"<base> LOH-C<k> <arm>"``, where ``k`` is one
       more than the highest LOH-C index among LOH labels on that arm.

    Mutates ``state.Sample`` labels, ``arm_summaries`` (DROPPED → LOH) and
    ``results`` (adds/extends a ``df_loh`` entry for the arm).  No similarity,
    SD, directional or NGT checks are re-run on the pool.
    """
    phase1_min_n = int(getattr(cfg, "phase1_min_clone_cells_all", 5))
    thr = float(cfg.min_extreme_fraction)
    low = cfg.low_allele_threshold
    high = cfg.high_allele_threshold
    min_block = 4  # same as Phase 1 QC

    # The pre-strip df_loh of dropped arms has already been deleted from
    # results, so candidates are found directly from each cell's AF profile.
    dropped_arms = [arm for arm, _, status in arm_summaries if status == "DROPPED"]
    if not dropped_arms:
        return

    state.refresh()

    reinstated_any = False

    for chr_arm in dropped_arms:
        vars_arm = arm_variants(chr_arm, state)
        if len(vars_arm) < min_block:
            continue

        vars_sorted = sorted(vars_arm, key=lambda v: int(v.split(":")[1].split("-")[0]))
        all_cells_in_df = list(state.df_numeric.index)

        # For each cell, compute per-cell extreme fraction on this arm
        af = state.df_numeric.loc[all_cells_in_df, vars_sorted].values.astype(float)
        fin = np.isfinite(af)
        per_cell_extreme = np.where(
            fin.sum(axis=1) > 0,
            ((af <= low) | (af >= high)).sum(axis=1) / np.maximum(fin.sum(axis=1), 1),
            0.0,
        )

        # Only consider cells with ≥ 50% extreme on this arm as candidates
        # (lower than the final threshold so we don't miss borderline cases
        # that become clearer once pooled)
        candidate_cells = [
            c for c, ef in zip(all_cells_in_df, per_cell_extreme) if ef >= 0.5
        ]
        if not candidate_cells:
            continue

        # Group candidate cells by base-genotype (strip ALL LOH tokens)
        from collections import defaultdict as _dd
        pools: dict[str, list] = _dd(list)
        for cell_id in candidate_cells:
            bc = state.cell_to_barcode.get(cell_id, str(cell_id))
            clone_lbl = state.barcode_to_clone.get(str(bc), "")
            base = get_base_clone(clone_lbl)   # strip all LOH tokens
            pools[base].append(cell_id)

        for base_name, pool_cells in pools.items():
            if len(pool_cells) < phase1_min_n:
                continue

            # Verify pooled mean profile actually passes extreme QC
            pool_in_df = [c for c in pool_cells if c in state.df_numeric.index]
            if not pool_in_df:
                continue

            af_pool = state.df_numeric.loc[pool_in_df, vars_sorted].values.astype(float)
            fin_pool = np.isfinite(af_pool)
            n_fin = int(fin_pool.sum())
            if n_fin == 0:
                continue

            pool_extreme = float(
                ((af_pool[fin_pool] <= low) | (af_pool[fin_pool] >= high)).sum()
            ) / n_fin

            if pool_extreme < thr:
                # Try localised block check: does the mean profile have a
                # contiguous run of ≥ min_block extreme variants?
                mu = np.nanmean(af_pool, axis=0)
                ext_1d = ((mu <= low) | (mu >= high)).astype(int)
                edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
                runs = edge_tolerant_extreme_runs(ext_1d, edge_tol)
                best_run = max((len(run["span_indices"]) for run in runs), default=0)
                if best_run < min_block:
                    continue
                # Check strict-support block extreme fraction.
                best_idx = []
                best_ext = 0.0
                for run in runs:
                    ridx_list = list(run["core_indices"])
                    if len(run["span_indices"]) < min_block:
                        continue
                    som_ok, _som_reason = _block_somatic_flanking_ok(vars_sorted, ridx_list, cfg)
                    if not som_ok:
                        continue
                    blk = af_pool[:, ridx_list]
                    bfin = np.isfinite(blk)
                    bn = int(bfin.sum())
                    if bn == 0:
                        continue
                    bext = float(
                        ((blk[bfin] <= low) | (blk[bfin] >= high)).sum()
                    ) / bn
                    if len(ridx_list) > len(best_idx) or (
                        len(ridx_list) == len(best_idx) and bext > best_ext
                    ):
                        best_idx = ridx_list
                        best_ext = bext
                if best_ext < thr:
                    continue

            # Determine next available LOH-C index for this arm + base
            existing_c_indices = []
            for lbl in state.clone_to_cells:
                if not is_loh_label(lbl):
                    continue
                if chr_arm not in extract_chr_arms(lbl):
                    continue
                m = re.search(r"LOH-C(\d+)", str(lbl))
                if m:
                    existing_c_indices.append(int(m.group(1)))
            next_c = max(existing_c_indices, default=-1) + 1

            new_loh_label = f"{base_name} LOH-C{next_c} {chr_arm}"

            # Relabel cells in Sample, preserving any surviving LOH arm tokens
            # (e.g. "... LOH-C2 chr5p" must not be overwritten by a chr7p
            # reinstate label): append the arm to existing LOH labels and use
            # the pooled label only for cells already stripped back to base.
            clone_arr = state.Sample.dna.row_attrs["Clone"]
            barcode_arr = pd.Series(state.Sample.dna.row_attrs["barcode"]).astype(str).values

            n_relabelled = 0
            relabelled_cells: list[str] = []
            relabelled_labels: list[str] = []

            for cell_id in pool_in_df:
                bc = state.cell_to_barcode.get(cell_id, str(cell_id))
                bc_str = str(bc)
                mask = barcode_arr == bc_str
                if not mask.any():
                    continue

                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                idx0 = int(idxs[0])

                old_label = str(clone_arr[idx0])
                old_arms = set(extract_chr_arms(old_label)) if is_loh_label(old_label) else set()

                if chr_arm in old_arms:
                    target_label = old_label
                elif is_loh_label(old_label):
                    target_label = f"{old_label} LOH-C{next_c} {chr_arm}"
                else:
                    target_label = new_loh_label

                clone_arr[mask] = target_label
                n_relabelled += 1
                relabelled_cells.append(cell_id)
                relabelled_labels.append(target_label)

            state.Sample.dna.row_attrs["Clone"] = clone_arr
            state.refresh()

            unique_labels = list(dict.fromkeys(relabelled_labels))
            label_msg = unique_labels[0] if len(unique_labels) == 1 else f"{len(unique_labels)} labels"
            vprint(
                f"\n  ── Pool-and-reinstate [{chr_arm}]: "
                f"base='{base_name}', n={n_relabelled} cells "
                f"(extreme={pool_extreme:.1%}) → {label_msg}",
                cfg,
            )

            # Update arm_summaries: flip DROPPED → LOH for this arm
            for i, (arm, n, status) in enumerate(arm_summaries):
                if arm == chr_arm and status == "DROPPED":
                    arm_summaries[i] = (arm, n_relabelled, "LOH")
                    break

            # Add/update a non-empty df_loh entry so downstream reconciliation
            # does not immediately delete the reinstated arm from results.
            # Phase 3 operates on state labels, but result keys are still used
            # by summaries and plotting/reconciliation.
            if relabelled_cells:
                df_loh_reinstated = pd.DataFrame(index=pd.Index(relabelled_cells, name="cell"))
                df_loh_reinstated["label"] = relabelled_labels

                if chr_arm in results and len(results[chr_arm][1]) > 0:
                    df_segments_old, df_loh_old, df_corr_old, df_homog_old = results[chr_arm]
                    df_loh_new = pd.concat([df_loh_old, df_loh_reinstated], axis=0)
                    df_loh_new = df_loh_new.loc[~df_loh_new.index.duplicated(keep="last")].copy()
                    results[chr_arm] = (df_segments_old, df_loh_new, df_corr_old, df_homog_old)
                else:
                    results[chr_arm] = (
                        pd.DataFrame(),
                        df_loh_reinstated,
                        pd.DataFrame(),
                        pd.DataFrame(),
                    )

            reinstated_any = True

    if reinstated_any:
        state.refresh()


# ─────────────────────────────────────────────────────────────────────
# Strip a failing arm's LOH annotation from clone labels
# ─────────────────────────────────────────────────────────────────────
def _strip_arm_from_failing_clones(
    state: PipelineState,
    chr_arm: str,
    failing_clone_labels: list[str],
    cfg: LOHConfig,
) -> int:
    """
    For each clone label in *failing_clone_labels*, remove the
    ' LOH-C<n> <chr_arm>' token from every cell that carries it.

    Also handles combined p+q tokens like 'LOH-C0 chr5p+chr5q' if the
    failing arm is one of the pair.

    Returns the number of cells whose labels were modified.
    """
    updated = state.current_clones.copy()
    n_modified = 0

    pat_simple = re.compile(rf" LOH-C\d+ {re.escape(chr_arm)}\b")
    pat_combined = re.compile(
        rf" LOH-C\d+ (?:{re.escape(chr_arm)}\+\S+|\S+\+{re.escape(chr_arm)})"
    )

    for i in range(len(updated)):
        old_label = updated.iloc[i]
        if old_label not in failing_clone_labels:
            continue

        new_label = pat_simple.sub("", old_label)
        new_label = pat_combined.sub("", new_label)
        new_label = new_label.strip()

        if new_label != old_label:
            updated.iloc[i] = new_label
            n_modified += 1

    if n_modified > 0:
        state.set_clone_labels(updated)

    return n_modified


# ─────────────────────────────────────────────────────────────────────
# Sync results labels with post-strip state
# ─────────────────────────────────────────────────────────────────────
def _sync_results_labels(state: PipelineState, results: dict) -> None:
    """
    After extreme-check stripping, synchronise *results* with the
    authoritative clone labels in *state*.

    For each arm still in results:
      1. Update the ``label`` column in df_loh from state.
      2. Keep only cells whose final label still mentions this arm.
      3. Filter df_segments to match the surviving cells.
      4. Drop the arm from results if no cells remain.
    """
    for chr_arm in list(results.keys()):
        df_segments, df_loh, df_corr, df_homog = results[chr_arm]

        if len(df_loh) == 0:
            del results[chr_arm]
            continue

        new_labels = []
        for cell_id in df_loh.index:
            bc = state.cell_to_barcode.get(cell_id, str(cell_id))
            new_labels.append(state.barcode_to_clone.get(str(bc), ""))
        df_loh["label"] = new_labels

        keep_mask = df_loh["label"].str.contains(chr_arm, na=False)
        df_loh = df_loh.loc[keep_mask].copy()

        if len(df_segments) > 0 and "cell" in df_segments.columns:
            surviving_cells = set(df_loh.index)
            df_segments = df_segments.loc[df_segments["cell"].isin(surviving_cells)].copy()

        if len(df_loh) == 0:
            del results[chr_arm]
        else:
            results[chr_arm] = (df_segments, df_loh, df_corr, df_homog)


# ─────────────────────────────────────────────────────────────────────
# Annotate surviving LOH clones with cytoband ranges
# ─────────────────────────────────────────────────────────────────────
def _annotate_loh_clones_with_cytobands(
    state: PipelineState,
    results: dict,
    cfg: LOHConfig,
) -> int:
    """
    Append cytoband range to surviving LOH clone labels.

    E.g. ``'TP53_R248W_Hem LOH-C2 chr5q'``
      →  ``'TP53_R248W_Hem LOH-C2 chr5q:q15'``
    """
    cytobands_df = getattr(state, 'cytobands_df', None)
    if cytobands_df is None:
        return 0

    state.refresh()

    loh_clones: dict[str, list[str]] = {}
    for clone_label in state.clone_to_cells:
        if not is_loh_label(clone_label):
            continue
        arms = extract_chr_arms(clone_label)
        if arms:
            loh_clones[clone_label] = arms

    if not loh_clones:
        return 0

    thr = cfg.min_extreme_fraction
    min_vars_per_band = cfg.min_contiguous_variants_initial

    rename: dict[str, str] = {}
    cyto_ranges: dict[str, list[str]] = {}

    for clone_label, arms in loh_clones.items():
        cells = state.clone_to_cells.get(clone_label, [])
        cells_in_df = [c for c in cells if c in state.df_numeric.index]
        if not cells_in_df:
            continue

        new_label = clone_label
        clone_bands: list[str] = []

        for arm in arms:
            vars_arm = arm_variants(arm, state)
            vars_sorted = sorted(vars_arm, key=lambda v: int(v.split(":")[1].split("-")[0]))
            if not vars_sorted:
                continue

            _v2b, _b2v = build_cytoband_maps(vars_sorted, cytobands_df, arm)
            if _v2b is None or _b2v is None:
                continue

            af = state.df_numeric.loc[cells_in_df, vars_sorted].values.astype(float)

            _af_fin = np.isfinite(af)
            _af_n = int(_af_fin.sum())
            if _af_n == 0:
                continue
            _arm_ext = float(
                np.sum(af[_af_fin] <= cfg.low_allele_threshold)
                + np.sum(af[_af_fin] >= cfg.high_allele_threshold)
            ) / _af_n

            band_order = sorted(_b2v.keys(), key=lambda b: min(_b2v[b]))

            if _arm_ext >= thr:
                clone_bands.extend(band_order)
                continue

            mu = np.nanmean(af, axis=0)
            ext_1d = ((mu <= cfg.low_allele_threshold) | (mu >= cfg.high_allele_threshold)).astype(int)
            edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
            best_idx: list[int] = []
            best_span_len = 0
            best_ext = np.nan
            for run in edge_tolerant_extreme_runs(ext_1d, edge_tol):
                idxs = np.asarray(run["core_indices"], dtype=int)
                span_len = len(run["span_indices"])
                if span_len < min_vars_per_band:
                    continue
                blk = af[:, idxs]
                blk_fin = np.isfinite(blk)
                blk_n = int(blk_fin.sum())
                if blk_n == 0:
                    continue
                blk_ext = float(
                    np.sum(blk[blk_fin] <= cfg.low_allele_threshold)
                    + np.sum(blk[blk_fin] >= cfg.high_allele_threshold)
                ) / blk_n
                if (span_len > best_span_len
                    or (span_len == best_span_len
                        and (not np.isfinite(best_ext) or blk_ext > best_ext))):
                    best_idx = list(map(int, idxs))
                    best_span_len = span_len
                    best_ext = float(blk_ext)

            if not best_idx:
                continue

            band_to_block_idx: dict[str, list[int]] = defaultdict(list)
            pk_bands: list[str] = []
            for vi in best_idx:
                b = _v2b.get(vi)
                if b:
                    b = str(b)
                    band_to_block_idx[b].append(int(vi))
                    if not pk_bands or pk_bands[-1] != b:
                        pk_bands.append(b)

            if not pk_bands:
                continue

            band_pass: dict[str, bool] = {}
            for b in pk_bands:
                idxs = band_to_block_idx.get(b, [])
                if not idxs:
                    band_pass[b] = False
                    continue
                bv = af[:, idxs]
                bf = np.isfinite(bv)
                bn = int(bf.sum())
                if bn == 0:
                    band_pass[b] = False
                    continue
                be = float(
                    np.sum(bv[bf] <= cfg.low_allele_threshold)
                    + np.sum(bv[bf] >= cfg.high_allele_threshold)
                ) / bn
                band_pass[b] = (be >= thr)

            runs: list[list[str]] = []
            cur: list[str] = []
            for b in pk_bands:
                if band_pass.get(b, False):
                    cur.append(b)
                else:
                    if cur:
                        runs.append(cur)
                        cur = []
            if cur:
                runs.append(cur)

            best_run: list[str] = []
            best_run_nvars = 0
            for run in runs:
                idxs = []
                for b in run:
                    idxs.extend(band_to_block_idx.get(b, []))
                nvars = len(idxs)
                if (nvars > best_run_nvars) or (nvars == best_run_nvars and len(run) > len(best_run)):
                    best_run_nvars = nvars
                    best_run = run

            passing_bands = best_run
            passing_block_idx = sorted({vi for b in passing_bands for vi in band_to_block_idx.get(b, [])})

            min_pass_vars = int(
                getattr(cfg, "phase1_localized_min_extreme_variants_passing_bands", max(min_vars_per_band, 5))
            )
            if len(passing_block_idx) < min_pass_vars:
                continue

            clone_bands.extend(passing_bands)

            start_b = passing_bands[0]
            end_b = passing_bands[-1]
            band_str = start_b if start_b == end_b else f"{start_b}-{end_b}"

            annotated = f"{arm}:{band_str}"
            if annotated in new_label:
                continue

            new_label = new_label.replace(f" {arm}", f" {annotated}", 1)

        if new_label != clone_label:
            rename[clone_label] = new_label
        if clone_bands:
            cyto_ranges[new_label if new_label != clone_label else clone_label] = clone_bands

    if not rename:
        state.loh_cytoband_ranges = cyto_ranges
        return 0

    updated = state.current_clones.copy()
    n_modified = 0
    for i in range(len(updated)):
        old = str(updated.iloc[i])
        if old in rename:
            updated.iloc[i] = rename[old]
            n_modified += 1

    if n_modified > 0:
        state.set_clone_labels(updated)

    state.loh_cytoband_ranges = cyto_ranges
    return n_modified


# ─────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────
def _print_phase1_summary(
    state: PipelineState,
    results: dict,
    arm_summaries: list[tuple[str, int, str]],
    cfg: LOHConfig,
) -> None:
    """Print a compact Phase 1 summary table using final state counts."""
    vprint("\n" + "=" * 60, cfg)
    vprint("PHASE 1 SUMMARY", cfg)
    vprint("=" * 60, cfg)

    final_arm_counts: dict[str, int] = defaultdict(int)
    for clone_label, cells in state.clone_to_cells.items():
        if not is_loh_label(clone_label):
            continue
        for arm in extract_chr_arms(clone_label):
            final_arm_counts[arm] += len(cells)

    vprint(f"  {'Arm':<10} {'Cells':>6}  {'Status'}", cfg)
    vprint(f"  {'─'*10} {'─'*6}  {'─'*10}", cfg)
    for arm, _mid_loop_n, status in arm_summaries:
        final_n = final_arm_counts.get(arm, 0)
        if status == "DROPPED":
            vprint(f"  {_RED}{arm:<10} {final_n:>6}  {status}{_RESET}", cfg)
        else:
            vprint(f"  {arm:<10} {final_n:>6}  {status}", cfg)

    arms_with_loh   = sum(1 for _, _, s in arm_summaries if s == "LOH")
    arms_dropped    = sum(1 for _, _, s in arm_summaries if s == "DROPPED")
    total_loh_cells = state.loh_cell_count()

    vprint(f"\n  Arms processed:  {len(arm_summaries)}", cfg)
    vprint(f"  Arms with LOH:   {arms_with_loh}", cfg)
    if arms_dropped:
        vprint(
            f"  {_RED}Arms dropped:    {arms_dropped} "
            f"(failed extreme check or amplicon validation){_RESET}",
            cfg,
        )
    vprint(f"  Total LOH cells: {total_loh_cells} (after extreme filtering)", cfg)