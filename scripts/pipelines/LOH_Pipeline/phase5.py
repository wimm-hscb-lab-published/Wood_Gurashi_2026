"""
Phase 5 – LOH Clone Summary & Hem-TP53 Warning
================================================

Produces a per-clone summary table and warns if any Hem-TP53 clone
lacks chr17p LOH.

This phase also performs a final LOH-label cleanup before summarizing:

1. Merge final LOH clones that share the same base clone name and same LOH
   arm set when their LOH-block AF profiles are highly correlated.
2. Rename every terminal clone with fewer than ``min_final_clone_size`` cells
   (and any legacy ``small: ...`` label) to the single label ``small``.

That final cleanup is necessary because Phase 4 can create or split LOH clones
after Phase 3's small-clone cleanup.

For every surviving LOH clone the strongest LOH block per arm is recomputed
from the clone-mean AF profile, its independent amplicon support sets a
low/mid/high confidence tier, and an ADO-plausibility readout (empirical
same-base background rates, or a fixed per-amplicon rate) is attached.

Main entry point
----------------
``run_phase5(state, cfg=None, *, loh_min_cells=None, heading=...)`` –
cleanup, summary table (stored in ``state.results["phase5_summary"]``) and
printed report. ``run_post_phase5_subclone_scan`` (Phase 6) reuses it as the
Phase 7 summary.

Call after Phase 4::

    from phase5 import run_phase5
    summary = run_phase5(state)
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .loh_config import (
    arm_variants,
    extract_chr_arms,
    get_base_clone,
    is_fp_label,
    is_hem_like,
    is_loh_label,
    is_small_label,
    is_sub_label,
    mean_profile,
    pearson_safe,
    sort_chromosome_arms,
    vprint,
)

if TYPE_CHECKING:
    from .loh_config import LOHConfig, PipelineState


# ── ANSI helpers ───────────────────────────────────────────────────
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"

_DEFAULT_HEADING = "PHASE 5: LOH CLONE SUMMARY"
_SMALL_LABEL = "small"

# Amplicon values treated as "no amplicon annotation".
_MISSING_AMPLICON_VALUES = (None, "", "nan", "None")

# Column order of the Phase 5 summary table (also used for the empty table).
_SUMMARY_COLUMNS = [
    "clone_label",
    "n_cells",
    "is_loh",
    "is_subclonal",
    "is_small",
    "loh_arms",
    "is_hem",
    "is_tp53",
    "has_chr17p_loh",
    "loh_block_details",
    "loh_block_min_amplicons",
    "loh_confidence",
    "loh_confidence_reason",
    "qc_alert",
    "ado_alert_explanation",
    "ado_rate_source",
    "ado_assumed_amplicon_rate",
    "ado_empirical_background_cells",
    "ado_empirical_dp_used",
    "ado_empirical_min_depth",
    "ado_empirical_amplicon_rates",
    "ado_cell_probability",
    "ado_original_clone_size",
    "ado_observed_loh_cells",
    "ado_expected_cells",
    "ado_pvalue_ge_observed",
    "ado_verdict",
    "ado_qc_scope",
]


def _is_active_loh(label: object) -> bool:
    """True for an active LOH label (excludes FP:/SUB: review labels)."""
    return is_loh_label(label)


def _is_active_non_loh(label: object) -> bool:
    """True for a label that is not LOH, FP, SUB or small."""
    return (
        (not is_loh_label(label))
        and (not is_fp_label(label))
        and (not is_sub_label(label))
        and (not is_small_label(label))
    )


def _is_subclonal_loh(label: object) -> bool:
    """True for an unresolved directional ``SUB:`` LOH label."""
    return is_sub_label(label)


def _merge_similar_loh_clones_before_phase5(
    state: "PipelineState",
    cfg: "LOHConfig",
) -> int:
    """
    Merge LOH clones immediately before Phase 5 if they:
      1. share the same base clone name,
      2. have the same LOH arm set,
      3. have highly similar mean AF profiles over those LOH arms.

    This catches final Phase 4 split subclones such as:
        BASE LOH-C0 chr5q
        BASE LOH-C1 chr5q
    when their LOH blocks are effectively the same biological event.

    Returns
    -------
    int
        Number of cells whose clone label was changed by this merge pass.

    Notes
    -----
    Disabled when ``cfg.phase5_merge_similar_loh_clones`` is False. Pairwise
    Pearson r of clone-mean AF over the group's arm variants must reach
    ``phase5_merge_similar_loh_min_corr`` (falling back to
    ``post_merge_correlation_threshold``, default 0.80) with at least
    ``min_variants_per_arm_for_corr`` (default 5) variants. Similar labels are
    joined transitively (union-find); each component is relabelled to its
    largest member (ties broken alphabetically).
    """
    enabled = bool(getattr(cfg, "phase5_merge_similar_loh_clones", True))
    if not enabled:
        return 0

    corr_threshold = float(
        getattr(
            cfg,
            "phase5_merge_similar_loh_min_corr",
            getattr(cfg, "post_merge_correlation_threshold", 0.80),
        )
    )
    min_arm_vars = int(getattr(cfg, "min_variants_per_arm_for_corr", 5))

    loh_labels = [
        str(lbl)
        for lbl, cells in state.clone_to_cells.items()
        if _is_active_loh(lbl) and len(cells) > 0
    ]

    if len(loh_labels) < 2:
        return 0

    # Group only labels with the same base clone and same LOH arm set.
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for lbl in loh_labels:
        base = get_base_clone(lbl)
        arms = tuple(sorted(set(extract_chr_arms(lbl))))
        if arms:
            groups[(base, arms)].append(lbl)

    total_changed_cells = 0

    for (base, arms), labels in groups.items():
        if len(labels) < 2:
            continue

        # Variants across all LOH arms in this group.
        vars_used: list[str] = []
        for arm in arms:
            vars_used.extend(
                [v for v in arm_variants(arm, state) if v in state.df_numeric.columns]
            )
        vars_used = list(dict.fromkeys(vars_used))

        if len(vars_used) < min_arm_vars:
            continue

        profiles = {}
        sizes = {}
        for lbl in labels:
            cells = [
                c
                for c in state.clone_to_cells.get(lbl, [])
                if c in state.df_numeric.index
            ]
            if len(cells) == 0:
                continue
            profiles[lbl] = mean_profile(state.df_numeric, cells, vars_used)
            sizes[lbl] = len(cells)

        labels = [lbl for lbl in labels if lbl in profiles and profiles[lbl] is not None]
        if len(labels) < 2:
            continue

        # Union-find over similar LOH clone labels.
        parent = {lbl: lbl for lbl in labels}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                r = pearson_safe(profiles[a], profiles[b], min_n=min_arm_vars)
                if r is not None and float(r) >= corr_threshold:
                    union(a, b)

        components: dict[str, list[str]] = defaultdict(list)
        for lbl in labels:
            components[find(lbl)].append(lbl)

        current = state.current_clones.copy()
        changed_this_group = 0

        for comp_labels in components.values():
            if len(comp_labels) < 2:
                continue

            # Keep the largest label as the representative.
            rep = sorted(comp_labels, key=lambda x: (-sizes.get(x, 0), x))[0]

            for old_lbl in comp_labels:
                if old_lbl == rep:
                    continue

                mask = current.astype(str) == old_lbl
                n = int(mask.sum())
                if n == 0:
                    continue

                current.loc[mask] = rep
                changed_this_group += n

                vprint(
                    f"  Phase5 pre-merge: {old_lbl} "
                    f"(n={n}) → {rep} "
                    f"[base={base}, arms={','.join(arms)}]",
                    cfg,
                )

        if changed_this_group > 0:
            state.set_clone_labels(current)
            total_changed_cells += changed_this_group

    return total_changed_cells


def _prune_small_terminal_clones_before_phase5(
    state: "PipelineState",
    cfg: "LOHConfig",
    *,
    loh_min_cells: int | None = None,
) -> int:
    """Rename every terminal clone below the final size threshold to ``small``.

    All sub-threshold terminal groups are intentionally collapsed into one
    review-only category rather than retained as named FP/SUB/base remnants.

    ``loh_min_cells`` optionally relaxes the floor for LOH labels only (used by
    the Phase 6 residual scan, where provisional small LOH children are kept for
    downstream merging). When None, the standard ``min_final_clone_size`` floor
    applies to every label.

    Returns the number of cells relabelled (0 when
    ``cfg.phase5_prune_all_small_clones`` is False).
    """
    if not bool(getattr(cfg, "phase5_prune_all_small_clones", True)):
        return 0

    min_cells = int(getattr(cfg, "min_final_clone_size", 5))
    loh_floor = int(loh_min_cells) if loh_min_cells is not None else min_cells

    def _floor_for(label) -> int:
        # LOH children get the (optionally relaxed) LOH floor; everything else
        # keeps the standard final-size floor.
        return loh_floor if is_loh_label(label) else min_cells

    counts = state.current_clones.value_counts()
    # Collapse legacy provenance-bearing ``small: ...`` labels as well as
    # every newly sub-threshold terminal group into the one final ``small``
    # category.
    terminal_small = [
        str(label)
        for label, n in counts.items()
        if (is_small_label(label) and str(label).strip() != _SMALL_LABEL)
        or ((not is_small_label(label)) and int(n) < _floor_for(label))
    ]
    if not terminal_small:
        return 0

    current = state.current_clones.copy()
    changed = 0
    for label in terminal_small:
        mask = current.astype(str).eq(label)
        n = int(mask.sum())
        if n == 0:
            continue
        new_label = _SMALL_LABEL
        current.loc[mask] = new_label
        changed += n
        vprint(
            f"  Phase5 final prune: {label} (n={n} < {_floor_for(label)}) → {new_label}",
            cfg,
        )

    if changed:
        state.set_clone_labels(current)
    return changed


# ── Final-clone LOH block support / QC-alert helpers ────────────────
def _contiguous_true_runs(mask: list[bool]) -> list[list[int]]:
    """Return index-runs where mask is True."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, ok in enumerate(mask):
        if ok:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _format_variant_block_span(
    vars_sorted: list[str],
    run_indices: list[int],
    chr_arm: str,
    state: "PipelineState",
) -> str:
    """Human-readable cytoband span for a contiguous variant block."""
    try:
        from .loh_config import build_cytoband_maps

        var_idx_to_band, _band_to_idx = build_cytoband_maps(
            vars_sorted,
            getattr(state, "cytobands_df", None),
            chr_arm,
        )
        if var_idx_to_band:
            bands = [var_idx_to_band.get(int(i)) for i in run_indices]
            bands = [b for b in bands if b not in (None, "", "nan", "None")]
            if bands:
                if bands[0] == bands[-1]:
                    return str(bands[0])
                return f"{bands[0]}-{bands[-1]}"
    except Exception:
        pass

    # Fallback to genomic coordinates if cytobands are unavailable.
    try:
        start = str(vars_sorted[int(run_indices[0])]).split(":")[1]
        end = str(vars_sorted[int(run_indices[-1])]).split(":")[1]
        return f"{start}-{end}"
    except Exception:
        return "block"


def _best_loh_block_support_for_clone_arm(
    state: "PipelineState",
    cfg: "LOHConfig",
    clone_label: str,
    chr_arm: str,
) -> dict | None:
    """
    Recompute the best final LOH block for one clone/arm and report the
    independent informative amplicon support.

    This mirrors Phase 1 amplicon validation: use the clone mean AF profile,
    find contiguous extreme runs, ignore somatic/context variants for amplicon
    support when cfg.informative_variants is supplied, and keep the strongest
    passing block.

    Returns None when the arm has fewer than ``min_contiguous_variants_final``
    variants, the clone has no cells, no variant→amplicon map exists, or no
    block passes. Otherwise returns a dict with ``chr_arm``, ``n_variants``,
    ``n_amplicons``, ``amplicons``, ``amplicon_ids``, ``amplicon_to_variants``,
    ``amplicon_to_variant_dirs`` (LOW/HIGH per variant), ``block_span`` and
    ``variant_ids``. Blocks are ranked by amplicons, then variants.
    """
    # Imported lazily: phase1 is only needed for this final-block recomputation.
    from .phase1 import (
        _block_somatic_flanking_ok,
        _build_variant_to_amplicon,
        _informative_amplicons_in_block,
        _variant_pos,
    )

    min_len = int(getattr(cfg, "min_contiguous_variants_final", 4))
    min_amps = int(getattr(cfg, "min_different_amplicons", 3))
    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))

    vars_arm = [v for v in arm_variants(chr_arm, state) if v in state.df_numeric.columns]
    vars_sorted = sorted(vars_arm, key=_variant_pos)
    if len(vars_sorted) < min_len:
        return None

    cells = [c for c in state.clone_to_cells.get(clone_label, []) if c in state.df_numeric.index]
    if not cells:
        return None

    sub = state.df_numeric.loc[cells, vars_sorted]
    mu = sub.mean(axis=0, skipna=True).values.astype(float)
    extreme_mask = [bool((x <= low) or (x >= high)) if np.isfinite(x) else False for x in mu]

    v2a = _build_variant_to_amplicon(state)
    if not v2a:
        return None

    candidates: list[dict] = []
    for run_indices in _contiguous_true_runs(extreme_mask):
        if len(run_indices) < min_len:
            continue

        ok, _reason = _block_somatic_flanking_ok(vars_sorted, list(map(int, run_indices)), cfg)
        if not ok:
            continue

        run_vars = [vars_sorted[int(i)] for i in run_indices]
        info_amps = _informative_amplicons_in_block(run_vars, v2a, cfg)
        all_amps = {
            str(v2a.get(v))
            for v in run_vars
            if v2a.get(v) not in _MISSING_AMPLICON_VALUES
        }
        amps = info_amps if info_amps else all_amps
        n_amps = int(len(amps))
        if n_amps < min_amps:
            continue

        mu_map = {str(v): float(mu[j]) for j, v in enumerate(vars_sorted)}
        amp_to_vars: dict[str, list[str]] = {}
        amp_to_variant_dirs: dict[str, dict[str, str]] = {}
        for v in run_vars:
            amp = str(v2a.get(v)) if v2a.get(v) not in _MISSING_AMPLICON_VALUES else str(v)
            if amp not in amps:
                continue
            amp_to_vars.setdefault(amp, []).append(str(v))
            x = mu_map.get(str(v), float("nan"))
            if np.isfinite(x) and x <= low:
                d = "LOW"
            elif np.isfinite(x) and x >= high:
                d = "HIGH"
            else:
                d = "EXTREME"
            amp_to_variant_dirs.setdefault(amp, {})[str(v)] = d

        candidates.append(
            {
                "chr_arm": chr_arm,
                "n_variants": int(len(run_vars)),
                "n_amplicons": n_amps,
                "amplicons": ",".join(sorted(amps)),
                "amplicon_ids": sorted(amps),
                "amplicon_to_variants": amp_to_vars,
                "amplicon_to_variant_dirs": amp_to_variant_dirs,
                "block_span": _format_variant_block_span(vars_sorted, list(map(int, run_indices)), chr_arm, state),
                "variant_ids": run_vars,
            }
        )

    if not candidates:
        return None

    # Prefer more amplicons, then more variants; this describes the strongest
    # final LOH block for the arm.
    return sorted(candidates, key=lambda d: (-d["n_amplicons"], -d["n_variants"], d["chr_arm"]))[0]


def _binom_sf_geq(k: int, n: int, p: float) -> float:
    """
    Exact binomial survival P[X >= k] for small/moderate k, stable enough for
    Phase 5 QC alerts. Returns 1.0 for k <= 0 and 0.0 for impossible cases.
    """
    k = int(k)
    n = int(n)
    p = float(p)

    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    # Sum lower tail P[X < k] using a recurrence from P0.
    # This avoids huge binomial coefficients.
    q = 1.0 - p
    try:
        prob = q ** n
    except OverflowError:
        prob = math.exp(n * math.log(q))

    cdf = prob
    for i in range(1, k):
        # P_i = P_{i-1} * ((n - i + 1) / i) * (p / q)
        prob *= ((n - i + 1) / i) * (p / q)
        cdf += prob

    sf = 1.0 - cdf
    if sf < 0.0 and sf > -1e-12:
        sf = 0.0
    if sf > 1.0 and sf < 1.0 + 1e-12:
        sf = 1.0
    return float(max(0.0, min(1.0, sf)))


def _base_lineage_size_for_clone(state: "PipelineState", clone_label: str) -> int:
    """
    Size of the original/base clone lineage in the final labels: all cells whose
    current label has the same get_base_clone(...) as clone_label. This is the
    practical denominator for asking whether an LOH-like subclone could arise
    from stochastic multi-amplicon ADO within its original clone background.
    """
    base = get_base_clone(str(clone_label))
    n = 0
    for lbl, cells in state.clone_to_cells.items():
        try:
            if get_base_clone(str(lbl)) == base:
                n += len(cells)
        except Exception:
            pass
    return int(max(n, len(state.clone_to_cells.get(clone_label, []))))


def _loh_confidence_from_min_amplicons(min_amplicons: int | None) -> tuple[str, str]:
    """Return final-clone LOH confidence category from minimum block amplicon support."""
    if min_amplicons is None:
        return "", ""
    try:
        n = int(min_amplicons)
    except Exception:
        return "", ""
    if n <= 3:
        return "low", "LOH block supported by 3 informative amplicons"
    if n == 4:
        return "mid", "LOH block supported by 4 informative amplicons"
    return "high", "LOH block supported by ≥5 informative amplicons"


def _background_cells_for_empirical_ado(
    state: "PipelineState",
    clone_label: str,
) -> list[str]:
    """
    Use same-base, currently non-LOH cells as empirical ADO background.

    This avoids inflating the ADO rate with final LOH subclones on the same arm.
    If no non-LOH same-base background exists, fall back to all same-base cells
    excluding the queried LOH clone.
    """
    base = get_base_clone(str(clone_label))
    primary: list[str] = []
    fallback: list[str] = []

    for lbl, cells in state.clone_to_cells.items():
        lbl_s = str(lbl)
        try:
            if get_base_clone(lbl_s) != base:
                continue
        except Exception:
            continue
        if lbl_s == str(clone_label):
            continue
        cells2 = [str(c) for c in cells if str(c) in state.df_numeric.index]
        fallback.extend(cells2)
        if _is_active_non_loh(lbl_s):
            primary.extend(cells2)

    return primary if primary else fallback


def _estimate_empirical_ado_rates_for_block(
    state: "PipelineState",
    cfg: "LOHConfig",
    clone_label: str,
    block: dict,
) -> dict:
    """
    Estimate per-amplicon ADO-like rates from same-base non-LOH background cells.

    A background cell is counted as ADO-like for an amplicon only among covered
    observations.  If a DP matrix is available on PipelineState, variants with
    DP < cfg.phase5_qc_alert_empirical_min_depth are treated as missingness and
    are not counted as ADO-like events.  This makes the Phase 5 QC alert a
    coverage-aware empirical estimate rather than AF-only extreme counting.

    Each background cell counts as ADO-like for an amplicon when at least
    ``phase5_qc_alert_empirical_amplicon_extreme_frac`` (default 1.0) of its
    covered variants match the block direction (LOW ≤ low threshold, HIGH ≥
    high threshold). Rates are Beta-smoothed
    ``(events + alpha) / (valid + alpha + beta)`` with defaults alpha=1,
    beta=9. The result is ``usable`` only if every amplicon has at least
    ``phase5_qc_alert_empirical_min_background_cells`` (default 50) valid
    cells; then ``p_cell`` is the product of the per-amplicon rates.
    """
    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))
    min_frac = float(getattr(cfg, "phase5_qc_alert_empirical_amplicon_extreme_frac", 1.0))
    min_depth = float(getattr(cfg, "phase5_qc_alert_empirical_min_depth", 10))
    use_dp = bool(getattr(cfg, "phase5_qc_alert_empirical_use_dp", True))
    alpha = float(getattr(cfg, "phase5_qc_alert_empirical_beta_alpha", 1.0))
    beta = float(getattr(cfg, "phase5_qc_alert_empirical_beta_beta", 9.0))

    dp_mat = getattr(state, "df_dp_numeric", None)
    has_dp = bool(
        use_dp
        and dp_mat is not None
        and getattr(dp_mat, "shape", (0, 0))[1] > 0
        and len(set(block.get("variants", []) or []).intersection(set(getattr(dp_mat, "columns", [])))) > 0
    )

    bg_cells = _background_cells_for_empirical_ado(state, clone_label)
    amp_to_vars = block.get("amplicon_to_variants", {}) or {}
    amp_to_dirs = block.get("amplicon_to_variant_dirs", {}) or {}

    rows: list[dict] = []
    rates: list[float] = []

    for amp in sorted(amp_to_vars):
        vars_amp = [str(v) for v in amp_to_vars.get(amp, []) if str(v) in state.df_numeric.columns]
        if not vars_amp:
            continue

        dirs = {str(k): str(v) for k, v in (amp_to_dirs.get(amp, {}) or {}).items()}
        n_valid = 0
        n_event = 0
        n_low_depth_or_missing = 0
        n_cells_seen = 0

        for c in bg_cells:
            if c not in state.df_numeric.index:
                continue
            n_cells_seen += 1
            vals = state.df_numeric.loc[c, vars_amp]
            dps = None
            if has_dp and c in dp_mat.index:
                dps = dp_mat.loc[c, vars_amp]

            n_covered = 0
            n_match = 0
            n_missing_here = 0
            for v, x in vals.items():
                # Coverage gate first. Low/no DP is missingness, not ADO.
                if dps is not None:
                    try:
                        dpv = float(dps.get(v, np.nan))
                    except Exception:
                        dpv = float("nan")
                    if (not math.isfinite(dpv)) or dpv < min_depth:
                        n_missing_here += 1
                        continue

                try:
                    xf = float(x)
                except Exception:
                    n_missing_here += 1
                    continue
                if not math.isfinite(xf):
                    n_missing_here += 1
                    continue

                n_covered += 1
                d = dirs.get(str(v), "EXTREME")
                if d == "LOW":
                    ok = xf <= low
                elif d == "HIGH":
                    ok = xf >= high
                else:
                    ok = (xf <= low) or (xf >= high)
                if ok:
                    n_match += 1

            if n_covered == 0:
                n_low_depth_or_missing += 1
                continue
            n_valid += 1
            if (n_match / n_covered) >= min_frac:
                n_event += 1

        raw_rate = (n_event / n_valid) if n_valid > 0 else float("nan")
        smooth_rate = ((n_event + alpha) / (n_valid + alpha + beta)) if n_valid > 0 else float("nan")
        rate = smooth_rate if math.isfinite(smooth_rate) else raw_rate

        if math.isfinite(rate):
            rate = max(0.0, min(1.0, float(rate)))
            rates.append(rate)

        rows.append(
            {
                "amplicon_id": str(amp),
                "n_background_cells": int(n_cells_seen),
                "n_background_cells_valid": int(n_valid),
                "n_background_low_depth_or_missing": int(n_low_depth_or_missing),
                "n_background_ado_like": int(n_event),
                "raw_rate": raw_rate,
                "smoothed_rate": smooth_rate,
                "used_rate": rate,
                "dp_used": bool(has_dp),
                "min_depth": float(min_depth) if has_dp else None,
                "variants": ",".join(vars_amp),
            }
        )

    min_bg = int(getattr(cfg, "phase5_qc_alert_empirical_min_background_cells", 50))
    usable = bool(rows) and all(int(r.get("n_background_cells_valid", 0)) >= min_bg for r in rows) and len(rates) > 0

    p_cell = float(np.prod(rates)) if usable else None

    return {
        "usable": usable,
        "background_n_cells": int(len(bg_cells)),
        "min_background_cells_required": int(min_bg),
        "dp_used": bool(has_dp),
        "min_depth": float(min_depth) if has_dp else None,
        "amplicon_rates": rows,
        "p_cell": p_cell,
    }


def _ado_plausibility_for_boundary_amp_alert(
    state: "PipelineState",
    cfg: "LOHConfig",
    clone_label: str,
    n_observed: int,
    block: dict,
) -> dict:
    """
    Explanatory ADO model for Phase 5 boundary-support alerts.

    By default this estimates per-block/per-amplicon ADO-like rates empirically
    from same-base non-LOH background cells. If empirical estimation is disabled
    or insufficient background cells are available, it falls back to the fixed
    per-amplicon rate cfg.phase5_qc_alert_ado_rate_per_amplicon.

    The observed LOH clone size is compared with Binomial(N, p_cell), where N
    is the base-lineage size; ``P(X >= observed)`` below
    ``phase5_qc_alert_ado_unlikely_pvalue`` (default 0.01) gives the verdict
    "unlikely from ADO alone".
    """
    n_amplicons = int(max(0, block.get("n_amplicons", 0)))
    n_observed = int(max(0, n_observed))
    n_original = _base_lineage_size_for_clone(state, clone_label)

    source = str(getattr(cfg, "phase5_qc_alert_ado_rate_source", "empirical_background")).lower()
    fixed_a = float(getattr(cfg, "phase5_qc_alert_ado_rate_per_amplicon", 0.10))
    fixed_a = max(0.0, min(1.0, fixed_a))

    empirical = {}
    used_source = "fixed"
    rates_text = ""

    if source in {"empirical", "empirical_background", "background"}:
        empirical = _estimate_empirical_ado_rates_for_block(state, cfg, clone_label, block)
        if empirical.get("usable") and empirical.get("p_cell") is not None:
            p_cell = float(empirical["p_cell"])
            used_source = "empirical_background"
            parts = []
            for r in empirical.get("amplicon_rates", []):
                rate = r.get("used_rate")
                raw = r.get("raw_rate")
                nev = r.get("n_background_ado_like")
                nv = r.get("n_background_cells_valid")
                if rate is not None and math.isfinite(float(rate)):
                    parts.append(f"{r['amplicon_id']}={float(rate):.3g} ({nev}/{nv} raw={float(raw):.3g})")
            rates_text = "; ".join(parts)
        else:
            p_cell = float(fixed_a ** n_amplicons) if n_amplicons > 0 else 1.0
            used_source = "fixed_fallback"
    else:
        p_cell = float(fixed_a ** n_amplicons) if n_amplicons > 0 else 1.0
        used_source = "fixed"

    p_cell = max(0.0, min(1.0, float(p_cell)))
    expected = float(n_original * p_cell)
    p_geq = _binom_sf_geq(n_observed, n_original, p_cell)

    cutoff = float(getattr(cfg, "phase5_qc_alert_ado_unlikely_pvalue", 0.01))
    if p_geq < cutoff:
        verdict = "unlikely from ADO alone"
    else:
        verdict = "statistically plausible under ADO; interpret cautiously"

    if used_source == "empirical_background":
        dp_note = "DP-filtered " if empirical.get("dp_used") else "AF-only "
        md = empirical.get("min_depth")
        md_note = f"(DP≥{md:g}) " if empirical.get("dp_used") and md is not None else ""
        explanation = (
            f"ADO model: empirical {dp_note}same-base non-LOH background rates {md_note}over this block; "
            f"p_cell≈{p_cell:.3g}; base-lineage N={n_original}, observed={n_observed}, "
            f"expected≈{expected:.2f}, P(X≥observed)≈{p_geq:.3g} → {verdict}"
        )
        if rates_text:
            explanation += f". Per-amp rates: {rates_text}"
    elif used_source == "fixed_fallback":
        bg_n = empirical.get("background_n_cells") if empirical else None
        min_bg = empirical.get("min_background_cells_required") if empirical else None
        dp_note = "DP-filtered " if empirical.get("dp_used") else "AF-only "
        explanation = (
            f"ADO model: empirical {dp_note}background unavailable/insufficient "
            f"(background cells={bg_n}, min={min_bg}); falling back to {fixed_a:.0%} "
            f"independent ADO per amplicon. {n_amplicons}amp block gives "
            f"p_cell={p_cell:.3g}; base-lineage N={n_original}, observed={n_observed}, "
            f"expected≈{expected:.2f}, P(X≥observed)≈{p_geq:.3g} → {verdict}"
        )
    else:
        explanation = (
            f"ADO model: assuming {fixed_a:.0%} independent ADO per amplicon, "
            f"{n_amplicons}amp block requires p_cell={p_cell:.3g}; "
            f"base-lineage N={n_original}, observed={n_observed}, "
            f"expected≈{expected:.2f}, P(X≥observed)≈{p_geq:.3g} → {verdict}"
        )

    return {
        "ado_rate_source": used_source,
        "ado_assumed_amplicon_rate": fixed_a,
        "ado_required_amplicons": n_amplicons,
        "ado_empirical_background_cells": empirical.get("background_n_cells") if empirical else None,
        "ado_empirical_dp_used": empirical.get("dp_used") if empirical else False,
        "ado_empirical_min_depth": empirical.get("min_depth") if empirical else None,
        "ado_empirical_amplicon_rates": empirical.get("amplicon_rates", []) if empirical else [],
        "ado_cell_probability": p_cell,
        "ado_original_clone_size": n_original,
        "ado_observed_loh_cells": n_observed,
        "ado_expected_cells": expected,
        "ado_pvalue_ge_observed": p_geq,
        "ado_verdict": verdict,
        "ado_explanation": explanation,
    }


def _final_clone_loh_block_qc(state: "PipelineState", cfg: "LOHConfig", clone_label: str) -> tuple[str, str, int | None, dict]:
    """
    Return (details, qc_alert, min_amplicons, ado_qc) for final LOH clone summary.

    A passing LOH block supported by exactly cfg.phase5_qc_alert_amplicons
    independent informative amplicons is not dropped; it is labelled as a QC
    alert because it is at the minimum support boundary. ADO plausibility is
    calculated for every final LOH confidence tier by default, so mid/high
    calls receive the same empirical-background QC readout as low calls.
    """
    if not bool(getattr(cfg, "phase5_alert_three_amplicon_loh_blocks", True)):
        return "", "", None, {}

    alert_n = int(getattr(cfg, "phase5_qc_alert_amplicons", 3))
    blocks: list[dict] = []
    for arm in sorted(set(extract_chr_arms(clone_label))):
        b = _best_loh_block_support_for_clone_arm(state, cfg, clone_label, arm)
        if b is not None:
            blocks.append(b)

    if not blocks:
        return "", "", None, {}

    details = "; ".join(
        f"{b['chr_arm']}:{b['block_span']} ({b['n_variants']}v/{b['n_amplicons']}amp)"
        for b in blocks
    )
    min_amp = min(int(b["n_amplicons"]) for b in blocks)

    alert_blocks = [b for b in blocks if int(b["n_amplicons"]) == alert_n]
    alerts = [
        f"{b['chr_arm']} block spans only {alert_n} informative amplicons"
        for b in alert_blocks
    ]

    ado_qc: dict = {}
    if bool(getattr(cfg, "phase5_qc_alert_explain_ado_plausibility", True)):
        # Default: calculate an empirical ADO plausibility estimate for all
        # confidence tiers. For multi-arm clones, report the most conservative
        # individual arm/block result (largest P[X≥observed]) rather than
        # overstating evidence through an unmodelled cross-arm independence
        # assumption.
        use_all_blocks = bool(getattr(cfg, "phase5_qc_explain_ado_all_loh_confidences", True))
        qc_blocks = blocks if use_all_blocks else alert_blocks
        if qc_blocks:
            n_obs = len(state.clone_to_cells.get(clone_label, []))
            qc_candidates = [
                _ado_plausibility_for_boundary_amp_alert(
                    state=state,
                    cfg=cfg,
                    clone_label=clone_label,
                    n_observed=n_obs,
                    block=b,
                )
                for b in qc_blocks
            ]
            ado_qc = sorted(
                qc_candidates,
                key=lambda d: float(d.get("ado_pvalue_ge_observed") or 0.0),
                reverse=True,
            )[0]
            ado_qc["ado_qc_scope"] = "all_loh_blocks" if use_all_blocks else "boundary_blocks_only"

    return details, "; ".join(alerts), min_amp, ado_qc


def _yes_flags(row: pd.Series) -> tuple[str, str, str]:
    """Return the Hem?/TP53?/17p? table cells (" Yes" or four blanks) for a row."""
    hem_str = " Yes" if row["is_hem"] else "    "
    tp53_str = " Yes" if row["is_tp53"] else "    "
    p17_str = " Yes" if row["has_chr17p_loh"] else "    "
    return hem_str, tp53_str, p17_str


def run_phase5(
    state: "PipelineState",
    cfg: "LOHConfig | None" = None,
    *,
    loh_min_cells: int | None = None,
    heading: str = "PHASE 5: LOH CLONE SUMMARY",
) -> pd.DataFrame:
    """
    Build an LOH clone summary table and check for Hem-TP53 clones
    missing chr17p LOH.

    Parameters
    ----------
    state : PipelineState
        Pipeline state after Phase 4. Its clone labels are modified in place
        by the pre-summary cleanup (similar-LOH merge, small-clone prune).
    cfg : LOHConfig, optional
        Configuration; defaults to ``state.config``.
    loh_min_cells : int, optional
        Optionally relaxes the small-clone prune floor for LOH labels only
        (used by the Phase 6 residual scan). When None, the standard
        ``min_final_clone_size`` floor applies to every label.
    heading : str, default "PHASE 5: LOH CLONE SUMMARY"
        Banner text; the part before ``:`` (title-cased) prefixes the cleanup
        message. A blank heading falls back to the default.

    Returns
    -------
    pandas.DataFrame
        One row per surviving clone (columns as in ``_SUMMARY_COLUMNS``).
        Also stored in ``state.results["phase5_summary"]``.

    Notes
    -----
    Sets ``state.ado_probability`` to an empty DataFrame. Prints (via
    ``vprint``) partition counts, per-category clone tables, detected LOH
    arms, confidence counts, QC alerts for blocks at exactly
    ``phase5_qc_alert_amplicons`` amplicons, ADO-plausibility readouts and a
    Hem-TP53 / chr17p consistency check.
    """
    if cfg is None:
        cfg = state.config

    # Kept from the original: unused, but fails fast if cfg lacks the field.
    min_size = cfg.min_final_clone_size  # noqa: F841

    heading = str(heading).strip() or _DEFAULT_HEADING
    phase_tag = heading.split(":", 1)[0].title()
    vprint("\n" + "=" * 60, cfg)
    vprint(heading, cfg)
    vprint("=" * 60, cfg)

    # ── Final LOH cleanup before summary/plotting ─────────────────────
    n_merged = _merge_similar_loh_clones_before_phase5(state, cfg)
    n_pruned = _prune_small_terminal_clones_before_phase5(
        state, cfg, loh_min_cells=loh_min_cells
    )

    if n_merged > 0 or n_pruned > 0:
        vprint(
            f"\n  {phase_tag} pre-summary cleanup: "
            f"{n_merged} cells merged, {n_pruned} terminal sub-threshold cells renamed small.",
            cfg,
        )

    # The ADO-combination probability model is no longer run here; WT/Het→
    # Hem/Hom ADO handling happens in Phase 4b. Keep the attribute, empty.
    state.ado_probability = pd.DataFrame()

    # ── Build per-clone records ───────────────────────────────────────
    records = []
    for label, cells in sorted(
        state.clone_to_cells.items(), key=lambda kv: -len(kv[1])
    ):
        n = len(cells)
        if n == 0:
            continue

        is_loh = _is_active_loh(label)
        is_sub = _is_subclonal_loh(label)
        is_small = is_small_label(label)

        arms = extract_chr_arms(label)
        is_hem = is_hem_like(label)
        is_tp53 = bool(re.search(r"TP53", str(label), flags=re.IGNORECASE))
        has_17p = "chr17p" in arms
        loh_block_details = ""
        loh_qc_alert = ""
        loh_block_min_amplicons = None
        loh_ado_qc = {}
        loh_confidence = ""
        loh_confidence_reason = ""
        if is_loh:
            loh_block_details, loh_qc_alert, loh_block_min_amplicons, loh_ado_qc = _final_clone_loh_block_qc(
                state,
                cfg,
                str(label),
            )
            loh_confidence, loh_confidence_reason = _loh_confidence_from_min_amplicons(loh_block_min_amplicons)

        records.append(
            {
                "clone_label": label,
                "n_cells": n,
                "is_loh": is_loh,
                "is_subclonal": is_sub,
                "is_small": is_small,
                "loh_arms": ", ".join(arms) if arms else "",
                "is_hem": is_hem,
                "is_tp53": is_tp53,
                "has_chr17p_loh": has_17p,
                "loh_block_details": loh_block_details,
                "loh_block_min_amplicons": loh_block_min_amplicons,
                "loh_confidence": loh_confidence,
                "loh_confidence_reason": loh_confidence_reason,
                "qc_alert": loh_qc_alert,
                "ado_alert_explanation": loh_ado_qc.get("ado_explanation", ""),
                "ado_rate_source": loh_ado_qc.get("ado_rate_source", ""),
                "ado_assumed_amplicon_rate": loh_ado_qc.get("ado_assumed_amplicon_rate"),
                "ado_empirical_background_cells": loh_ado_qc.get("ado_empirical_background_cells"),
                "ado_empirical_dp_used": loh_ado_qc.get("ado_empirical_dp_used"),
                "ado_empirical_min_depth": loh_ado_qc.get("ado_empirical_min_depth"),
                "ado_empirical_amplicon_rates": loh_ado_qc.get("ado_empirical_amplicon_rates"),
                "ado_cell_probability": loh_ado_qc.get("ado_cell_probability"),
                "ado_original_clone_size": loh_ado_qc.get("ado_original_clone_size"),
                "ado_observed_loh_cells": loh_ado_qc.get("ado_observed_loh_cells"),
                "ado_expected_cells": loh_ado_qc.get("ado_expected_cells"),
                "ado_pvalue_ge_observed": loh_ado_qc.get("ado_pvalue_ge_observed"),
                "ado_verdict": loh_ado_qc.get("ado_verdict", ""),
                "ado_qc_scope": loh_ado_qc.get("ado_qc_scope", ""),
            }
        )

    summary = pd.DataFrame(records)

    if summary.empty:
        summary = pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
        state.results["phase5_summary"] = summary
        vprint("\n  No surviving clones to summarize.", cfg)
        vprint(f"\n  Result keys: {sorted(state.results.keys())}", cfg)
        vprint("=" * 60 + "\n", cfg)
        return summary

    # ── Partition counts ──────────────────────────────────────────────
    mask_loh = summary["is_loh"]
    mask_sub = summary.get("is_subclonal", pd.Series(False, index=summary.index)).astype(bool)
    mask_small = summary.get("is_small", pd.Series(False, index=summary.index)).astype(bool)
    mask_non = ~(mask_loh | mask_sub | mask_small)
    n_loh = int(mask_loh.sum())
    n_loh_cells = int(summary.loc[mask_loh, "n_cells"].sum())
    n_sub = int(mask_sub.sum())
    n_sub_cells = int(summary.loc[mask_sub, "n_cells"].sum())
    n_small = int(mask_small.sum())
    n_small_cells = int(summary.loc[mask_small, "n_cells"].sum())
    n_non = int(mask_non.sum())
    n_non_cells = int(summary.loc[mask_non, "n_cells"].sum())
    total_cells = int(summary["n_cells"].sum())

    vprint(f"\n  Total clones:    {len(summary)}", cfg)
    vprint(f"  Total cells:     {total_cells}", cfg)
    vprint(f"  LOH clones:      {n_loh}  ({n_loh_cells} cells)", cfg)
    vprint(f"  SUB clones:      {n_sub}  ({n_sub_cells} cells)", cfg)
    vprint(f"  Small clones:    {n_small}  ({n_small_cells} cells)", cfg)
    vprint(f"  Non-LOH clones:  {n_non}  ({n_non_cells} cells)", cfg)

    # ── Detail table: LOH clones ──────────────────────────────────────
    header = (
        f"  {'Clone Label':<60s} {'Cells':>5s}  {'LOH?':>5s}  "
        f"{'Arms':<30s}  {'Conf':>6s}  {'Hem?':>5s}  {'TP53?':>5s}  {'17p?':>5s}"
    )
    sep = "  " + "-" * 140

    vprint("\n  ── LOH Clones ──", cfg)
    vprint(header, cfg)
    vprint(sep, cfg)

    loh_rows = summary[mask_loh].sort_values("n_cells", ascending=False)
    for _, row in loh_rows.iterrows():
        hem_str, tp53_str, p17_str = _yes_flags(row)
        conf_str = str(row.get("loh_confidence", "") or "")
        vprint(
            f"  {row['clone_label']:<60s} {row['n_cells']:>5d}   Yes  "
            f"{row['loh_arms']:<30s}  {conf_str:>6s}  {hem_str:>5s}  {tp53_str:>5s}  {p17_str:>5s}",
            cfg,
        )

    # ── Detail table: unresolved directional SUB clones ───────────────
    if n_sub > 0:
        vprint("\n  ── Unresolved SUB Clones ──", cfg)
        vprint(header, cfg)
        vprint(sep, cfg)
        sub_rows = summary[mask_sub].sort_values("n_cells", ascending=False)
        for _, row in sub_rows.iterrows():
            vprint(
                f"  {row['clone_label']:<60s} {row['n_cells']:>5d}   SUB  "
                f"{row['loh_arms']:<30s}  {'—':>6s}  {'':>5s}  {'':>5s}  {'':>5s}",
                cfg,
            )

    # ── Detail table: terminal small clones ───────────────────────────
    if n_small > 0:
        vprint("\n  ── Small Clones (< final size threshold) ──", cfg)
        vprint(header, cfg)
        vprint(sep, cfg)
        small_rows = summary[mask_small].sort_values("n_cells", ascending=False)
        for _, row in small_rows.iterrows():
            vprint(
                f"  {row['clone_label']:<60s} {row['n_cells']:>5d}  small  "
                f"{row['loh_arms']:<30s}  {'—':>6s}  {'':>5s}  {'':>5s}  {'':>5s}",
                cfg,
            )

    # ── Detail table: Non-LOH clones ─────────────────────────────────
    vprint("\n  ── Non-LOH Clones ──", cfg)
    vprint(header, cfg)
    vprint(sep, cfg)

    non_loh_rows = summary[mask_non].sort_values("n_cells", ascending=False)
    for _, row in non_loh_rows.iterrows():
        hem_str, tp53_str, p17_str = _yes_flags(row)
        vprint(
            f"  {row['clone_label']:<60s} {row['n_cells']:>5d}     —  "
            f"{'—':<30s}  {'—':>6s}  {hem_str:>5s}  {tp53_str:>5s}  {p17_str:>5s}",
            cfg,
        )

    # ── LOH arms detected ────────────────────────────────────────────
    all_arms: set[str] = set()
    for a in summary.loc[mask_loh | mask_sub, "loh_arms"]:
        if a:
            all_arms.update(x.strip() for x in a.split(","))
    if all_arms:
        vprint(
            f"\n  LOH arms detected: "
            f"{', '.join(sort_chromosome_arms(list(all_arms)))}",
            cfg,
        )

    # ── LOH confidence by independent amplicon support ────────────────
    if n_loh > 0 and "loh_confidence" in summary.columns:
        conf_counts = (
            summary.loc[mask_loh, "loh_confidence"]
            .fillna("")
            .astype(str)
            .replace("", "unknown")
            .value_counts()
            .to_dict()
        )
        order = ["low", "mid", "high", "unknown"]
        parts = [f"{k}={conf_counts[k]}" for k in order if k in conf_counts]
        if parts:
            vprint(
                "\n  LOH confidence: " + ", ".join(parts)
                + "  (low=3 amps, mid=4 amps, high≥5 amps)",
                cfg,
            )

    # ── LOH QC alerts for minimum-amplicon blocks ─────────────────────
    qc_alert_rows = summary[
        summary["qc_alert"].fillna("").astype(str).str.len() > 0
    ].copy()
    if len(qc_alert_rows) > 0:
        vprint(f"\n  {_YELLOW}⚠  LOH QC ALERTS{_RESET}", cfg)
        for _, row in qc_alert_rows.iterrows():
            details = str(row.get("loh_block_details", ""))
            alert = str(row.get("qc_alert", ""))
            vprint(
                f"    • {row['clone_label']}  ({int(row['n_cells'])} cells): "
                f"{alert} [{details}]",
                cfg,
            )
            ado_msg = str(row.get("ado_alert_explanation", ""))
            if ado_msg:
                vprint(f"      {_YELLOW}{ado_msg}{_RESET}", cfg)
        vprint(
            "    These clones pass the minimum amplicon-support rule but sit at "
            "the 3-amplicon QC boundary.",
            cfg,
        )

    # ── ADO plausibility QC for non-boundary mid/high calls ───────────
    ado_report_rows = summary[
        mask_loh
        & summary["ado_alert_explanation"].fillna("").astype(str).str.len().gt(0)
        & summary["qc_alert"].fillna("").astype(str).str.len().eq(0)
    ].copy()
    if len(ado_report_rows) > 0:
        vprint("\n  ℹ  LOH ADO PLAUSIBILITY QC", cfg)
        for _, row in ado_report_rows.sort_values("n_cells", ascending=False).iterrows():
            conf = str(row.get("loh_confidence", "") or "unknown")
            details = str(row.get("loh_block_details", ""))
            vprint(
                f"    • {row['clone_label']}  ({int(row['n_cells'])} cells; "
                f"confidence={conf}) [{details}]",
                cfg,
            )
            vprint(f"      {_YELLOW}{row['ado_alert_explanation']}{_RESET}", cfg)

    # ── Hem-TP53 without chr17p LOH warning ───────────────────────────
    hem_tp53_no17p = summary[
        summary["is_hem"]
        & summary["is_tp53"]
        & ~summary["has_chr17p_loh"]
        & ~mask_small
    ]

    if len(hem_tp53_no17p) > 0:
        vprint(f"\n  {_YELLOW}⚠⚠⚠  WARNING  ⚠⚠⚠{_RESET}", cfg)
        vprint(
            "  The following Hem-TP53 clone(s) do NOT have LOH on chr17p:",
            cfg,
        )
        for _, row in hem_tp53_no17p.iterrows():
            tag = "(LOH clone)" if row["is_loh"] else "(non-LOH)"
            vprint(
                f"    • {row['clone_label']}  ({row['n_cells']} cells)  {tag}",
                cfg,
            )
        vprint(
            "  Hemizygous TP53 clones are typically expected to show LOH on chr17p.",
            cfg,
        )
        vprint(
            "  Double check chr17p informative variant allele frequencies,",
            cfg,
        )
        vprint("  or discard clone as likely FP.", cfg)
    else:
        any_hem_tp53 = summary[summary["is_hem"] & summary["is_tp53"]]
        if len(any_hem_tp53) > 0:
            vprint(
                f"\n  {_GREEN}✓ All Hem-TP53 clones have chr17p LOH.{_RESET}",
                cfg,
            )
        else:
            vprint("\n  ℹ  No Hem-TP53 clones detected.", cfg)

    # ── Store on state ────────────────────────────────────────────────
    state.results["phase5_summary"] = summary

    vprint(f"\n  Result keys: {sorted(state.results.keys())}", cfg)
    vprint("=" * 60 + "\n", cfg)

    return summary
