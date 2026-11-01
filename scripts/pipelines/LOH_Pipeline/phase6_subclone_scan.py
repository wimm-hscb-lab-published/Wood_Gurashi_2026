"""Targeted post-Phase-5 LOH subclone discovery.

This optional diagnostic/refinement pass scans one or more *final* non-LOH
clone annotations independently after the ordinary Phase 0–5 workflow has
finished.  It is deliberately not a full pipeline restart:

* pre-existing LOH/SUB/FP/small populations are not reconsidered;
* only explicitly requested final parent labels are eligible;
* each parent is scanned with a frozen copy of the Phase-5 labels, so new LOH
  children retain their final parent identity;
* detection uses the regular per-arm LOH engine, with its final clone-size and
  homogeneity gates enabled;
* Phase 4 broad merging is not re-run. Primary Phase-6 children may retain
  highly concordant members and nested secondary-arm evidence without creating
  flat-label sub-splits.

Typical use after ``out = LOH_Pipeline.run(...)``::

    post = LOH_Pipeline.run_post_phase5_subclone_scan(
        state=out["state"],
        clone_labels=["TP53_R248W_Hem"],
        rerun_phase5_summary=True,
    )

The function modifies ``state.Sample.dna.row_attrs['Clone']`` in place and
returns a per-parent/per-arm audit table.  Replot with
``LOH_Pipeline.plot_loh_clones(...)`` or re-run the wrapper's plotting step.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .detect_loh import detect_loh_and_update_clones
from .loh_config import (
    ADO_LABEL,
    check_directional_consistency,
    edge_tolerant_extreme_runs,
    is_fp_label,
    is_loh_label,
    is_small_label,
    is_sub_label,
)
from .phase5 import run_phase5

# Amplicon values treated as "no amplicon annotation".
_MISSING_AMPLICON_VALUES = (None, "", "nan", "None")


# ── small shared helpers ─────────────────────────────────────────────
def _sample_idx_for_cell(state, cell_id):
    """Positional Sample index of a DataFrame cell id (via its barcode), or None."""
    return state.sample_idx_for_barcode(str(state.cell_to_barcode.get(cell_id, cell_id)))


def _cells_with_label(state, cells, labels: pd.Series, label) -> list:
    """Cells (in input order) whose positional label in ``labels`` equals ``str(label)``."""
    out = []
    for cell_id in cells:
        sample_idx = _sample_idx_for_cell(state, cell_id)
        if sample_idx is not None and str(labels.iloc[sample_idx]) == str(label):
            out.append(cell_id)
    return out


def _min_child_size(cfg) -> int:
    """Phase-6 child size floor: ``phase6_residual_min_child_size`` else ``min_final_clone_size`` (5)."""
    return int(
        getattr(
            cfg,
            "phase6_residual_min_child_size",
            getattr(cfg, "min_final_clone_size", 5),
        )
    )


def _to_float_array(values) -> np.ndarray:
    """Coerce a Series/DataFrame of AFs to a float ndarray (non-numeric → NaN)."""
    return values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _clone_label_series(state) -> pd.Series:
    """Current ``Clone`` row attribute as a string Series (positional index)."""
    return pd.Series(state.Sample.dna.row_attrs["Clone"]).astype(str)


def _normalise_clone_labels(clone_labels: Iterable[str] | str | None) -> list[str]:
    """Return stripped, de-duplicated (order-kept) parent labels.

    A single string is treated as one label. Raises ValueError if
    ``clone_labels`` is None or contains no non-empty label.
    """
    if clone_labels is None:
        raise ValueError(
            "clone_labels is required. Provide explicit final parent labels, "
            "for example ['TP53_R248W_Hem']."
        )
    if isinstance(clone_labels, str):
        clone_labels = [clone_labels]

    out: list[str] = []
    seen: set[str] = set()
    for label in clone_labels:
        label = str(label).strip()
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    if not out:
        raise ValueError("clone_labels did not contain a usable clone label.")
    return out


def _eligible_parent_label(label: str) -> tuple[bool, str]:
    """Return whether a final label can be rescanned as an independent parent."""
    s = str(label).strip()
    if not s:
        return False, "empty_label"
    if s == ADO_LABEL:
        return False, "ado_label"
    if is_small_label(s):
        return False, "small_label"
    if is_fp_label(s):
        return False, "possible_loh_fp_label"
    if is_sub_label(s):
        return False, "subclonal_loh_sub_label"
    if is_loh_label(s):
        return False, "already_active_loh"
    return True, "eligible"


def _parent_has_possible_loh_block(
    state,
    cfg,
    parent_cells,
    scan_arms,
    *,
    min_candidate_cells: int | None = None,
) -> bool:
    """Cheap heterogeneity pre-filter for auto parent selection.

    Returns True if at least ``min_candidate_cells`` individual cells in the
    parent carry a contiguous extreme run of >= min_contiguous_variants_initial
    variants on some scan arm. This is per-cell on purpose: a subclonal LOH
    block lives in only a fraction of the parent's cells, so a clone-mean test
    would wash it out (e.g. an LOH block in 7 of 17 cells leaves the mean near
    50% and looks "clean"). Full per-cell detection and the strict block QC
    still run inside the scan; this only skips parents with no plausible block.
    """
    cells = [c for c in parent_cells if c in state.df_numeric.index]
    min_block = int(getattr(cfg, "min_contiguous_variants_initial", 4))
    if len(cells) < min_block:
        return False

    if min_candidate_cells is None:
        min_candidate_cells = max(2, _min_child_size(cfg))

    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))

    for chr_arm in scan_arms:
        arm_vars = [
            v for v in state.grouped_by_chr_arm.get(chr_arm, [])
            if v in state.df_numeric.columns
        ]
        if len(arm_vars) < min_block:
            continue
        mat = _to_float_array(state.df_numeric.loc[cells, arm_vars])
        # NaN -> not extreme, which also breaks a contiguous run (conservative).
        extreme = (mat <= low) | (mat >= high)
        n_candidate = 0
        for row in extreme:
            longest = current = 0
            for is_ext in row:
                if is_ext:
                    current += 1
                    if current > longest:
                        longest = current
                else:
                    current = 0
            if longest >= min_block:
                n_candidate += 1
                if n_candidate >= min_candidate_cells:
                    return True
    return False


def _auto_select_scan_parents(
    state,
    cfg,
    scan_arms,
    *,
    min_parent_cells: int,
    exclude_labels: Iterable[str] | None = None,
    include_wt: bool = False,
    require_possible_block: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Auto-pick parents for the residual scan.

    Selects eligible non-LOH base clones (not LOH/SUB/FP/small/ADO) with at
    least ``min_parent_cells`` cells. The dominant exact ``WT`` parent is
    excluded by default because a broad residual scan there produces many
    low-prevalence technical partitions; pass ``include_wt=True`` only for an
    explicit WT-focused investigation. With ``require_possible_block=True``
    (``"auto"`` mode) only clones that still carry a possible LOH block on some
    scan arm are kept. With ``require_possible_block=False`` (``"all"`` mode)
    every eligible parent at/above the size floor is scanned, regardless of the
    pre-screen — the full detection + strict QC inside the scan then decides.
    """
    exclude = {str(x).strip() for x in (exclude_labels or [])}
    state.refresh()
    counts = state.current_clones.value_counts()

    candidates: list[str] = []
    skipped_clean: list[str] = []
    for label, n in counts.items():
        label = str(label)
        if label in exclude:
            continue
        if not include_wt and label.strip().upper() == "WT":
            continue
        ok, _reason = _eligible_parent_label(label)
        if not ok:
            continue
        if int(n) < int(min_parent_cells):
            continue
        if not require_possible_block:
            candidates.append(label)
            continue
        parent_cells = [
            cell_id
            for cell_id in state.df.index
            if str(state.barcode_to_clone.get(
                str(state.cell_to_barcode.get(cell_id, cell_id)), "",
            )) == label
        ]
        if _parent_has_possible_loh_block(state, cfg, parent_cells, scan_arms):
            candidates.append(label)
        else:
            skipped_clean.append(label)

    # Largest parents first: discovering children there frees the most cells.
    candidates = sorted(candidates, key=lambda l: -int(counts.get(l, 0)))

    if verbose:
        mode = "all eligible" if not require_possible_block else "possible-LOH"
        print(
            f"  Auto-selected {len(candidates)} parent clone(s) ({mode}, "
            f"n≥{min_parent_cells}): {candidates}"
        )
        if skipped_clean:
            print(
                f"  Skipped {len(skipped_clean)} eligible parent(s) with no "
                f"possible LOH block: {skipped_clean}"
            )
    return candidates


def _variant_position(variant: str) -> int:
    """Genomic start of ``chr:pos[-end]:...`` ids; unparseable ids sort last (10**18)."""
    try:
        return int(str(variant).split(":")[1].split("-")[0])
    except Exception:
        return 10**18


def _variant_to_amplicon(state) -> dict[str, str]:
    """Map variant id → amplicon from DNA ``col_attrs``; {} if unavailable or misaligned."""
    try:
        ids = list(state.Sample.dna.col_attrs.get("id", []))
        amps = list(state.Sample.dna.col_attrs.get("amplicon", []))
        if len(ids) == len(amps) and ids:
            return {str(v): str(a) for v, a in zip(ids, amps)}
    except Exception:
        pass
    return {}


def _remove_arm_token(label: str, chr_arm: str) -> str:
    """Remove the post-scan LOH token for exactly one chromosome arm."""
    out = re.sub(
        rf"\s+LOH-C\d+\s+{re.escape(str(chr_arm))}(?::\S+)?",
        "",
        str(label),
    ).strip()
    return re.sub(r"\s+", " ", out)


def _sample_row_attr_array(state, key: str, *, fill_value: str = "") -> np.ndarray:
    """Return a writable, row-aligned object array for a DNA row attribute."""
    n = len(state.Sample.dna.row_attrs["Clone"])
    existing = state.Sample.dna.row_attrs.get(key)
    if existing is None:
        arr = np.full(n, fill_value, dtype=object)
    else:
        arr = np.asarray(existing, dtype=object).copy()
        if len(arr) != n:
            arr = np.full(n, fill_value, dtype=object)
    return arr


def _write_sample_row_attr(state, key: str, arr: np.ndarray) -> None:
    """Store ``arr`` as an object array in ``Sample.dna.row_attrs[key]``."""
    state.Sample.dna.row_attrs[key] = np.asarray(arr, dtype=object)


def _append_semicolon_token(value, token: str) -> str:
    """Append a unique metadata token without duplicating prior evidence."""
    token = str(token).strip()
    existing = [x.strip() for x in str(value or "").split(";") if x.strip()]
    if token and token not in existing:
        existing.append(token)
    return ";".join(existing)


def _set_phase6_primary_metadata(
    state,
    cell_ids,
    *,
    child_label: str,
    chr_arm: str,
    member_role: str,
) -> None:
    """Persist primary-child membership without changing flat clone labels."""
    primary_child = _sample_row_attr_array(state, "phase6_primary_loh_child")
    primary_arm = _sample_row_attr_array(state, "phase6_primary_loh_arm")
    member_role_arr = _sample_row_attr_array(state, "phase6_primary_loh_member_role")

    for cell_id in cell_ids:
        sample_idx = _sample_idx_for_cell(state, cell_id)
        if sample_idx is None:
            continue
        primary_child[sample_idx] = str(child_label)
        primary_arm[sample_idx] = str(chr_arm)
        member_role_arr[sample_idx] = str(member_role)

    _write_sample_row_attr(state, "phase6_primary_loh_child", primary_child)
    _write_sample_row_attr(state, "phase6_primary_loh_arm", primary_arm)
    _write_sample_row_attr(state, "phase6_primary_loh_member_role", member_role_arr)


def _record_phase6_secondary_arm_evidence(state, cell_ids, chr_arm: str) -> None:
    """Record an additional LOH arm without fragmenting the primary child."""
    secondary = _sample_row_attr_array(state, "phase6_secondary_loh_arms")
    for cell_id in cell_ids:
        sample_idx = _sample_idx_for_cell(state, cell_id)
        if sample_idx is None:
            continue
        secondary[sample_idx] = _append_semicolon_token(secondary[sample_idx], chr_arm)
    _write_sample_row_attr(state, "phase6_secondary_loh_arms", secondary)


def _safe_profile_similarity(
    row: np.ndarray,
    representative: np.ndarray,
    *,
    min_variants: int,
    constant_profile_max_abs_delta: float,
) -> tuple[float, float, int]:
    """Return correlation, median |delta|, and comparable positions."""
    mask = np.isfinite(row) & np.isfinite(representative)
    n = int(mask.sum())
    if n < int(min_variants):
        return np.nan, np.nan, n
    x = row[mask]
    y = representative[mask]
    med_abs_delta = float(np.nanmedian(np.abs(x - y)))
    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
        r = 1.0 if med_abs_delta <= float(constant_profile_max_abs_delta) else 0.0
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    return r, med_abs_delta, n


def _expand_primary_child_with_similar_members(
    state,
    *,
    parent_label: str,
    parent_cells,
    child_label: str,
    chr_arm: str,
) -> pd.DataFrame:
    """Expand a strict primary child with evidence-backed near-representatives.

    The strict Phase-6 validator first establishes a genuine LOH core. This
    expansion does *not* turn a weak pattern into an LOH clone. It only adds
    cells that remain close to the core representative over the accepted block,
    retain substantial extreme-AF evidence, and agree with the core's signed
    allelic direction at their informative block positions.

    Notes
    -----
    Only the strict core is validated (``_strict_arm_metrics_for_cells``).
    Every residual-parent cell that passes the per-cell gates (default:
    r ≥ 0.85 to the core median profile, median |ΔAF| ≤ 15, block extreme
    fraction ≥ 0.75, signed agreement ≥ 0.75 over at least
    ``max(2, ceil(0.75 * n_block))`` signed positions) is relabelled to the
    child; the combined child is *not* re-validated after additions, and
    ``strict_child_pass_after_addition`` is simply set True for accepted rows.
    Returns one audit row per evaluated parent cell.
    """
    cfg = state.config
    state.refresh()
    columns = [
        "parent_clone", "child_label", "chr_arm", "barcode",
        "member_action", "reason", "representative_corr",
        "median_abs_delta", "n_compared_variants",
        "cell_block_extreme_fraction", "signed_agreement",
        "n_signed_positions", "strict_child_pass_after_addition",
    ]

    if not bool(getattr(cfg, "phase6_expand_child_members", True)):
        return pd.DataFrame(columns=columns)

    parent_cells = [c for c in parent_cells if c in state.df_numeric.index]
    labels = state.current_clones.copy().astype(str)
    core_cells = _cells_with_label(state, parent_cells, labels, child_label)
    if not core_cells:
        return pd.DataFrame(columns=columns)

    strict_core = _strict_arm_metrics_for_cells(state, core_cells, chr_arm)
    if not bool(strict_core.get("strict_pass", False)):
        return pd.DataFrame([{
            "parent_clone": parent_label,
            "child_label": child_label,
            "chr_arm": chr_arm,
            "barcode": "",
            "member_action": "not_expanded",
            "reason": "primary_core_failed_strict_qc:" + str(strict_core.get("reason", "")),
            "representative_corr": np.nan,
            "median_abs_delta": np.nan,
            "n_compared_variants": 0,
            "cell_block_extreme_fraction": np.nan,
            "signed_agreement": np.nan,
            "n_signed_positions": 0,
            "strict_child_pass_after_addition": False,
        }], columns=columns)

    block_vars = [v for v in str(strict_core.get("block_variants", "")).split(";") if v]
    if not block_vars:
        return pd.DataFrame(columns=columns)

    core_values = _to_float_array(state.df_numeric.loc[core_cells, block_vars])
    representative = np.nanmedian(core_values, axis=0)
    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))
    ref_low = np.nansum(core_values <= low, axis=0)
    ref_high = np.nansum(core_values >= high, axis=0)
    representative_sign = np.where(ref_low >= ref_high, -1, 1)

    min_compared = int(getattr(
        cfg, "phase6_expand_min_compared_variants",
        getattr(cfg, "phase3_min_variants_for_cell_corr", 4),
    ))
    min_corr = float(getattr(cfg, "phase6_expand_min_corr_to_representative", 0.85))
    max_delta = float(getattr(cfg, "phase6_expand_max_median_abs_delta", 15.0))
    min_extreme = float(getattr(cfg, "phase6_expand_min_block_extreme_fraction", 0.75))
    min_signed_agreement = float(getattr(cfg, "phase6_expand_min_signed_agreement", 0.75))
    constant_delta = float(getattr(cfg, "phase6_expand_constant_profile_max_abs_delta", max_delta))

    candidate_rows: list[dict] = []
    candidates: list[tuple[float, float, object, dict]] = []
    for cell_id in parent_cells:
        barcode = str(state.cell_to_barcode.get(cell_id, cell_id))
        sample_idx = state.sample_idx_for_barcode(barcode)
        if sample_idx is None or str(labels.iloc[sample_idx]) != str(parent_label):
            continue
        row = _to_float_array(state.df_numeric.loc[cell_id, block_vars])
        r, med_delta, n_compared = _safe_profile_similarity(
            row,
            representative,
            min_variants=min_compared,
            constant_profile_max_abs_delta=constant_delta,
        )
        finite = np.isfinite(row)
        signs = np.where(row <= low, -1, np.where(row >= high, 1, 0))
        extreme_fraction = float(np.mean(np.abs(signs[finite]) > 0)) if bool(finite.any()) else 0.0
        signed_mask = finite & (signs != 0) & (representative_sign != 0)
        n_signed = int(signed_mask.sum())
        signed_agreement = (
            float(np.mean(signs[signed_mask] == representative_sign[signed_mask]))
            if n_signed else 0.0
        )
        reasons = []
        if n_compared < min_compared:
            reasons.append(f"insufficient_compared_variants:{n_compared}<{min_compared}")
        if not np.isfinite(r) or r < min_corr:
            reasons.append(f"representative_corr:{r:.3f}<{min_corr:.3f}" if np.isfinite(r) else "representative_corr:NA")
        if not np.isfinite(med_delta) or med_delta > max_delta:
            reasons.append(f"median_abs_delta:{med_delta:.3f}>{max_delta:.3f}" if np.isfinite(med_delta) else "median_abs_delta:NA")
        if extreme_fraction < min_extreme:
            reasons.append(f"cell_block_extreme_fraction:{extreme_fraction:.3f}<{min_extreme:.3f}")
        min_signed_positions = max(
            2,
            int(np.ceil(min_extreme * max(len(block_vars), 1))),
        )
        if n_signed < min_signed_positions:
            reasons.append(
                f"insufficient_signed_positions:{n_signed}<{min_signed_positions}"
            )
        elif signed_agreement < min_signed_agreement:
            reasons.append(f"signed_agreement:{signed_agreement:.3f}<{min_signed_agreement:.3f}")

        record = {
            "parent_clone": parent_label,
            "child_label": child_label,
            "chr_arm": chr_arm,
            "barcode": barcode,
            "member_action": "rejected_similarity",
            "reason": ";".join(reasons) if reasons else "candidate_passes_similarity",
            "representative_corr": r,
            "median_abs_delta": med_delta,
            "n_compared_variants": n_compared,
            "cell_block_extreme_fraction": extreme_fraction,
            "signed_agreement": signed_agreement,
            "n_signed_positions": n_signed,
            "strict_child_pass_after_addition": False,
        }
        candidate_rows.append(record)
        if not reasons:
            # Highest correlation / lowest delta first keeps the core stable.
            candidates.append((-float(r), float(med_delta), cell_id, record))

    if not candidates:
        return pd.DataFrame(candidate_rows, columns=columns)

    # The strict core is the LOH call. Expansion members are attached by
    # individual evidence against that core, so a single mild dropout cannot
    # erase the core's >=90% per-variant support. This deliberately avoids
    # re-evaluating directional/block fractions on the mixed core+associate
    # family, which would otherwise recreate the all-members-must-be-perfect
    # rule that Phase 6 is intended to relax.
    accepted_records: list[dict] = []
    candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
    for _neg_r, _delta, cell_id, record in candidates:
        accepted_records.append(record)
        record["member_action"] = "expanded_similarity"
        record["reason"] = "accepted_similarity_to_strict_core"
        record["strict_child_pass_after_addition"] = True

    if accepted_records:
        updated = state.current_clones.copy().astype(str)
        for _neg_r, _delta, cell_id, record in candidates:
            if record.get("member_action") != "expanded_similarity":
                continue
            sample_idx = _sample_idx_for_cell(state, cell_id)
            if sample_idx is not None and str(updated.iloc[sample_idx]) == str(parent_label):
                updated.iloc[sample_idx] = str(child_label)
        state.set_clone_labels(updated)
        _set_phase6_primary_metadata(
            state,
            [cell_id for _neg_r, _delta, cell_id, record in candidates if record.get("member_action") == "expanded_similarity"],
            child_label=child_label,
            chr_arm=chr_arm,
            member_role="expanded_similarity",
        )

    return pd.DataFrame(candidate_rows, columns=columns)


def _contiguous_runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous True runs from a 1-D boolean mask."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _expand_primary_child_with_partial_block_members(
    state,
    *,
    parent_label: str,
    parent_cells,
    child_label: str,
    chr_arm: str,
) -> pd.DataFrame:
    """Attach near-core cells carrying a validated contiguous sub-block.

    A primary Phase-6 child is established *before* this function runs.  A
    residual-parent cell may then join only when it retains an independently
    convincing subset of that same LOH block: a contiguous directional run of
    at least the primary block minimum, sufficient independent amplicons, and
    close AF agreement with the established child over that exact sub-block.

    This is intentionally different from a broad residual re-scan.  It cannot
    create a child and cannot connect two unrelated extreme-AF cells through
    the clone mean.  It simply avoids requiring every genuine family member to
    span 100% of the strict-core block.
    """
    cfg = state.config
    state.refresh()
    columns = [
        "parent_clone", "child_label", "chr_arm", "barcode",
        "member_action", "reason", "representative_corr",
        "median_abs_delta", "n_compared_variants",
        "n_supported_block_variants", "supported_block_variants",
        "n_independent_amplicons", "signed_fraction",
        "strict_core_block_variants",
    ]
    if not bool(getattr(cfg, "phase6_expand_partial_block_members", True)):
        return pd.DataFrame(columns=columns)

    parent_cells = [c for c in parent_cells if c in state.df_numeric.index]
    labels = state.current_clones.copy().astype(str)
    core_cells = _cells_with_label(state, parent_cells, labels, child_label)
    if not core_cells:
        return pd.DataFrame(columns=columns)

    strict_core = _strict_arm_metrics_for_cells(state, core_cells, chr_arm)
    if not bool(strict_core.get("strict_pass", False)):
        return pd.DataFrame([{
            "parent_clone": parent_label,
            "child_label": child_label,
            "chr_arm": chr_arm,
            "barcode": "",
            "member_action": "not_expanded_partial_block",
            "reason": "primary_core_failed_strict_qc:" + str(strict_core.get("reason", "")),
            "representative_corr": np.nan,
            "median_abs_delta": np.nan,
            "n_compared_variants": 0,
            "n_supported_block_variants": 0,
            "supported_block_variants": "",
            "n_independent_amplicons": 0,
            "signed_fraction": np.nan,
            "strict_core_block_variants": str(strict_core.get("block_variants", "")),
        }], columns=columns)

    block_vars = [v for v in str(strict_core.get("block_variants", "")).split(";") if v]
    if not block_vars:
        return pd.DataFrame(columns=columns)

    core_values = _to_float_array(state.df_numeric.loc[core_cells, block_vars])
    representative = np.nanmedian(core_values, axis=0)
    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))
    core_low = np.nansum(core_values <= low, axis=0)
    core_high = np.nansum(core_values >= high, axis=0)
    core_sign = np.where(core_low >= core_high, -1, 1)

    min_block = max(
        int(getattr(cfg, "min_contiguous_variants_initial", 4)),
        int(getattr(cfg, "phase6_expand_partial_min_block_variants", 4)),
    )
    min_corr = float(getattr(cfg, "phase6_expand_partial_min_corr_to_representative", 0.80))
    max_delta = float(getattr(cfg, "phase6_expand_partial_max_median_abs_delta", 15.0))
    min_amps = int(getattr(
        cfg,
        "phase6_expand_partial_min_independent_amplicons",
        getattr(cfg, "min_different_amplicons", 3),
    ))
    min_signed_fraction = float(getattr(cfg, "phase6_expand_partial_min_signed_fraction", 1.00))
    constant_delta = float(getattr(
        cfg,
        "phase6_expand_partial_constant_profile_max_abs_delta",
        max_delta,
    ))
    v2a = _variant_to_amplicon(state)
    info_set = getattr(cfg, "informative_variants", None)
    info_set = set(map(str, info_set)) if info_set is not None else None

    rows: list[dict] = []
    accepted: list[tuple[float, float, object, dict]] = []
    for cell_id in parent_cells:
        barcode = str(state.cell_to_barcode.get(cell_id, cell_id))
        sample_idx = state.sample_idx_for_barcode(barcode)
        if sample_idx is None or str(labels.iloc[sample_idx]) != str(parent_label):
            continue

        row = _to_float_array(state.df_numeric.loc[cell_id, block_vars])
        signs = np.where(row <= low, -1, np.where(row >= high, 1, 0))
        signed_match = np.isfinite(row) & (signs != 0) & (signs == core_sign)
        runs = _contiguous_runs_from_mask(signed_match)
        eligible_runs = [(s, e) for s, e in runs if (e - s + 1) >= min_block]

        best: dict | None = None
        for start, end in eligible_runs:
            idx = np.arange(start, end + 1, dtype=int)
            run_row = row[idx]
            run_ref = representative[idx]
            r, med_delta, n_compared = _safe_profile_similarity(
                run_row,
                run_ref,
                min_variants=min_block,
                constant_profile_max_abs_delta=constant_delta,
            )
            run_finite = np.isfinite(run_row)
            run_signs = signs[idx]
            signed_fraction = float(
                np.mean((run_signs[run_finite] == core_sign[idx][run_finite]))
            ) if bool(run_finite.any()) else 0.0
            run_vars = [block_vars[i] for i in idx]
            amps = {
                str(v2a.get(v)) for v in run_vars
                if (info_set is None or v in info_set)
                and v2a.get(v) not in _MISSING_AMPLICON_VALUES
            }
            candidate = {
                "start": int(start),
                "end": int(end),
                "n_supported": int(len(idx)),
                "run_vars": run_vars,
                "r": r,
                "med_delta": med_delta,
                "n_compared": n_compared,
                "n_amps": int(len(amps)),
                "signed_fraction": signed_fraction,
            }
            if best is None or (
                candidate["n_supported"],
                np.nan_to_num(candidate["r"], nan=-np.inf),
                -np.nan_to_num(candidate["med_delta"], nan=np.inf),
            ) > (
                best["n_supported"],
                np.nan_to_num(best["r"], nan=-np.inf),
                -np.nan_to_num(best["med_delta"], nan=np.inf),
            ):
                best = candidate

        reasons: list[str] = []
        if best is None:
            reasons.append(f"no_signed_contiguous_subblock≥{min_block}")
            rec = {
                "parent_clone": parent_label,
                "child_label": child_label,
                "chr_arm": chr_arm,
                "barcode": barcode,
                "member_action": "rejected_partial_block",
                "reason": ";".join(reasons),
                "representative_corr": np.nan,
                "median_abs_delta": np.nan,
                "n_compared_variants": 0,
                "n_supported_block_variants": 0,
                "supported_block_variants": "",
                "n_independent_amplicons": 0,
                "signed_fraction": 0.0,
                "strict_core_block_variants": ";".join(block_vars),
            }
            rows.append(rec)
            continue

        if best["n_compared"] < min_block:
            reasons.append(f"insufficient_compared_variants:{best['n_compared']}<{min_block}")
        if not np.isfinite(best["r"]) or best["r"] < min_corr:
            reasons.append(
                f"representative_corr:{best['r']:.3f}<{min_corr:.3f}"
                if np.isfinite(best["r"]) else "representative_corr:NA"
            )
        if not np.isfinite(best["med_delta"]) or best["med_delta"] > max_delta:
            reasons.append(
                f"median_abs_delta:{best['med_delta']:.3f}>{max_delta:.3f}"
                if np.isfinite(best["med_delta"]) else "median_abs_delta:NA"
            )
        if best["n_amps"] < min_amps:
            reasons.append(f"independent_amplicons:{best['n_amps']}<{min_amps}")
        if best["signed_fraction"] < min_signed_fraction:
            reasons.append(
                f"signed_fraction:{best['signed_fraction']:.3f}<{min_signed_fraction:.3f}"
            )

        rec = {
            "parent_clone": parent_label,
            "child_label": child_label,
            "chr_arm": chr_arm,
            "barcode": barcode,
            "member_action": "rejected_partial_block",
            "reason": ";".join(reasons) if reasons else "candidate_passes_partial_block",
            "representative_corr": best["r"],
            "median_abs_delta": best["med_delta"],
            "n_compared_variants": best["n_compared"],
            "n_supported_block_variants": best["n_supported"],
            "supported_block_variants": ";".join(best["run_vars"]),
            "n_independent_amplicons": best["n_amps"],
            "signed_fraction": best["signed_fraction"],
            "strict_core_block_variants": ";".join(block_vars),
        }
        rows.append(rec)
        if not reasons:
            accepted.append((-float(best["r"]), float(best["med_delta"]), cell_id, rec))

    if accepted:
        accepted.sort(key=lambda x: (x[0], x[1], str(x[2])))
        updated = state.current_clones.copy().astype(str)
        accepted_cells = []
        for _neg_r, _delta, cell_id, rec in accepted:
            sample_idx = _sample_idx_for_cell(state, cell_id)
            if sample_idx is None or str(updated.iloc[sample_idx]) != str(parent_label):
                continue
            updated.iloc[sample_idx] = str(child_label)
            rec["member_action"] = "expanded_partial_block"
            rec["reason"] = "accepted_contiguous_signed_subblock_to_strict_core"
            accepted_cells.append(cell_id)
        if accepted_cells:
            state.set_clone_labels(updated)
            _set_phase6_primary_metadata(
                state,
                accepted_cells,
                child_label=child_label,
                chr_arm=chr_arm,
                member_role="expanded_partial_block",
            )

    return pd.DataFrame(rows, columns=columns)


def _collapse_secondary_arm_subset_to_primary_family(
    state,
    *,
    parent_cells,
    labels_before: pd.Series,
    chr_arm: str,
    passing_child_labels: set[str] | None = None,
) -> pd.DataFrame:
    """Keep a primary LOH family intact when a secondary arm is only a subset.

    Flat ``Clone`` labels cannot encode overlapping nested populations. If a
    validated secondary arm applies to only a subset of an already validated
    Phase-6 primary child, appending that arm would split one biological chr5q
    family into, for example, 5 chr5q+chr7p cells and 4 chr5q-only cells. The
    latter can then be incorrectly pruned despite a nine-cell shared chr5q
    LOH population. This helper preserves the primary label and records the
    extra arm as row-level nested evidence instead.
    """
    cfg = state.config
    state.refresh()
    columns = [
        "chr_arm", "primary_child_label", "secondary_child_label",
        "n_secondary_cells", "n_primary_family_cells", "secondary_fraction",
        "action", "barcodes",
    ]
    labels_after = state.current_clones.copy().astype(str)
    if labels_before is None or labels_after.empty:
        return pd.DataFrame(columns=columns)

    parent_sample_indices: set[int] = set()
    for cell_id in parent_cells:
        sample_idx = _sample_idx_for_cell(state, cell_id)
        if sample_idx is not None:
            parent_sample_indices.add(int(sample_idx))

    changed_idx = [
        idx for idx in parent_sample_indices
        if str(labels_after.iloc[idx]) != str(labels_before.iloc[idx])
    ]
    if not changed_idx:
        return pd.DataFrame(columns=columns)

    min_fraction = float(getattr(cfg, "phase6_secondary_arm_label_min_fraction", 0.90))
    passing_child_labels = {str(x) for x in (passing_child_labels or set())}
    rows: list[dict] = []
    reverted = False

    changed_by_after: dict[str, list[int]] = {}
    for idx in changed_idx:
        changed_by_after.setdefault(str(labels_after.iloc[idx]), []).append(idx)

    for secondary_label, indices in changed_by_after.items():
        if passing_child_labels and secondary_label not in passing_child_labels:
            continue
        primary_label = _remove_arm_token(secondary_label, chr_arm)
        # Only collapse a true nested secondary arm added onto an existing
        # primary child; never suppress an initial single-arm child call.
        if not is_loh_label(primary_label) or str(chr_arm) in str(primary_label):
            continue
        family_indices = [
            idx for idx in parent_sample_indices
            if str(labels_before.iloc[idx]) == str(primary_label)
        ]
        if not family_indices:
            continue
        frac = len(indices) / max(len(family_indices), 1)
        if frac >= min_fraction:
            continue

        for idx in indices:
            labels_after.iloc[idx] = primary_label
        reverted = True
        barcodes = [str(state.Sample.dna.row_attrs.get("barcode", [idx])[idx])
                    if "barcode" in state.Sample.dna.row_attrs else str(idx)
                    for idx in indices]
        # Use state mappings for the metadata update when available.
        index_set = set(indices)
        cell_ids = []
        for cell_id in parent_cells:
            sample_idx = _sample_idx_for_cell(state, cell_id)
            if sample_idx in index_set:
                cell_ids.append(cell_id)
        _record_phase6_secondary_arm_evidence(state, cell_ids, chr_arm)
        rows.append({
            "chr_arm": chr_arm,
            "primary_child_label": primary_label,
            "secondary_child_label": secondary_label,
            "n_secondary_cells": len(indices),
            "n_primary_family_cells": len(family_indices),
            "secondary_fraction": frac,
            "action": "recorded_as_nested_secondary_evidence",
            "barcodes": ";".join(barcodes),
        })

    if reverted:
        state.set_clone_labels(labels_after)
    return pd.DataFrame(rows, columns=columns)


def _strict_arm_metrics_for_cells(state, cells, chr_arm: str) -> dict:
    """Evaluate one explicit cell group on one arm with strict Phase-6 QC.

    The block is defined from cell-level extreme support rather than clone-mean
    AF. This preserves valid reciprocal LOH patterns: the ALT fraction may be
    LOW in one cell group and HIGH in another, while every individual cell is
    still extreme at the same germline loci.
    """
    cfg = state.config
    state.refresh()
    cells = [c for c in cells if c in state.df_numeric.index]
    arm_vars = sorted(
        [
            str(v) for v in state.grouped_by_chr_arm.get(chr_arm, [])
            if v in state.df_numeric.columns
        ],
        key=_variant_position,
    )
    out = {
        "chr_arm": chr_arm,
        "n_cells": len(cells),
        "strict_pass": False,
        "reason": "",
        "n_block_variants": 0,
        "block_variants": "",
        "block_extreme_fraction": np.nan,
        "mean_cell_to_mean_r": np.nan,
        "good_cell_fraction": 0.0,
        "direction": "NA",
        "direction_block_pass_frac": np.nan,
        "direction_n_variants_evaluated": 0,
        "direction_n_variants_passing": 0,
        "n_independent_amplicons": 0,
        "block_support_mode": "cell_extreme_fraction",
        "block_min_cell_extreme_support": np.nan,
        "block_median_cell_extreme_support": np.nan,
    }
    if not cells or not arm_vars:
        out["reason"] = "no_cells_or_arm_variants"
        return out

    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))
    min_block = int(getattr(cfg, "min_contiguous_variants_initial", 4))
    min_child = _min_child_size(cfg)
    min_extreme = float(getattr(cfg, "min_extreme_fraction", 0.90))
    min_corr = float(getattr(cfg, "phase3_min_mean_corr_to_mean", 0.80))
    min_good_frac = float(getattr(cfg, "min_homogeneity_fraction", 0.80))
    min_corr_vars = int(getattr(cfg, "phase3_min_variants_for_cell_corr", 4))
    edge_tol = int(getattr(cfg, "block_edge_tolerance", 0))
    min_amplicons = int(getattr(cfg, "min_different_amplicons", 3))
    v2a = _variant_to_amplicon(state)
    info_set = getattr(cfg, "informative_variants", None)
    info_set = set(map(str, info_set)) if info_set is not None else None

    if len(cells) < min_child:
        out["reason"] = f"child_too_small:{len(cells)}<{min_child}"
        return out

    X = _to_float_array(state.df_numeric.loc[cells, arm_vars])
    finite_full = np.isfinite(X)
    extreme_full = ((X <= low) | (X >= high)) & finite_full
    n_finite_per_variant = finite_full.sum(axis=0)
    support_per_variant = np.divide(
        extreme_full.sum(axis=0),
        n_finite_per_variant,
        out=np.zeros(X.shape[1], dtype=float),
        where=n_finite_per_variant > 0,
    )
    support_mask = support_per_variant >= min_extreme

    best_core: list[int] = []
    best_span_len = -1
    for run in edge_tolerant_extreme_runs(support_mask, edge_tol):
        core = list(run["core_indices"])
        span = list(run["span_indices"])
        if len(span) >= min_block and (
            len(core) > len(best_core)
            or (len(core) == len(best_core) and len(span) > best_span_len)
        ):
            best_core = core
            best_span_len = len(span)

    if len(best_core) < min_block:
        out.update({
            "reason": f"block_too_short:{len(best_core)}<{min_block}",
            "n_block_variants": len(best_core),
        })
        return out

    block_vars = [arm_vars[i] for i in best_core]
    block_values = X[:, best_core]
    finite = np.isfinite(block_values)
    n_finite = int(finite.sum())
    extreme_frac = (
        float((((block_values <= low) | (block_values >= high)) & finite).sum() / n_finite)
        if n_finite else 0.0
    )

    # Correlation is evaluated within the proposed signed child. Thus reciprocal
    # groups are separated before this metric is applied; no biologically real
    # low/high mixture is averaged into one profile.
    block_mu = np.nanmean(block_values, axis=0)
    cell_rs: list[float] = []
    for row in block_values:
        mask = np.isfinite(row) & np.isfinite(block_mu)
        if int(mask.sum()) < min_corr_vars:
            continue
        x = row[mask]
        y = block_mu[mask]
        if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
            r = 1.0 if float(np.nanmedian(np.abs(x - y))) <= 10.0 else 0.0
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r):
            cell_rs.append(r)

    mean_r = float(np.mean(cell_rs)) if cell_rs else np.nan
    good_frac = float(np.mean(np.asarray(cell_rs) >= min_corr)) if cell_rs else 0.0

    dir_min_extreme = int(
        getattr(cfg, "directional_min_extreme_cells_per_variant", 5)
    )
    dir_min_extreme = max(2, min(dir_min_extreme, len(cells)))
    direction = check_directional_consistency(
        block_values,
        low_thr=low,
        high_thr=high,
        min_per_var_agreement=float(
            getattr(cfg, "directional_min_per_variant_agreement", 0.95)
        ),
        min_block_pass_frac=float(
            getattr(cfg, "directional_min_block_pass_frac", 1.00)
        ),
        min_extreme_cells_per_var=dir_min_extreme,
    )

    block_amps = {
        str(v2a.get(v)) for v in block_vars
        if (info_set is None or v in info_set)
        and v2a.get(v) not in _MISSING_AMPLICON_VALUES
    }
    n_amps = len(block_amps)

    reasons: list[str] = []
    if extreme_frac < min_extreme:
        reasons.append(
            f"block_extreme_fraction:{extreme_frac:.3f}<{min_extreme:.3f}"
        )
    if not cell_rs:
        reasons.append("insufficient_cell_correlation_evidence")
    elif good_frac < min_good_frac:
        reasons.append(f"cell_homogeneity:{good_frac:.3f}<{min_good_frac:.3f}")
    if bool(getattr(cfg, "directional_check_enabled", True)) and not bool(
        direction.get("passes", False)
    ):
        reasons.append(
            "directional_consistency_fail:"
            f"{int(direction.get('n_variants_passing', 0))}/"
            f"{int(direction.get('n_variants_evaluated', 0))}"
        )
    if n_amps < min_amplicons:
        reasons.append(f"independent_amplicons:{n_amps}<{min_amplicons}")

    out.update({
        "strict_pass": len(reasons) == 0,
        "reason": "pass" if not reasons else ";".join(reasons),
        "n_block_variants": len(block_vars),
        "block_variants": ";".join(block_vars),
        "block_extreme_fraction": extreme_frac,
        "mean_cell_to_mean_r": mean_r,
        "good_cell_fraction": good_frac,
        "direction": direction.get("dominant_direction", "NA"),
        "direction_block_pass_frac": direction.get("block_pass_frac", np.nan),
        "direction_n_variants_evaluated": int(
            direction.get("n_variants_evaluated", 0)
        ),
        "direction_n_variants_passing": int(
            direction.get("n_variants_passing", 0)
        ),
        "n_independent_amplicons": n_amps,
        "block_support_mode": "cell_extreme_fraction",
        "block_min_cell_extreme_support": float(
            np.nanmin(support_per_variant[best_core])
        ) if best_core else np.nan,
        "block_median_cell_extreme_support": float(
            np.nanmedian(support_per_variant[best_core])
        ) if best_core else np.nan,
    })
    return out


def _strict_validate_new_arm_children(state, parent_cells, chr_arm: str) -> pd.DataFrame:
    """Validate regular-detector children and roll back only failed arm tokens."""
    state.refresh()
    labels = state.current_clones.copy()

    candidate_labels: list[str] = []
    for cell_id in parent_cells:
        sample_idx = _sample_idx_for_cell(state, cell_id)
        if sample_idx is None:
            continue
        label = str(labels.iloc[sample_idx])
        if is_loh_label(label) and chr_arm in label:
            candidate_labels.append(label)
    candidate_labels = sorted(set(candidate_labels))

    rows: list[dict] = []
    failed_labels: set[str] = set()
    for label in candidate_labels:
        cells = _cells_with_label(state, parent_cells, labels, label)
        metrics = _strict_arm_metrics_for_cells(state, cells, chr_arm)
        metrics["child_label"] = label
        rows.append(metrics)
        if not bool(metrics.get("strict_pass", False)):
            failed_labels.add(label)

    # Roll back only this failed arm token. Earlier accepted Phase-6 arms stay.
    if failed_labels:
        current = state.current_clones.copy()
        changed = False
        for failed in failed_labels:
            mask = current.astype(str).eq(str(failed))
            if bool(mask.any()):
                current.loc[mask] = _remove_arm_token(str(failed), chr_arm)
                changed = True
        if changed:
            state.set_clone_labels(current)

    return pd.DataFrame(rows)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end intervals for contiguous True values.

    Identical to ``_contiguous_runs_from_mask``; kept as a separate name for
    existing callers.
    """
    return _contiguous_runs_from_mask(mask)


def _next_phase6_child_number(state, parent_label: str) -> int:
    """Return a non-conflicting LOH-C integer for a Phase-6 child.

    One more than the largest ``LOH-C<n>`` found in any label starting with
    ``parent_label``; 0 if none exist.
    """
    labels = _clone_label_series(state)
    prefix = str(parent_label).strip()
    numbers: list[int] = []
    for label in labels:
        if str(label).startswith(prefix):
            numbers.extend(int(x) for x in re.findall(r"\bLOH-C(\d+)\b", str(label)))
    return (max(numbers) + 1) if numbers else 0


def _signed_block_partition_children(state, parent_cells, parent_label: str, chr_arm: str) -> pd.DataFrame:
    """Split a directionally mixed detector call before strict validation.

    A single greedy detector cluster may contain cells with *reciprocal signed
    allele patterns* on the same LOH block. Such a mixture must not be accepted
    as one clone because each variant has no dominant LOW/HIGH direction. This
    fallback partitions it using the signed contiguous AF signature of each seed
    cell, then validates each proposed child with the unchanged strict Phase-6
    gates. It is not a threshold relaxation.
    """
    cfg = state.config
    state.refresh()
    parent_cells = [c for c in parent_cells if c in state.df_numeric.index]
    arm_vars = sorted(
        [
            str(v) for v in state.grouped_by_chr_arm.get(chr_arm, [])
            if v in state.df_numeric.columns
        ],
        key=_variant_position,
    )
    columns = [
        "child_label", "chr_arm", "n_cells", "strict_pass", "reason",
        "n_block_variants", "block_variants", "block_extreme_fraction",
        "mean_cell_to_mean_r", "good_cell_fraction", "direction",
        "direction_block_pass_frac", "direction_n_variants_evaluated",
        "direction_n_variants_passing", "n_independent_amplicons",
        "block_support_mode", "block_min_cell_extreme_support",
        "block_median_cell_extreme_support", "partition_method", "seed_cell",
        "seed_start_variant", "seed_end_variant", "seed_match_requirement",
        "barcodes",
    ]
    if not parent_cells or not arm_vars:
        return pd.DataFrame(columns=columns)

    low = float(getattr(cfg, "low_allele_threshold", 10.0))
    high = float(getattr(cfg, "high_allele_threshold", 90.0))
    min_block = int(getattr(cfg, "min_contiguous_variants_initial", 4))
    min_child = _min_child_size(cfg)
    min_shared = int(getattr(cfg, "min_shared_variants", 4))

    X = _to_float_array(state.df_numeric.loc[parent_cells, arm_vars])
    signed = np.where(X <= low, -1, np.where(X >= high, 1, 0)).astype(int)

    # Each seed defines an allelic LOW/HIGH signature across a contiguous block.
    # Candidates must reproduce the same signs at sufficient positions. The
    # strict validator then independently enforces block, direction, correlation,
    # and amplicon criteria.
    candidate_meta: dict[tuple, dict] = {}
    for seed_idx, seed_cell in enumerate(parent_cells):
        for start, end in _contiguous_true_runs(signed[seed_idx] != 0):
            run_len = end - start + 1
            if run_len < min_block:
                continue
            seed_sign = signed[seed_idx, start:end + 1]
            support = seed_sign != 0
            n_support = int(support.sum())
            if n_support < min_block:
                continue

            requirements = sorted({
                min_shared,
                int(np.ceil(0.80 * n_support)),
                int(np.ceil(0.90 * n_support)),
                n_support,
            })
            matches = (
                (signed[:, start:end + 1] == seed_sign[None, :])
                & support[None, :]
            ).sum(axis=1)
            for required in requirements:
                required = max(min_shared, min(int(required), n_support))
                member_idx = np.where(matches >= required)[0]
                if len(member_idx) < min_child:
                    continue
                group_cells = tuple(sorted(parent_cells[j] for j in member_idx))
                candidate = {
                    "seed_cell": seed_cell,
                    "seed_start": start,
                    "seed_end": end,
                    "seed_requirement": int(required),
                    "seed_run_length": int(run_len),
                }
                old = candidate_meta.get(group_cells)
                if old is None or (
                    candidate["seed_requirement"], candidate["seed_run_length"]
                ) > (
                    old["seed_requirement"], old["seed_run_length"]
                ):
                    candidate_meta[group_cells] = candidate

    if not candidate_meta:
        return pd.DataFrame(columns=columns)

    passing: list[dict] = []
    failing: list[dict] = []
    for group_cells, meta in candidate_meta.items():
        metrics = _strict_arm_metrics_for_cells(state, list(group_cells), chr_arm)
        rec = dict(metrics)
        rec.update({
            "partition_method": "signed_contiguous_block_partition",
            "seed_cell": str(meta["seed_cell"]),
            "seed_start_variant": arm_vars[meta["seed_start"]],
            "seed_end_variant": arm_vars[meta["seed_end"]],
            "seed_match_requirement": int(meta["seed_requirement"]),
            "barcodes": ";".join(
                str(state.cell_to_barcode.get(c, c)) for c in group_cells
            ),
            "_cells": tuple(group_cells),
        })
        (passing if bool(metrics.get("strict_pass", False)) else failing).append(rec)

    # Select non-overlapping groups. Prefer large, long, tightly signed children
    # so a diffuse superset cannot swallow mutually exclusive descendants.
    passing = sorted(
        passing,
        key=lambda r: (
            -int(r.get("n_cells", 0)),
            -int(r.get("n_block_variants", 0)),
            -int(r.get("seed_match_requirement", 0)),
            tuple(map(str, r.get("_cells", ()))),
        ),
    )
    selected: list[dict] = []
    used: set = set()
    for rec in passing:
        cells = set(rec["_cells"])
        if cells & used:
            continue
        selected.append(rec)
        used |= cells

    if selected:
        labels = state.current_clones.copy()
        next_id = _next_phase6_child_number(state, parent_label)
        for rec in selected:
            child_id = next_id
            next_id += 1
            token = f" LOH-C{child_id} {chr_arm}"
            child_labels: set[str] = set()
            for cell_id in rec["_cells"]:
                sample_idx = _sample_idx_for_cell(state, cell_id)
                if sample_idx is None:
                    continue
                current = str(labels.iloc[sample_idx])
                # Only members of the residual lineage can accumulate a Phase-6
                # arm. This permits an accepted same-pass child to add another
                # arm while preventing cross-lineage reassignment.
                if not (
                    current == str(parent_label)
                    or current.startswith(f"{parent_label} LOH-C")
                ):
                    continue
                if chr_arm not in current:
                    labels.iloc[sample_idx] = current + token
                child_labels.add(str(labels.iloc[sample_idx]))
            rec["child_label"] = "; ".join(sorted(child_labels))
        state.set_clone_labels(labels)
        state.refresh()

    output_rows: list[dict] = []
    for rec in selected + failing:
        rec = dict(rec)
        rec.pop("_cells", None)
        rec.setdefault("child_label", "")
        output_rows.append(rec)
    return pd.DataFrame(output_rows, columns=columns)


def _skipped_parent_audit_row(parent_label: str, n_parent: int, reason: str) -> dict:
    """Audit row for a parent that is not scanned at all."""
    return {
        "parent_clone": parent_label,
        "scan_pass": pd.NA,
        "chr_arm": pd.NA,
        "n_parent_cells": n_parent,
        "n_residual_cells_before": n_parent,
        "n_residual_cells_after": n_parent,
        "status": "SKIPPED",
        "reason": reason,
        "n_detected_cells": 0,
        "n_relabelled_cells": 0,
        "n_child_labels": 0,
        "child_labels": "",
    }


def _has_strict_pass_column(validation: pd.DataFrame) -> bool:
    """True when a validation frame is non-empty and has a ``strict_pass`` column."""
    return not validation.empty and "strict_pass" in validation.columns


def _strict_passing_child_labels(validation: pd.DataFrame) -> list[str]:
    """``child_label`` values of rows that passed strict validation."""
    if not _has_strict_pass_column(validation):
        return []
    return validation.loc[
        validation["strict_pass"].fillna(False),
        "child_label",
    ].astype(str).tolist()


def _n_strict_passing(validation: pd.DataFrame) -> int:
    """Number of strict-passing rows (0 for an empty / column-less frame)."""
    return int(validation["strict_pass"].sum()) if _has_strict_pass_column(validation) else 0


def _strict_fail_reasons(validation: pd.DataFrame) -> str:
    """'; '-joined ``reason`` of rows that failed strict validation."""
    if not _has_strict_pass_column(validation):
        return ""
    return "; ".join(
        validation.loc[~validation["strict_pass"].fillna(False), "reason"].astype(str).tolist()
    )


def _freeze_residual_reference(
    frozen_reference_by_barcode: dict,
    sample_barcodes: pd.Series,
    labels: pd.Series,
    residual_barcodes: set,
) -> None:
    """Copy the current labels of residual-parent barcodes into the frozen reference."""
    for barcode, label in zip(sample_barcodes, labels):
        if str(barcode) in residual_barcodes:
            frozen_reference_by_barcode[str(barcode)] = str(label)


def run_post_phase5_subclone_scan(
    *,
    state,
    clone_labels: Iterable[str] | str,
    chr_arms: Iterable[str] | None = None,
    min_parent_cells: int | None = None,
    rerun_phase5_summary: bool = True,
    merge_back_after_scan: bool = True,
    auto_exclude_labels: Iterable[str] | None = None,
    auto_include_wt: bool = False,
    verbose: bool = True,
) -> dict:
    """Iteratively discover LOH children inside selected final parent clones.

    This is a *residual-parent* scan.  Each pass scans only cells that still
    carry the exact requested base label.  Once an LOH child is accepted, its
    cells are removed from the parent pool and the remaining parent cells are
    rescanned from scratch.  This is essential for heterogeneous final clones:
    a large first child can otherwise mask a second coherent LOH child in the
    pooled parent profile.

    Arm calls within a single pass still use a frozen parent reference so a
    genuine multi-arm child can accumulate its arm tokens.  The next pass then
    works only on the unlabelled residual parent cells.  Existing Phase-5
    LOH/SUB/FP/small labels remain ineligible and are never reopened.

    ``clone_labels="auto"`` selects every eligible non-LOH parent with
    ``>= min_parent_cells`` cells that still carries a possible LOH block on some
    scan arm (see ``_auto_select_scan_parents``), instead of requiring an
    explicit list.

    ``merge_back_after_scan`` (default True) runs strict Phase-4-style
    consolidation once splitting is done. During this scoped merge-back, a
    smaller same-base child may rejoin a larger sibling with the *same* LOH arm
    set, provided all normal per-cell profile, AF-delta, extreme-fraction,
    block-overlap, and direction gates pass.

    In automatic modes the exact ``WT`` parent is excluded by default. Set
    ``auto_include_wt=True`` only for a deliberate WT-focused scan.

    Parameters
    ----------
    state : PipelineState
        Final state after Phase 5; ``Sample.dna.row_attrs['Clone']`` and the
        ``phase6_*`` row attributes are modified in place.
    clone_labels : str or iterable of str
        Explicit parent labels, or ``"auto"`` / ``"all"`` (case-insensitive).
        ``"all"`` scans every eligible parent at/above the size floor without
        the possible-LOH-block pre-screen.
    chr_arms : iterable of str, optional
        Arms to scan; defaults to ``state.chr_arms``. Unknown arms are skipped
        (with a warning when ``verbose``).
    min_parent_cells : int, optional
        Parent/residual size floor; defaults to
        ``max(2 * min_cluster_size, min_final_clone_size)``.
    rerun_phase5_summary : bool, default True
        Run ``run_phase5`` afterwards (as "PHASE 7", with the LOH prune floor
        relaxed to the Phase-6 child size).
    merge_back_after_scan : bool, default True
        Run the scoped Phase-4 small-LOH merge with equal arm sets allowed.
    auto_exclude_labels : iterable of str, optional
        Labels never auto-selected.
    auto_include_wt : bool, default False
        Allow the exact ``WT`` parent in auto modes.
    verbose : bool, default True
        Print per-parent / per-arm progress. Banner lines are always printed.

    Returns
    -------
    dict
        ``audit_df`` (one row per parent/pass/arm), ``parent_results``,
        ``summary_df``, ``merge_back_df``, ``clone_labels``, ``chr_arms``,
        ``min_parent_cells``, ``auto_include_wt`` and ``scan_mode``; also stored
        in ``state.results["phase6_post_phase5_subclone_scan"]``. If auto mode
        finds no parents, a smaller dict is returned early (not stored).

    Raises
    ------
    ValueError
        No scannable arms, ``min_parent_cells < 1``, or unusable explicit
        ``clone_labels``.
    """
    cfg = state.config
    state.refresh()

    if chr_arms is None:
        scan_arms = [str(a) for a in state.chr_arms]
    else:
        requested_arms = [str(a) for a in chr_arms]
        scan_arms = [a for a in requested_arms if a in set(state.chr_arms)]
        missing_arms = sorted(set(requested_arms) - set(scan_arms))
        if missing_arms and verbose:
            print(
                "  ⚠ Post-Phase-5 scan skipped unavailable arm(s): "
                + ", ".join(missing_arms)
            )

    if not scan_arms:
        raise ValueError("No eligible chromosome arms are available for the scan.")

    if min_parent_cells is None:
        min_parent_cells = max(
            2 * int(getattr(cfg, "min_cluster_size", 5)),
            int(getattr(cfg, "min_final_clone_size", 5)),
        )
    min_parent_cells = int(min_parent_cells)
    if min_parent_cells < 1:
        raise ValueError("min_parent_cells must be >= 1.")

    # Resolve the parent set.
    #   "auto" → eligible parents that still show a possible LOH block.
    #   "all"  → every eligible parent with >= min_parent_cells (no pre-screen).
    sentinel = (
        str(clone_labels).strip().lower()
        if isinstance(clone_labels, str)
        else None
    )
    auto_mode = sentinel in {"auto", "all"}
    if auto_mode:
        requested = _auto_select_scan_parents(
            state,
            cfg,
            scan_arms,
            min_parent_cells=min_parent_cells,
            exclude_labels=auto_exclude_labels,
            include_wt=auto_include_wt,
            require_possible_block=(sentinel == "auto"),
            verbose=verbose,
        )
        if not requested:
            if verbose:
                print("  Auto scan: no eligible parent clones found.")
            return {
                "audit_df": pd.DataFrame(),
                "parent_results": {},
                "summary_df": run_phase5(state) if rerun_phase5_summary else None,
                "merge_back_df": None,
                "clone_labels": [],
                "chr_arms": scan_arms,
                "min_parent_cells": min_parent_cells,
                "scan_mode": f"iterative_residual_parent_{sentinel}",
            }
    else:
        requested = _normalise_clone_labels(clone_labels)

    min_child = _min_child_size(cfg)
    # A finite cap protects against unexpected label cycles.  In ordinary use
    # each successful pass removes >=min_child cells from the residual parent,
    # so this cannot be reached for realistic clone sizes.
    max_passes = max(1, int(np.ceil(max(len(state.df), 1) / max(min_child, 1))))

    sample_barcodes = pd.Series(state.Sample.dna.row_attrs["barcode"]).astype(str)

    def _residual_parent_cells(parent_label: str) -> list:
        """Cells still carrying the exact non-LOH parent label."""
        state.refresh()
        return [
            cell_id
            for cell_id in state.df.index
            if str(state.barcode_to_clone.get(
                str(state.cell_to_barcode.get(cell_id, cell_id)),
                "",
            )) == str(parent_label)
        ]

    print("\n" + "=" * 60)
    print("PHASE 6: ITERATIVE POST-PHASE-5 RESIDUAL SUBCLONE SCAN")
    print("=" * 60)
    print("  Parent labels are frozen at the start of each residual pass.")
    print("  Each accepted child is removed before the next pass.")
    print("  Existing LOH/SUB/FP/small labels are not rescanned.")
    print(
        f"  Parent size gate: n≥{min_parent_cells}; "
        f"child size gate: n≥{min_child} (same final-size floor as primary LOH calls)."
    )
    if auto_mode and not auto_include_wt:
        print("  Auto mode excludes the exact WT parent by default.")
    print("  Broad Phase-4 merging is deliberately not rerun; primary-child similarity expansion and nested-arm preservation are applied before strict consolidation.")

    audit_rows: list[dict] = []
    parent_results: dict[str, dict] = {}

    for parent_label in requested:
        parent_ok, parent_reason = _eligible_parent_label(parent_label)
        initial_parent_cells = _residual_parent_cells(parent_label)
        n_initial_parent = len(initial_parent_cells)

        if not parent_ok:
            audit_rows.append(
                _skipped_parent_audit_row(parent_label, n_initial_parent, parent_reason)
            )
            if verbose:
                print(f"\n  {parent_label}: SKIP ({parent_reason})")
            continue

        if n_initial_parent < min_parent_cells:
            audit_rows.append(
                _skipped_parent_audit_row(
                    parent_label,
                    n_initial_parent,
                    f"parent_too_small:{n_initial_parent}<{min_parent_cells}",
                )
            )
            if verbose:
                print(
                    f"\n  {parent_label}: SKIP "
                    f"(n={n_initial_parent} < {min_parent_cells})"
                )
            continue

        if verbose:
            print(f"\n── Parent: {parent_label}  (initial n={n_initial_parent}) ──")

        parent_results[parent_label] = {}
        seen_residual_states: set[tuple[str, ...]] = set()

        for scan_pass in range(1, max_passes + 1):
            parent_cells = _residual_parent_cells(parent_label)
            n_residual_before = len(parent_cells)
            residual_key = tuple(sorted(map(str, parent_cells)))

            if n_residual_before < min_parent_cells:
                if verbose:
                    print(
                        f"  residual pass {scan_pass}: stop "
                        f"(n={n_residual_before} < {min_parent_cells})"
                    )
                break
            if residual_key in seen_residual_states:
                if verbose:
                    print(
                        f"  residual pass {scan_pass}: stop (repeated residual state)"
                    )
                break
            seen_residual_states.add(residual_key)

            if verbose:
                print(
                    f"  residual pass {scan_pass}: scanning "
                    f"{n_residual_before} exact '{parent_label}' cell(s)"
                )

            # The detector must refer to this *pass-local* parent only.  Using
            # the original n cells in every pass is what allowed a large child
            # to mask a second child in the previous implementation.
            residual_barcodes = {
                str(state.cell_to_barcode.get(cell_id, cell_id))
                for cell_id in parent_cells
            }
            current_labels = _clone_label_series(state)
            frozen_reference_by_barcode = dict(zip(sample_barcodes, current_labels))
            for barcode in residual_barcodes:
                frozen_reference_by_barcode[barcode] = str(parent_label)

            parent_df = state.df.loc[parent_cells].copy()
            pass_changed_cells: set[str] = set()
            pass_arm_results: dict[str, tuple] = {}

            for chr_arm in scan_arms:
                arm_vars = [
                    v for v in state.grouped_by_chr_arm.get(chr_arm, [])
                    if v in parent_df.columns
                ]
                if len(arm_vars) < int(getattr(cfg, "min_contiguous_variants_initial", 4)):
                    audit_rows.append({
                        "parent_clone": parent_label,
                        "scan_pass": scan_pass,
                        "chr_arm": chr_arm,
                        "n_parent_cells": n_initial_parent,
                        "n_residual_cells_before": n_residual_before,
                        "status": "SKIPPED",
                        "reason": "insufficient_informative_variants",
                        "n_detected_cells": 0,
                        "n_relabelled_cells": 0,
                        "n_child_labels": 0,
                        "child_labels": "",
                    })
                    continue

                labels_before = _clone_label_series(state).copy()

                segments, loh_cells, correlations, homogeneity = (
                    detect_loh_and_update_clones(
                        Sample=state.Sample,
                        df=parent_df,
                        grouped_by_chr_arm=state.grouped_by_chr_arm,
                        target_chr=chr_arm,
                        low_allele_threshold=float(getattr(cfg, "low_allele_threshold", 10.0)),
                        high_allele_threshold=float(getattr(cfg, "high_allele_threshold", 90.0)),
                        min_segment_length=int(getattr(cfg, "min_contiguous_variants_initial", 4)),
                        min_extreme_fraction=float(getattr(cfg, "min_extreme_fraction", 0.90)),
                        min_shared_variants=int(getattr(cfg, "min_shared_variants", 4)),
                        min_cluster_size=int(
                            getattr(
                                cfg,
                                "phase6_residual_min_cluster_size",
                                getattr(cfg, "min_cluster_size", 5),
                            )
                        ),
                        min_final_clone_size=min_child,
                        correlation_threshold=float(getattr(cfg, "correlation_threshold", 0.80)),
                        restore_from_backup=False,
                        skip_homogeneity_check=True,
                        min_homogeneity_fraction=float(getattr(cfg, "min_homogeneity_fraction", 0.80)),
                        skip_size_filtering=False,
                        cytobands_df=getattr(state, "cytobands_df", None),
                        keep_incoherent_clusters=False,
                        verbose=bool(getattr(cfg, "detect_loh_verbose", False)) and verbose,
                        clone_reference_labels=frozen_reference_by_barcode,
                    )
                )

                state.refresh()
                strict_validation = _strict_validate_new_arm_children(
                    state,
                    parent_cells,
                    chr_arm,
                )
                state.refresh()
                labels_after = _clone_label_series(state)
                changed = labels_after.ne(labels_before)
                changed_barcodes = set(sample_barcodes.loc[changed].astype(str))

                # A regular detector cluster can combine reciprocal LOW/HIGH
                # signatures, then fail directional consistency as one pooled
                # child. Split that candidate by its signed contiguous pattern
                # before accepting or rejecting the arm.
                signed_partition_validation = pd.DataFrame()
                if not bool(changed.any()):
                    signed_partition_validation = _signed_block_partition_children(
                        state,
                        parent_cells,
                        parent_label,
                        chr_arm,
                    )
                    state.refresh()
                    labels_after = _clone_label_series(state)
                    changed = labels_after.ne(labels_before)
                    changed_barcodes = set(sample_barcodes.loc[changed].astype(str))
                    if bool(changed.any()):
                        # Allow one accepted signed child to accumulate an
                        # additional arm later in this same residual pass.
                        _freeze_residual_reference(
                            frozen_reference_by_barcode,
                            sample_barcodes,
                            labels_after,
                            residual_barcodes,
                        )

                # Core / signed children are strict primary discoveries.  Expand
                # a primary LOH family only with residual-parent cells that are
                # strongly concordant with its block representative; this keeps
                # real, slightly imperfect members without accepting unrelated
                # extreme-AF cells.
                primary_expansion = pd.DataFrame()
                labels_after = _clone_label_series(state)
                newly_created_primary_labels = []
                for child_label in sorted(set(labels_after.loc[labels_after.ne(labels_before)].astype(str))):
                    primary_label_without_arm = _remove_arm_token(child_label, chr_arm)
                    if (
                        is_loh_label(child_label)
                        and chr_arm in child_label
                        and not is_loh_label(primary_label_without_arm)
                    ):
                        newly_created_primary_labels.append(child_label)

                for child_label in newly_created_primary_labels:
                    core_cells = _cells_with_label(state, parent_cells, labels_after, child_label)
                    if core_cells:
                        _set_phase6_primary_metadata(
                            state,
                            core_cells,
                            child_label=child_label,
                            chr_arm=chr_arm,
                            member_role="strict_core",
                        )
                        expansion_df = _expand_primary_child_with_similar_members(
                            state,
                            parent_label=parent_label,
                            parent_cells=parent_cells,
                            child_label=child_label,
                            chr_arm=chr_arm,
                        )
                        partial_expansion_df = _expand_primary_child_with_partial_block_members(
                            state,
                            parent_label=parent_label,
                            parent_cells=parent_cells,
                            child_label=child_label,
                            chr_arm=chr_arm,
                        )
                        expansion_frames = [
                            x for x in (expansion_df, partial_expansion_df)
                            if isinstance(x, pd.DataFrame) and not x.empty
                        ]
                        if expansion_frames:
                            primary_expansion = pd.concat(
                                [primary_expansion, *expansion_frames],
                                ignore_index=True,
                                sort=False,
                            )

                state.refresh()
                labels_after = _clone_label_series(state)

                # Do not fragment an already established primary child when a
                # different arm is present only in a subset of that family.
                # Keep the primary Clone label and retain the new arm as nested
                # row-level evidence, so phase-5 pruning cannot discard the
                # remainder of the same primary LOH block.
                passing_labels = set()
                passing_labels.update(_strict_passing_child_labels(strict_validation))
                passing_labels.update(_strict_passing_child_labels(signed_partition_validation))
                secondary_annotation = _collapse_secondary_arm_subset_to_primary_family(
                    state,
                    parent_cells=parent_cells,
                    labels_before=labels_before,
                    chr_arm=chr_arm,
                    passing_child_labels=passing_labels,
                )
                state.refresh()
                labels_after = _clone_label_series(state)
                changed = labels_after.ne(labels_before)
                changed_barcodes = set(sample_barcodes.loc[changed].astype(str))

                # Freeze the accepted primary family for the rest of this pass.
                # Secondary nested evidence remains in metadata and therefore
                # cannot produce a flat-label sub-split.
                if bool(changed.any()):
                    _freeze_residual_reference(
                        frozen_reference_by_barcode,
                        sample_barcodes,
                        labels_after,
                        residual_barcodes,
                    )

                parent_child_labels = []
                for cell_id in parent_cells:
                    sample_idx = _sample_idx_for_cell(state, cell_id)
                    if sample_idx is None:
                        continue
                    child_label = str(labels_after.iloc[sample_idx])
                    if is_loh_label(child_label) and chr_arm in child_label:
                        parent_child_labels.append(child_label)
                parent_child_labels = sorted(set(parent_child_labels))

                n_detected = int(len(loh_cells)) if loh_cells is not None else 0
                n_changed = int(sum(
                    str(state.cell_to_barcode.get(cell_id, cell_id)) in changed_barcodes
                    for cell_id in parent_cells
                ))
                pass_changed_cells.update(changed_barcodes)

                n_strict_passing_children = _n_strict_passing(strict_validation)
                strict_fail_reasons = _strict_fail_reasons(strict_validation)
                n_signed_partition_children = _n_strict_passing(signed_partition_validation)
                signed_partition_fail_reasons = _strict_fail_reasons(signed_partition_validation)
                n_nested_secondary_evidence_cells = int(
                    secondary_annotation.get("n_secondary_cells", pd.Series(dtype=int)).sum()
                ) if isinstance(secondary_annotation, pd.DataFrame) and not secondary_annotation.empty else 0
                n_primary_similarity_expanded = int(
                    primary_expansion.get("member_action", pd.Series(dtype=object)).isin(
                        ["expanded_similarity", "expanded_partial_block"]
                    ).sum()
                ) if isinstance(primary_expansion, pd.DataFrame) and not primary_expansion.empty else 0
                if n_changed > 0 and n_signed_partition_children > 0:
                    status = "LOH_CHILD_FOUND_SIGNED_PARTITION"
                    reason = "signed_partition_and_strict_block_qc_pass"
                elif n_changed > 0:
                    status = "LOH_CHILD_FOUND"
                    reason = "strict_detection_and_block_qc_pass"
                elif n_nested_secondary_evidence_cells > 0:
                    status = "LOH_SECONDARY_ARM_ANNOTATED"
                    reason = "nested_secondary_arm_recorded_without_flat_label_split"
                else:
                    status = "NO_CHILD"
                    reason = (
                        strict_fail_reasons
                        or signed_partition_fail_reasons
                        or "no_passing_child"
                    )

                audit_rows.append({
                    "parent_clone": parent_label,
                    "scan_pass": scan_pass,
                    "chr_arm": chr_arm,
                    "n_parent_cells": n_initial_parent,
                    "n_residual_cells_before": n_residual_before,
                    "status": status,
                    "reason": reason,
                    "n_detected_cells": n_detected,
                    "n_relabelled_cells": n_changed,
                    "n_child_labels": len(parent_child_labels),
                    "child_labels": "; ".join(parent_child_labels),
                    "n_segments": int(len(segments)) if segments is not None else 0,
                    "n_correlation_rows": int(len(correlations)) if correlations is not None else 0,
                    "n_homogeneity_rows": int(len(homogeneity)) if homogeneity is not None else 0,
                    "n_strict_passing_children": n_strict_passing_children,
                    "strict_validation": strict_validation.to_dict("records"),
                    "n_signed_partition_children": n_signed_partition_children,
                    "signed_partition_validation": signed_partition_validation.to_dict("records"),
                    "n_primary_similarity_expanded": n_primary_similarity_expanded,
                    "primary_similarity_expansion": primary_expansion.to_dict("records") if isinstance(primary_expansion, pd.DataFrame) else [],
                    "n_nested_secondary_evidence_cells": n_nested_secondary_evidence_cells,
                    "nested_secondary_arm_evidence": secondary_annotation.to_dict("records") if isinstance(secondary_annotation, pd.DataFrame) else [],
                })

                if n_changed > 0:
                    pass_arm_results[chr_arm] = (
                        segments,
                        loh_cells,
                        correlations,
                        homogeneity,
                    )

                if verbose:
                    detail = "" if (status.startswith("LOH_CHILD_FOUND") or status == "LOH_SECONDARY_ARM_ANNOTATED") else f"; reason={reason}"
                    print(
                        f"    {chr_arm}: {status}; detected={n_detected}; "
                        f"relabelled={n_changed}; children={len(parent_child_labels)}{detail}"
                    )

            state.refresh()
            residual_after_cells = _residual_parent_cells(parent_label)
            n_residual_after = len(residual_after_cells)
            for row in reversed(audit_rows):
                if (
                    row.get("parent_clone") == parent_label
                    and row.get("scan_pass") == scan_pass
                    and "n_residual_cells_after" not in row
                ):
                    row["n_residual_cells_after"] = n_residual_after

            parent_results[parent_label][f"pass_{scan_pass}"] = pass_arm_results
            n_removed = n_residual_before - n_residual_after
            if verbose:
                print(
                    f"  residual pass {scan_pass}: "
                    f"{n_removed} cell(s) assigned to child LOH labels; "
                    f"{n_residual_after} remain in '{parent_label}'"
                )

            if n_removed <= 0:
                break

    state.refresh()
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        sort_cols = [
            c for c in ["parent_clone", "scan_pass", "chr_arm", "status"]
            if c in audit_df.columns
        ]
        audit_df = audit_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # ── Strict Phase-6 consolidation ───────────────────────────────────
    # The ordinary Phase-4 cleanup permits LOH→LOH movement only when the
    # source arm set is a *strict* subset of a larger target. That leaves
    # split fragments with the exact same arm set stranded as separate clones.
    # During this scoped post-Phase-6 pass only, allow exact same-arm siblings
    # to rejoin a larger same-base target. All existing per-cell profile, AF
    # delta, extreme-fraction, block-overlap, and directional checks remain
    # unchanged; no broad non-LOH rescan is performed.
    merge_back_df = None
    if merge_back_after_scan:
        from .loh_config import arm_variants, extract_chr_arms
        from .phase4 import _merge_small_loh_clones_after_annotation

        scoped_attr = "phase4_merge_small_loh_allow_equal_arm_set_to_larger_target"
        old_scoped_value = getattr(cfg, scoped_attr, False)
        setattr(cfg, scoped_attr, True)
        try:
            print(
                "\n  Phase-6 consolidation: reabsorbing compatible same-arm "
                "split children into larger same-base LOH clones ..."
            )
            merge_back_df = _merge_small_loh_clones_after_annotation(
                state,
                cfg,
                LOH_TAG="LOH",
                arm_variants_fn=arm_variants,
                extract_chr_arms_fn=extract_chr_arms,
                low_af=float(getattr(cfg, "low_allele_threshold", 10.0)),
                high_af=float(getattr(cfg, "high_allele_threshold", 90.0)),
            )
        finally:
            setattr(cfg, scoped_attr, old_scoped_value)

        state.refresh()
        n_merged_back = (
            int(merge_back_df["merged"].fillna(False).astype(bool).sum())
            if isinstance(merge_back_df, pd.DataFrame)
            and "merged" in merge_back_df.columns
            else 0
        )
        if verbose:
            print(f"  Phase-6 consolidation reassigned {n_merged_back} cell(s).")

    # Phase numbering is intentionally sequential: Phase 6 owns residual
    # child discovery and scoped consolidation; the refreshed clone summary is
    # Phase 7, never a second "Phase 5" printed after Phase 6.
    print("\nPHASE 6 COMPLETE")
    print(f"  Parents scanned: {len(parent_results)}")
    print(f"  Newly labelled LOH-child cells: {int(audit_df.get('n_relabelled_cells', pd.Series(dtype=int)).fillna(0).sum()) if not audit_df.empty else 0}")

    summary_df = None
    if rerun_phase5_summary:
        print("\n  Proceeding to Phase 7 final clone summary ...")
        summary_df = run_phase5(
            state,
            loh_min_cells=min_child,
            heading="PHASE 7: FINAL LOH CLONE SUMMARY AFTER PHASE 6",
        )

    state.results["phase6_post_phase5_subclone_scan"] = {
        "audit_df": audit_df,
        "parent_results": parent_results,
        "summary_df": summary_df,
        "merge_back_df": merge_back_df,
        "clone_labels": requested,
        "chr_arms": scan_arms,
        "min_parent_cells": min_parent_cells,
        "auto_include_wt": bool(auto_include_wt),
        "scan_mode": "iterative_residual_parent",
    }

    return state.results["phase6_post_phase5_subclone_scan"]
