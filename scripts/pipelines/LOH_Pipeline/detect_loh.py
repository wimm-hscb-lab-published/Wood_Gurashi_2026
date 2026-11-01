"""
LOH Pipeline – core per-arm LOH detection.

Main entry point
----------------
``detect_loh_and_update_clones()`` is the per-arm LOH detection engine that
Phase 1 (and the Phase 6 residual subclone scan) call once per chromosome arm.
For one arm it:

1. finds per-cell contiguous runs of extreme allele frequencies (AF),
2. greedily groups cells that share direction-consistent extreme variants,
3. drops clusters that are incoherent across the full arm,
4. merges correlated LOH clusters and expands them into small, highly
   correlated non-LOH clones of the same base clone,
5. optionally applies the final clone-size gate,
6. appends ``" LOH-C<n> <arm>"`` annotations to ``Sample.dna.row_attrs['Clone']``,
7. optionally removes clones that fail a pairwise-correlation homogeneity check.

Cytobands are mapped and reported for information only; they never gate
segment discovery, so sparse amplicon panels (1–2 variants per band) do not
suppress true LOH blocks. Cytoband QC happens later, in Phase 1.

The function was previously a method inside ``CNV_Utils``; it lives here so it
can be imported, tested and versioned independently.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndimage_label
from scipy.stats import pearsonr

from .loh_config import build_cytoband_maps, is_loh_label

# Intra-cluster coherence (Step 3b): deliberately soft, because localised LOH
# clones may span cells from different base genotypes that only share the LOH
# region. The check only catches completely random clusters.
_COHERENCE_R_THRESHOLD = 0.30
_COHERENCE_MIN_FRAC = 0.40
# Minimum shared finite variants for a coherence-check pair.
_COHERENCE_MIN_VALID_VARIANTS = 3
# A profile with SD below this is treated as constant (Pearson undefined).
_CONSTANT_PROFILE_STD_EPS = 1e-6
# Two constant profiles "agree" when their mean |ΔAF| is below this (AF %).
_CONSTANT_PROFILE_MAX_MEAN_ABS_DIFF = 15.0
# Minimum shared finite variants for clone-merge, expansion and
# homogeneity correlations.
_MIN_VALID_VARIANTS_FOR_CORR = 10
# Print a homogeneity progress line every N clones.
_PROGRESS_EVERY = 5


def _empty_result() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the four empty DataFrames used for every early exit."""
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _make_vprint(verbose: bool) -> Callable[[str], None]:
    """Return a print function that only prints (and flushes) when *verbose*."""

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    return vprint


def _extract_pos(variant_name: str) -> int:
    """Genomic position from ``chrN:POS[-END]:REF/ALT``; 0 if unparsable."""
    try:
        return int(variant_name.split(":")[1].split("-")[0])
    except Exception:
        return 0


def _get_base_clone(clone_label):
    """Strip every ``' LOH-C<n> <arm>'`` suffix from a (string) clone label."""
    return re.sub(r" LOH-C\d+ \S+", "", clone_label)


def _loh_clone_label(base_clone, cluster_id, target_chr: str) -> str:
    """Build the per-arm LOH clone label ``'<base> LOH-C<id> <arm>'``."""
    return f"{base_clone} LOH-C{cluster_id} {target_chr}"


def _find_loh_segments_1d(
    row_binary: np.ndarray,
    row_values: np.ndarray,
    min_length: int,
    min_extreme_frac: float,
    low_allele_threshold: float,
    high_allele_threshold: float,
) -> List[Tuple[int, int]]:
    """Contiguous extreme-AF runs (start, end indices) for one cell on one arm.

    A run qualifies when it has at least *min_length* variants and its extreme
    fraction is at least *min_extreme_frac*. Cytobands are not consulted.
    """
    if row_binary.sum() < min_length:
        return []

    labeled, num_features = ndimage_label(row_binary)
    segments: List[Tuple[int, int]] = []

    for i in range(1, num_features + 1):
        indices = np.where(labeled == i)[0]
        if len(indices) < min_length:
            continue

        segment_values = row_values[indices]
        extreme_count = (
            (segment_values <= low_allele_threshold)
            | (segment_values >= high_allele_threshold)
        ).sum()
        if len(segment_values) == 0:
            continue
        if extreme_count / len(segment_values) < min_extreme_frac:
            continue

        segments.append((int(indices[0]), int(indices[-1])))

    return segments


def _resolve_reference_clones(
    Sample,
    sample_barcodes: pd.Series,
    restore_from_backup: Optional[bool],
    clone_reference_labels,
    vprint: Callable[[str], None],
) -> pd.Series:
    """Return the parent (base-clone) label per Sample row, positionally indexed.

    Without *clone_reference_labels*, ``Clone_backup`` is used (created from
    ``Clone`` if missing; optionally restored into ``Clone``). Otherwise the
    caller-supplied labels are aligned by barcode (dict / mismatched-length
    Series) or by position, and ``Clone_backup`` is left untouched.
    """
    if clone_reference_labels is None:
        # Legacy full-pipeline behaviour: use Clone_backup as the immutable
        # parent-label source across sequential arm calls.
        try:
            if "Clone_backup" in Sample.dna.row_attrs:
                if restore_from_backup is None:
                    pass
                elif restore_from_backup:
                    Sample.dna.row_attrs["Clone"] = Sample.dna.row_attrs[
                        "Clone_backup"
                    ].copy()
                    vprint("✓ Restored Clone labels from Clone_backup")
            else:
                original_clones = pd.Series(Sample.dna.row_attrs["Clone"]).copy()
                Sample.dna.row_attrs["Clone_backup"] = original_clones.values
                vprint("✓ Created backup: Clone_backup")

            return pd.Series(
                Sample.dna.row_attrs["Clone_backup"]
            ).astype(str).reset_index(drop=True)
        except Exception as e:
            vprint(f"❌ Error with backup: {repr(e)}")
            raise

    # Post-pipeline refinement mode: the caller supplies a frozen parent-label
    # vector for this discovery pass, so Clone_backup is not overwritten.
    if isinstance(clone_reference_labels, dict):
        reference_clones = sample_barcodes.map(
            lambda bc: str(clone_reference_labels.get(str(bc), ""))
        )
    elif isinstance(clone_reference_labels, pd.Series):
        ref = clone_reference_labels.copy()
        if len(ref) == len(sample_barcodes):
            reference_clones = ref.astype(str).reset_index(drop=True)
        else:
            ref.index = ref.index.astype(str)
            reference_clones = sample_barcodes.map(ref.astype(str)).fillna("")
    else:
        ref = pd.Series(clone_reference_labels)
        if len(ref) != len(sample_barcodes):
            raise ValueError(
                "clone_reference_labels must be barcode-keyed or have "
                "one entry per Sample.dna row."
            )
        reference_clones = ref.astype(str).reset_index(drop=True)

    if len(reference_clones) != len(sample_barcodes):
        raise ValueError(
            "clone_reference_labels could not be aligned to Sample.dna rows."
        )
    vprint("✓ Using caller-supplied frozen parent labels for this scan")
    return reference_clones


def _print_configuration(
    vprint: Callable[[str], None],
    low_allele_threshold: float,
    high_allele_threshold: float,
    min_segment_length: int,
    min_extreme_fraction: float,
    min_shared_variants: int,
    min_cluster_size: int,
    min_final_clone_size: int,
    correlation_threshold: float,
    skip_size_filtering: bool,
    skip_homogeneity_check: bool,
    min_homogeneity_fraction: float,
) -> None:
    """Print the run configuration block."""
    vprint(f"\n  Configuration:")
    vprint(
        f"    Low/High thresholds: ≤{low_allele_threshold}% / ≥{high_allele_threshold}%"
    )
    vprint(
        f"    Min segment: {min_segment_length} variants, {min_extreme_fraction:.0%} extreme"
    )
    vprint(
        f"    Min shared: {min_shared_variants} variants (direction-consistent; LOW/HIGH mix allowed)"
    )
    vprint(f"    Min cluster: {min_cluster_size} cells")
    vprint(f"    Min final clone: {min_final_clone_size} cells")
    vprint(f"    Correlation threshold: {correlation_threshold:.2f}")
    vprint(f"    Size filtering: {'DEFERRED' if skip_size_filtering else 'ENABLED'}")
    vprint(
        f"    Homogeneity check: "
        f"{'DISABLED' if skip_homogeneity_check else f'ENABLED (≥{min_homogeneity_fraction:.0%} cells must agree)'}"
    )


def _directional_variant_sets(
    cell_segments: dict,
    df: pd.DataFrame,
    X_low: np.ndarray,
    X_high: np.ndarray,
) -> Tuple[dict, dict]:
    """Per cell, the indices inside its LOH segments that are LOW and HIGH."""
    cell_low_variants: dict = {}
    cell_high_variants: dict = {}

    for cell_id in cell_segments:
        cell_idx = df.index.get_loc(cell_id)
        all_loh_variants: set = set()
        for start, end in cell_segments[cell_id]:
            all_loh_variants.update(range(start, end + 1))
        loh_indices = sorted(all_loh_variants)
        cell_low_variants[cell_id] = {
            i for i in loh_indices if X_low[cell_idx, i] == 1
        }
        cell_high_variants[cell_id] = {
            i for i in loh_indices if X_high[cell_idx, i] == 1
        }
    return cell_low_variants, cell_high_variants


def _greedy_directional_clusters(
    all_cells: list,
    cell_low_variants: dict,
    cell_high_variants: dict,
    min_shared_variants: int,
    min_cluster_size: int,
) -> dict:
    """Single-pass greedy clustering on shared direction-consistent variants.

    Each unprocessed seed cell absorbs every later unprocessed cell with
    ``|LOW∩LOW| + |HIGH∩HIGH| >= min_shared_variants`` (compared with the seed
    only). Groups smaller than *min_cluster_size* are discarded, but their
    cells stay "processed". Returns ``{cell_id: cluster_id}``.
    """
    cell_to_cluster: dict = {}
    cluster_counter = 0
    processed_cells: set = set()

    for i, cell_a in enumerate(all_cells):
        if cell_a in processed_cells:
            continue
        current_cluster = {cell_a}
        processed_cells.add(cell_a)

        for cell_b in all_cells[i + 1 :]:
            if cell_b in processed_cells:
                continue
            shared_directional = len(
                cell_low_variants[cell_a] & cell_low_variants[cell_b]
            ) + len(cell_high_variants[cell_a] & cell_high_variants[cell_b])
            if shared_directional >= min_shared_variants:
                current_cluster.add(cell_b)
                processed_cells.add(cell_b)

        if len(current_cluster) >= min_cluster_size:
            for cell in current_cluster:
                cell_to_cluster[cell] = cluster_counter
            cluster_counter += 1
    return cell_to_cluster


def _empty_cluster_record() -> dict:
    """Default per-cluster record used by the cluster-info defaultdict."""
    return {
        "cells": [],
        "n_low": 0,
        "n_high": 0,
        "shared_low": set(),
        "shared_high": set(),
    }


def _build_cluster_info(
    cell_to_cluster: dict,
    cell_low_variants: dict,
    cell_high_variants: dict,
) -> defaultdict:
    """Group cells by cluster and record the LOW/HIGH variants shared by all.

    Only cells that have their own LOH segments contribute to the shared sets;
    a cluster with no such cells keeps empty defaults.
    """
    cluster_info: defaultdict = defaultdict(_empty_cluster_record)
    for cell_id, cluster_id in cell_to_cluster.items():
        cluster_info[cluster_id]["cells"].append(cell_id)

    for cluster_id, info in cluster_info.items():
        cluster_cells_with_loh = [
            c for c in info["cells"] if c in cell_low_variants
        ]
        if not cluster_cells_with_loh:
            continue
        elif len(cluster_cells_with_loh) == 1:
            cell = cluster_cells_with_loh[0]
            info["shared_low"] = cell_low_variants[cell]
            info["shared_high"] = cell_high_variants[cell]
        else:
            shared_low = cell_low_variants[cluster_cells_with_loh[0]].copy()
            shared_high = cell_high_variants[cluster_cells_with_loh[0]].copy()
            for cell in cluster_cells_with_loh[1:]:
                shared_low &= cell_low_variants[cell]
                shared_high &= cell_high_variants[cell]
            info["shared_low"] = shared_low
            info["shared_high"] = shared_high
        info["n_low"] = len(info["shared_low"])
        info["n_high"] = len(info["shared_high"])
    return cluster_info


def _pairwise_coherence_rs(X: np.ndarray, cell_indices: list) -> list:
    """Full-arm pairwise similarity scores between the given cell rows.

    Pearson r for pairs with >= 3 shared finite variants; when either profile
    is constant, 1.0 if the mean |ΔAF| < 15 else 0.0.
    """
    pairwise_rs: list = []
    for ii in range(len(cell_indices)):
        for jj in range(ii + 1, len(cell_indices)):
            profile_i = X[cell_indices[ii], :]
            profile_j = X[cell_indices[jj], :]
            valid_mask = ~(np.isnan(profile_i) | np.isnan(profile_j))
            if valid_mask.sum() < _COHERENCE_MIN_VALID_VARIANTS:
                continue
            pi = profile_i[valid_mask]
            pj = profile_j[valid_mask]
            std_i = float(np.std(pi))
            std_j = float(np.std(pj))
            if std_i < _CONSTANT_PROFILE_STD_EPS or std_j < _CONSTANT_PROFILE_STD_EPS:
                # Pearson is undefined for a constant profile: near-identical
                # profiles count as agreement, different ones as none.
                mad = float(np.mean(np.abs(pi - pj)))
                pairwise_rs.append(1.0 if mad < _CONSTANT_PROFILE_MAX_MEAN_ABS_DIFF else 0.0)
                continue
            r, _ = pearsonr(pi, pj)
            if np.isfinite(r):
                pairwise_rs.append(r)
    return pairwise_rs


def _drop_incoherent_clusters(
    cluster_info: defaultdict,
    cell_to_cluster: dict,
    df: pd.DataFrame,
    X: np.ndarray,
    keep_incoherent_clusters: bool,
    vprint: Callable[[str], None],
) -> None:
    """Step 3b: remove clusters whose cells do not resemble each other arm-wide.

    Mutates *cluster_info* and *cell_to_cluster* in place.
    """
    coherence_r_threshold = _COHERENCE_R_THRESHOLD
    clusters_to_drop: set = set()
    for cluster_id, info in cluster_info.items():
        cluster_cells = info["cells"]
        n_cc = len(cluster_cells)
        if n_cc < 2:
            continue

        cell_indices = [df.index.get_loc(c) for c in cluster_cells if c in df.index]
        if len(cell_indices) < 2:
            clusters_to_drop.add(cluster_id)
            continue

        pairwise_rs = _pairwise_coherence_rs(X, cell_indices)

        if not pairwise_rs:
            vprint(f"    Cluster {cluster_id} ({n_cc} cells): no valid pairs → DROP")
            clusters_to_drop.add(cluster_id)
            continue

        pairwise_rs = [r for r in pairwise_rs if np.isfinite(r)]
        if not pairwise_rs:
            vprint(f"    Cluster {cluster_id} ({n_cc} cells): all NaN pairs → DROP")
            clusters_to_drop.add(cluster_id)
            continue

        mean_r = float(np.mean(pairwise_rs))
        min_r = float(np.min(pairwise_rs))
        frac_good = float(np.mean(np.array(pairwise_rs) >= coherence_r_threshold))

        vprint(
            f"    Cluster {cluster_id} ({n_cc} cells): "
            f"mean_r={mean_r:.3f}, min_r={min_r:.3f}, "
            f"frac≥{coherence_r_threshold:.2f}={frac_good:.0%}"
        )

        if frac_good < _COHERENCE_MIN_FRAC:
            if keep_incoherent_clusters:
                vprint(f"      → KEEP (intra-cluster coherence too low, but keep_incoherent_clusters=True)")
            else:
                vprint(f"      → DROP (intra-cluster coherence too low)")
                clusters_to_drop.add(cluster_id)

    if clusters_to_drop:
        for cid in clusters_to_drop:
            cells_in_cluster = cluster_info[cid]["cells"]
            for cell_id in cells_in_cluster:
                if cell_id in cell_to_cluster:
                    del cell_to_cluster[cell_id]
            del cluster_info[cid]
        vprint(f"    Dropped {len(clusters_to_drop)} incoherent cluster(s)")
        vprint(f"    Remaining clustered cells: {len(cell_to_cluster)}")
    else:
        vprint(f"    All {len(cluster_info)} clusters passed coherence check")


def _map_cells_to_barcodes(
    cell_to_cluster: dict,
    df: pd.DataFrame,
    barcodes: pd.Series,
) -> dict:
    """Map df row ids to Sample barcodes (identity, ``barcode`` column, or position)."""
    sample_index_match = list(cell_to_cluster.keys())[0] in barcodes.values
    if sample_index_match:
        return {cell: cell for cell in df.index}
    if "barcode" in df.columns:
        return df["barcode"].to_dict()
    return dict(zip(df.index, barcodes))


def _clone_mean_profiles(
    all_clone_to_cells: dict,
    df: pd.DataFrame,
    X: np.ndarray,
) -> dict:
    """Arm-wide nan-mean AF profile per clone label."""
    clone_mean_profiles: dict = {}
    for clone_label, cells in all_clone_to_cells.items():
        cell_indices = [df.index.get_loc(c) for c in cells if c in df.index]
        if cell_indices:
            clone_mean_profiles[clone_label] = np.nanmean(
                X[cell_indices, :], axis=0
            )
    return clone_mean_profiles


def _find_cluster_merges(
    base_clone_to_clones: dict,
    clone_mean_profiles: dict,
    correlation_threshold: float,
    vprint: Callable[[str], None],
) -> dict:
    """Pairs of same-base LOH clones with profile r >= threshold.

    Returns ``{merge_from_cluster: merge_into_cluster}`` (larger id → smaller).
    """
    vprint("    Checking for cluster merges...")
    cluster_merges: dict = {}
    merge_count = 0
    total_comparisons = 0

    for base_name, clones in base_clone_to_clones.items():
        loh_clones_for_base = [c for c in clones if is_loh_label(c)]
        if len(loh_clones_for_base) < 2:
            continue
        for i, clone_a in enumerate(loh_clones_for_base):
            for clone_b in loh_clones_for_base[i + 1 :]:
                if (
                    clone_a not in clone_mean_profiles
                    or clone_b not in clone_mean_profiles
                ):
                    continue
                profile_a = clone_mean_profiles[clone_a]
                profile_b = clone_mean_profiles[clone_b]
                valid_mask = ~(np.isnan(profile_a) | np.isnan(profile_b))
                if valid_mask.sum() < _MIN_VALID_VARIANTS_FOR_CORR:
                    continue
                r, _ = pearsonr(profile_a[valid_mask], profile_b[valid_mask])
                total_comparisons += 1
                if r >= correlation_threshold:
                    match_a = re.search(r"LOH-C(\d+)", clone_a)
                    match_b = re.search(r"LOH-C(\d+)", clone_b)
                    if match_a and match_b:
                        cluster_a = int(match_a.group(1))
                        cluster_b = int(match_b.group(1))
                        merge_into = min(cluster_a, cluster_b)
                        merge_from = max(cluster_a, cluster_b)
                        cluster_merges[merge_from] = merge_into
                        merge_count += 1

    vprint(f"      {total_comparisons} comparisons, {merge_count} merges")
    return cluster_merges


def _apply_cluster_merges(
    cluster_merges: dict,
    cell_to_cluster: dict,
    vprint: Callable[[str], None],
) -> dict:
    """Resolve merge chains (cycle-safe) and relabel *cell_to_cluster* in place.

    Returns ``{merge_from: final_destination}``.
    """
    resolved_merges: dict = {}
    vprint("    Applying cluster merges...")
    for merge_from, merge_into in cluster_merges.items():
        final_destination = merge_into
        visited = {merge_from}
        while (
            final_destination in cluster_merges
            and final_destination not in visited
        ):
            visited.add(final_destination)
            final_destination = cluster_merges[final_destination]
        resolved_merges[merge_from] = final_destination

    vprint(f"      Resolved {len(resolved_merges)} merge chains")
    for cell_id in list(cell_to_cluster.keys()):
        cluster_id = cell_to_cluster[cell_id]
        if cluster_id in resolved_merges:
            cell_to_cluster[cell_id] = resolved_merges[cluster_id]
    vprint(f"      Merges applied to {len(cell_to_cluster)} cells")
    return resolved_merges


def _expand_to_non_loh_clones(
    base_clone_to_clones: dict,
    clone_mean_profiles: dict,
    all_clone_to_cells: dict,
    resolved_merges: dict,
    correlation_threshold: float,
    vprint: Callable[[str], None],
) -> Tuple[list, dict]:
    """Absorb same-base non-LOH clones into correlated, not-smaller LOH clones.

    Returns the comparison records and ``{cell_id: cluster_id}`` for every cell
    of an absorbed non-LOH clone.
    """
    vprint("    Expanding to non-LOH clones...")
    correlation_results: list = []
    additional_loh_assignments: dict = {}
    expansion_count = 0
    expansion_comparisons = 0

    for base_name, clones in base_clone_to_clones.items():
        loh_clones_for_base = [c for c in clones if is_loh_label(c)]
        non_loh_clones_for_base = [c for c in clones if not is_loh_label(c)]
        if not loh_clones_for_base or not non_loh_clones_for_base:
            continue

        for loh_clone in loh_clones_for_base:
            if loh_clone not in clone_mean_profiles:
                continue
            loh_profile = clone_mean_profiles[loh_clone]
            loh_clone_size = len(all_clone_to_cells[loh_clone])
            match = re.search(r"LOH-C(\d+)", loh_clone)
            if not match:
                continue
            loh_cluster_id = int(match.group(1))
            if loh_cluster_id in resolved_merges:
                loh_cluster_id = resolved_merges[loh_cluster_id]

            for non_loh_clone in non_loh_clones_for_base:
                if non_loh_clone not in clone_mean_profiles:
                    continue
                non_loh_clone_size = len(all_clone_to_cells[non_loh_clone])
                non_loh_profile = clone_mean_profiles[non_loh_clone]
                valid_mask = ~(
                    np.isnan(loh_profile) | np.isnan(non_loh_profile)
                )
                if valid_mask.sum() < _MIN_VALID_VARIANTS_FOR_CORR:
                    continue
                r, p_value = pearsonr(
                    loh_profile[valid_mask], non_loh_profile[valid_mask]
                )
                expansion_comparisons += 1

                correlation_results.append(
                    {
                        "base_clone": base_name,
                        "loh_clone": loh_clone,
                        "non_loh_clone": non_loh_clone,
                        "correlation": r,
                        "p_value": p_value,
                        "n_valid_variants": valid_mask.sum(),
                        "loh_size": loh_clone_size,
                        "non_loh_size": non_loh_clone_size,
                        "size_ratio": (
                            non_loh_clone_size / loh_clone_size
                            if loh_clone_size > 0
                            else float("inf")
                        ),
                    }
                )

                if (
                    r >= correlation_threshold
                    and non_loh_clone_size <= loh_clone_size
                ):
                    for cell in all_clone_to_cells[non_loh_clone]:
                        additional_loh_assignments[cell] = loh_cluster_id
                        expansion_count += 1

    vprint(
        f"      Total comparisons: {expansion_comparisons}, cells expanded: {expansion_count}"
    )
    return correlation_results, additional_loh_assignments


def _correlation_merge_and_expand(
    cell_to_cluster: dict,
    cluster_info: defaultdict,
    cell_low_variants: dict,
    cell_high_variants: dict,
    df: pd.DataFrame,
    X: np.ndarray,
    cell_to_barcode: dict,
    barcode_to_backup_clone: dict,
    target_chr: str,
    correlation_threshold: float,
    vprint: Callable[[str], None],
) -> Tuple[pd.DataFrame, dict, defaultdict]:
    """Step 5: merge correlated LOH clusters and expand into non-LOH clones.

    Mutates *cell_to_cluster* in place. Returns the clone–clone correlation
    table, the cells assigned by expansion, and the (possibly rebuilt)
    cluster info.
    """
    vprint("\n  Correlation Analysis:")

    cell_to_backup_clone: dict = {}
    for cell_id in df.index:
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            if barcode in barcode_to_backup_clone:
                cell_to_backup_clone[cell_id] = barcode_to_backup_clone[barcode]

    clone_to_cells: dict = defaultdict(list)
    for cell_id, clone_label in cell_to_backup_clone.items():
        clone_to_cells[clone_label].append(cell_id)

    base_clone_to_clones: dict = defaultdict(list)
    for clone_label in clone_to_cells:
        base_clone_to_clones[_get_base_clone(clone_label)].append(clone_label)

    temp_loh_clones: dict = {}
    for cell_id, cluster_id in cell_to_cluster.items():
        if cell_id in cell_to_backup_clone:
            base_clone = cell_to_backup_clone[cell_id]
            temp_loh_clones[cell_id] = _loh_clone_label(base_clone, cluster_id, target_chr)

    temp_clone_to_cells: dict = defaultdict(list)
    for cell_id, temp_clone in temp_loh_clones.items():
        temp_clone_to_cells[temp_clone].append(cell_id)

    for temp_clone in temp_clone_to_cells:
        base_name = _get_base_clone(temp_clone)
        if temp_clone not in base_clone_to_clones[base_name]:
            base_clone_to_clones[base_name].append(temp_clone)

    all_clone_to_cells: dict = defaultdict(list)
    all_clone_to_cells.update(clone_to_cells)
    all_clone_to_cells.update(temp_clone_to_cells)

    correlation_results: list = []
    additional_loh_assignments: dict = {}

    if temp_clone_to_cells:
        vprint(f"    Computing mean profiles for {len(all_clone_to_cells)} clones...")
        clone_mean_profiles = _clone_mean_profiles(all_clone_to_cells, df, X)
        vprint(
            f"      Progress: {len(all_clone_to_cells)}/{len(all_clone_to_cells)} clones - DONE"
        )

        cluster_merges = _find_cluster_merges(
            base_clone_to_clones, clone_mean_profiles, correlation_threshold, vprint
        )
        resolved_merges: dict = {}
        if cluster_merges:
            resolved_merges = _apply_cluster_merges(cluster_merges, cell_to_cluster, vprint)

        correlation_results, additional_loh_assignments = _expand_to_non_loh_clones(
            base_clone_to_clones,
            clone_mean_profiles,
            all_clone_to_cells,
            resolved_merges,
            correlation_threshold,
            vprint,
        )

        if additional_loh_assignments:
            vprint("    Updating cell assignments...")
            for cell_id, cluster_id in additional_loh_assignments.items():
                cell_to_cluster[cell_id] = cluster_id
            vprint("      Assignments updated")

        vprint("    Rebuilding cluster info...")
        cluster_info = _build_cluster_info(
            cell_to_cluster, cell_low_variants, cell_high_variants
        )
        vprint("      Cluster info rebuilt")

    df_correlations = (
        pd.DataFrame(correlation_results)
        if correlation_results
        else pd.DataFrame()
    )
    return df_correlations, additional_loh_assignments, cluster_info


def _filter_by_final_clone_size(
    cell_to_cluster: dict,
    cell_to_barcode: dict,
    barcode_to_backup_clone: dict,
    target_chr: str,
    min_final_clone_size: int,
    vprint: Callable[[str], None],
) -> Optional[dict]:
    """Step 6: keep cells whose final ``<base> LOH-C<id> <arm>`` clone is large enough.

    Returns the filtered ``{cell_id: cluster_id}``, or None if no clone passes.
    """
    vprint("\n  Pre-checking Final Clone Sizes:")

    final_clone_counts: dict = defaultdict(int)
    for cell_id, cluster_id in cell_to_cluster.items():
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            if barcode in barcode_to_backup_clone:
                base_clone = barcode_to_backup_clone[barcode]
                final_clone_counts[_loh_clone_label(base_clone, cluster_id, target_chr)] += 1

    valid_final_clones = {
        label
        for label, count in final_clone_counts.items()
        if count >= min_final_clone_size
    }
    vprint(
        f"    Valid final clones (≥{min_final_clone_size} cells): {len(valid_final_clones)}"
    )

    if not valid_final_clones:
        vprint(f"    ⚠️  No final clones meet threshold")
        return None

    filtered_cell_to_cluster: dict = {}
    for cell_id, cluster_id in cell_to_cluster.items():
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            if barcode in barcode_to_backup_clone:
                base_clone = barcode_to_backup_clone[barcode]
                final_clone_label = _loh_clone_label(base_clone, cluster_id, target_chr)
                if final_clone_label in valid_final_clones:
                    filtered_cell_to_cluster[cell_id] = cluster_id

    vprint(f"    Filtered cells: {len(filtered_cell_to_cluster)}")
    return filtered_cell_to_cluster


def _build_output_frames(
    df: pd.DataFrame,
    X: np.ndarray,
    variants_chr_sorted: list,
    cell_segments: dict,
    cell_to_cluster: dict,
    cluster_info: defaultdict,
    additional_loh_assignments: dict,
    target_chr: str,
    low_allele_threshold: float,
    high_allele_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 7: per-segment records and the per-cell LOH subset of *df*."""
    sorted_cells: list = []
    for cluster_id in sorted(cluster_info.keys()):
        sorted_cells.extend(cluster_info[cluster_id]["cells"])

    segment_records: list = []
    for cell_id in sorted_cells:
        cluster_id = cell_to_cluster[cell_id]
        info = cluster_info[cluster_id]
        if cell_id not in cell_segments:
            continue
        for start_idx, end_idx in cell_segments[cell_id]:
            start_var = variants_chr_sorted[start_idx]
            end_var = variants_chr_sorted[end_idx]
            pos_start = _extract_pos(start_var)
            pos_end = _extract_pos(end_var)
            cell_idx = df.index.get_loc(cell_id)
            segment_values = X[cell_idx, start_idx : end_idx + 1]

            segment_records.append(
                {
                    "cell": cell_id,
                    "cluster": cluster_id,
                    "chr_arm": target_chr,
                    "start_variant": start_var,
                    "end_variant": end_var,
                    "genomic_start": pos_start,
                    "genomic_end": pos_end,
                    "length_bp": pos_end - pos_start,
                    "n_variants": end_idx - start_idx + 1,
                    "mean_allele_freq": np.nanmean(segment_values),
                    "median_allele_freq": np.nanmedian(segment_values),
                    "extreme_fraction": (
                        (segment_values <= low_allele_threshold)
                        | (segment_values >= high_allele_threshold)
                    ).sum()
                    / len(segment_values),
                    "cluster_shared_low": info["n_low"],
                    "cluster_shared_high": info["n_high"],
                }
            )

    df_loh_segments = pd.DataFrame(segment_records)

    df_loh_only = df.loc[sorted_cells].copy()
    df_loh_only["loh_cluster"] = df_loh_only.index.map(
        lambda c: cell_to_cluster[c]
    )
    df_loh_only["assigned_by_correlation"] = df_loh_only.index.map(
        lambda c: c in additional_loh_assignments
    )

    if len(df_loh_segments) > 0:
        summary = (
            df_loh_segments.groupby("cell")
            .agg(
                {
                    "n_variants": "sum",
                    "length_bp": "sum",
                    "mean_allele_freq": "mean",
                    "extreme_fraction": "mean",
                }
            )
            .add_prefix("total_loh_")
        )
        for col in summary.columns:
            df_loh_only[col] = df_loh_only.index.map(
                lambda c: summary.loc[c, col] if c in summary.index else 0
            )
    return df_loh_segments, df_loh_only


def _count_loh_cells(Sample, target_chr: str) -> Tuple[int, int]:
    """(cells with any active LOH label, cells whose label mentions *target_chr*)."""
    loh_count = sum(
        is_loh_label(label) for label in Sample.dna.row_attrs["Clone"]
    )
    loh_for_this_chr = (
        pd.Series(Sample.dna.row_attrs["Clone"])
        .str.contains(target_chr, na=False)
        .sum()
    )
    return loh_count, loh_for_this_chr


def _append_loh_annotations(
    Sample,
    df_loh_only: pd.DataFrame,
    cell_to_barcode: dict,
    barcodes: pd.Series,
    target_chr: str,
    vprint: Callable[[str], None],
) -> None:
    """Step 8: append ``' LOH-C<id> <arm>'`` to each LOH cell's current label (once)."""
    vprint("\n  Updating final clone labels...")

    current_clones = pd.Series(Sample.dna.row_attrs["Clone"]).copy()

    loh_barcodes_to_cluster: dict = {}
    for cell_id in df_loh_only.index:
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            loh_barcodes_to_cluster[barcode] = df_loh_only.loc[
                cell_id, "loh_cluster"
            ]

    updated_clones = current_clones.copy()
    update_count = 0
    for idx, barcode in enumerate(barcodes):
        if barcode in loh_barcodes_to_cluster:
            current_label = updated_clones.iloc[idx]
            cluster_id = loh_barcodes_to_cluster[barcode]
            annotation = f" LOH-C{cluster_id} {target_chr}"
            if annotation not in current_label:
                updated_clones.iloc[idx] = current_label + annotation
                update_count += 1

    Sample.dna.row_attrs["Clone"] = updated_clones.values
    vprint(f"    Clone labels updated for {update_count} cells")

    loh_count, loh_for_this_chr = _count_loh_cells(Sample, target_chr)
    vprint(f"    Total cells with any LOH: {loh_count}")
    vprint(f"    Cells with LOH for {target_chr}: {loh_for_this_chr}")


def _homogeneity_record(final_label, n_cells, status, value, n_correlations=0) -> dict:
    """Homogeneity stats row for a clone that was not correlation-scored."""
    return {
        "final_clone": final_label,
        "n_cells": n_cells,
        "mean_correlation": value,
        "min_correlation": value,
        "n_correlations": n_correlations,
        "fraction_good_correlations": value,
        "status": status,
    }


def _score_final_clones(
    final_clone_to_cells: dict,
    df: pd.DataFrame,
    X: np.ndarray,
    skip_size_filtering: bool,
    min_final_clone_size: int,
    correlation_threshold: float,
    min_homogeneity_fraction: float,
    vprint: Callable[[str], None],
) -> Tuple[set, list]:
    """Pairwise-correlation homogeneity per final clone.

    Returns the labels to remove and the per-clone stats records.
    """
    final_clones_to_remove: set = set()
    final_homogeneity_stats: list = []

    clone_idx = 0
    for clone_idx, (final_label, clone_cells_list) in enumerate(
        final_clone_to_cells.items(), 1
    ):
        n_cells = len(clone_cells_list)
        if clone_idx % _PROGRESS_EVERY == 0:
            vprint(
                f"      Progress: {clone_idx}/{len(final_clone_to_cells)} clones"
            )

        if not skip_size_filtering and n_cells < min_final_clone_size:
            vprint(
                f"      {final_label}: {n_cells} cells < {min_final_clone_size} threshold - REMOVE"
            )
            final_clones_to_remove.add(final_label)
            final_homogeneity_stats.append(
                _homogeneity_record(final_label, n_cells, "REMOVED_SIZE", np.nan)
            )
            continue

        if n_cells < 2:
            final_homogeneity_stats.append(
                _homogeneity_record(final_label, n_cells, "KEPT", 1.0)
            )
            continue

        clone_cell_indices = [
            df.index.get_loc(c) for c in clone_cells_list if c in df.index
        ]
        if len(clone_cell_indices) < 2:
            final_clones_to_remove.add(final_label)
            continue

        correlations: list = []
        for ii in range(len(clone_cell_indices)):
            for jj in range(ii + 1, len(clone_cell_indices)):
                profile_i = X[clone_cell_indices[ii], :]
                profile_j = X[clone_cell_indices[jj], :]
                valid_mask = ~(np.isnan(profile_i) | np.isnan(profile_j))
                if valid_mask.sum() < _MIN_VALID_VARIANTS_FOR_CORR:
                    continue
                r, _ = pearsonr(profile_i[valid_mask], profile_j[valid_mask])
                correlations.append(r)

        if not correlations:
            final_clones_to_remove.add(final_label)
            continue

        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)
        n_good_correlations = sum(
            1 for r in correlations if r >= correlation_threshold
        )
        fraction_good = n_good_correlations / len(correlations)

        final_homogeneity_stats.append(
            {
                "final_clone": final_label,
                "n_cells": n_cells,
                "n_correlations": len(correlations),
                "mean_correlation": mean_corr,
                "min_correlation": min_corr,
                "fraction_good_correlations": fraction_good,
                "status": (
                    "KEPT"
                    if fraction_good >= min_homogeneity_fraction
                    else "REMOVED_CORRELATION"
                ),
            }
        )
        if fraction_good < min_homogeneity_fraction:
            final_clones_to_remove.add(final_label)

    if final_clone_to_cells and clone_idx % _PROGRESS_EVERY != 0:
        vprint(
            f"      Progress: {len(final_clone_to_cells)}/{len(final_clone_to_cells)} clones - DONE"
        )
    return final_clones_to_remove, final_homogeneity_stats


def _cells_with_final_label(df_loh_only, cell_to_barcode, barcode_to_clone) -> dict:
    """``{cell_id: final clone label}`` for LOH cells whose barcode has a label."""
    cell_to_final_clone: dict = {}
    for cell_id in df_loh_only.index:
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            if barcode in barcode_to_clone:
                cell_to_final_clone[cell_id] = barcode_to_clone[barcode]
    return cell_to_final_clone


def _strip_removed_clones(
    Sample,
    barcodes: pd.Series,
    final_clones_to_remove: set,
    final_homogeneity_stats: list,
    target_chr: str,
    correlation_threshold: float,
    vprint: Callable[[str], None],
) -> None:
    """Report removed clones and strip this arm's LOH annotation from their cells.

    Annotations from other arms are preserved.
    """
    vprint(
        f"    Removing {len(final_clones_to_remove)} non-homogeneous clones for {target_chr}..."
    )
    for final_label in final_clones_to_remove:
        stats = [
            s
            for s in final_homogeneity_stats
            if s["final_clone"] == final_label
        ]
        if stats:
            s = stats[0]
            vprint(f"      {final_label}:")
            vprint(f"        Cells: {s['n_cells']}")
            vprint(f"        Correlations: {s['n_correlations']}")
            vprint(f"        Mean correlation: {s['mean_correlation']:.3f}")
            vprint(
                f"        Fraction good (≥{correlation_threshold:.2f}): "
                f"{s['fraction_good_correlations']:.1%}"
            )
            vprint(f"        Status: {s['status']}")

    current_clones_with_all_loh = pd.Series(
        Sample.dna.row_attrs["Clone"]
    ).copy()
    removal_count = 0
    for idx, barcode in enumerate(barcodes):
        current_label = current_clones_with_all_loh.iloc[idx]
        if current_label in final_clones_to_remove:
            pattern = rf" LOH-C\d+ {re.escape(target_chr)}"
            cleaned_label = re.sub(pattern, "", current_label)
            current_clones_with_all_loh.iloc[idx] = cleaned_label
            removal_count += 1

    Sample.dna.row_attrs["Clone"] = current_clones_with_all_loh.values
    vprint(f"      Removed {target_chr} LOH from {removal_count} cells")
    vprint("      (Preserved LOH from other chromosomes)")


def _homogeneity_check(
    Sample,
    df: pd.DataFrame,
    X: np.ndarray,
    df_loh_segments: pd.DataFrame,
    df_loh_only: pd.DataFrame,
    df_correlations: pd.DataFrame,
    cell_to_barcode: dict,
    barcodes: pd.Series,
    target_chr: str,
    skip_size_filtering: bool,
    min_final_clone_size: int,
    correlation_threshold: float,
    min_homogeneity_fraction: float,
    vprint: Callable[[str], None],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 9: remove non-homogeneous final clones and return the four outputs."""
    vprint("\n  Homogeneity Check:")
    vprint(
        f"    Requiring ≥{min_homogeneity_fraction:.0%} of pairwise correlations to meet threshold"
    )

    updated_clones = pd.Series(Sample.dna.row_attrs["Clone"]).astype(str)
    barcode_to_clone = dict(zip(barcodes, updated_clones))

    final_clone_to_cells: dict = defaultdict(list)
    for cell_id in df_loh_only.index:
        if cell_id in cell_to_barcode:
            barcode = cell_to_barcode[cell_id]
            if barcode in barcode_to_clone:
                final_label = barcode_to_clone[barcode]
                if target_chr in final_label and is_loh_label(final_label):
                    final_clone_to_cells[final_label].append(cell_id)

    vprint(
        f"    Checking {len(final_clone_to_cells)} final clones for {target_chr}..."
    )

    final_clones_to_remove, final_homogeneity_stats = _score_final_clones(
        final_clone_to_cells,
        df,
        X,
        skip_size_filtering,
        min_final_clone_size,
        correlation_threshold,
        min_homogeneity_fraction,
        vprint,
    )

    df_final_homogeneity = pd.DataFrame(final_homogeneity_stats)

    if final_clones_to_remove:
        _strip_removed_clones(
            Sample,
            barcodes,
            final_clones_to_remove,
            final_homogeneity_stats,
            target_chr,
            correlation_threshold,
            vprint,
        )

        # barcode_to_clone still holds the pre-removal labels on purpose.
        kept_cells: list = []
        for cell_id in df_loh_only.index:
            if cell_id in cell_to_barcode:
                barcode = cell_to_barcode[cell_id]
                if barcode in barcode_to_clone:
                    final_label = barcode_to_clone[barcode]
                    if final_label not in final_clones_to_remove:
                        kept_cells.append(cell_id)

        if not kept_cells:
            vprint(f"      ⚠️  All cells removed for {target_chr}")
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                df_correlations,
                df_final_homogeneity,
            )

        df_loh_only = df_loh_only.loc[kept_cells].copy()
        vprint(f"      Kept {len(kept_cells)} cells for {target_chr}")
    else:
        vprint(
            f"    ✓ All {len(final_clone_to_cells)} clones passed for {target_chr}"
        )

    if len(df_final_homogeneity) > 0:
        cell_to_final_clone_map = _cells_with_final_label(
            df_loh_only, cell_to_barcode, barcode_to_clone
        )
        final_clone_to_stats = df_final_homogeneity.set_index(
            "final_clone"
        ).to_dict("index")
        for col in [
            "mean_correlation",
            "min_correlation",
            "n_correlations",
            "fraction_good_correlations",
        ]:
            df_loh_only[f"final_{col}"] = df_loh_only.index.map(
                lambda c: final_clone_to_stats.get(
                    cell_to_final_clone_map.get(c, ""), {}
                ).get(col, np.nan)
            )

    vprint(
        f"\n  ✅ {target_chr}: {len(df_loh_only)} cells with homogeneous LOH"
    )
    loh_count, loh_for_this_chr = _count_loh_cells(Sample, target_chr)
    vprint(
        f"  Final: Total LOH cells = {loh_count}, {target_chr} LOH cells = {loh_for_this_chr}"
    )

    return df_loh_segments, df_loh_only, df_correlations, df_final_homogeneity


def detect_loh_and_update_clones(
    Sample,
    df: pd.DataFrame,
    grouped_by_chr_arm: Dict[str, List[str]],
    target_chr: str,
    low_allele_threshold: float = 10.0,
    high_allele_threshold: float = 90.0,
    min_segment_length: int = 4,
    min_extreme_fraction: float = 0.80,
    min_shared_variants: int = 4,
    min_cluster_size: int = 5,
    min_final_clone_size: int = 5,
    correlation_threshold: float = 0.60,
    restore_from_backup: Optional[bool] = None,
    skip_homogeneity_check: bool = False,
    min_homogeneity_fraction: float = 0.70,
    skip_size_filtering: bool = False,
    cytobands_df: Optional[pd.DataFrame] = None,
    keep_incoherent_clusters: bool = False,
    verbose: bool = True,
    clone_reference_labels=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detect LOH patterns on one chromosome arm and annotate clone labels.

    Parameters
    ----------
    Sample : missionbio.mosaic Sample
        Uses ``Sample.dna.row_attrs['barcode']``, ``['Clone']`` and
        ``['Clone_backup']``. ``Clone`` is modified in place (and
        ``Clone_backup`` is created from it if missing and
        *clone_reference_labels* is None).
    df : pd.DataFrame
        Cells × variants AF matrix (percent, 0–100). The index is either the
        barcode, or rows are mapped to barcodes via a ``barcode`` column or,
        failing that, by position.
    grouped_by_chr_arm : dict[str, list[str]]
        Arm → variant names. Variants missing from *df* are ignored; the rest
        are sorted by genomic position (``chrN:POS...``).
    target_chr : str
        Arm to analyse, e.g. ``"chr17p"``.
    low_allele_threshold, high_allele_threshold : float, default 10.0, 90.0
        AF ≤ low is "LOW"/extreme, AF ≥ high is "HIGH"/extreme.
    min_segment_length : int, default 4
        Minimum contiguous extreme variants for a per-cell LOH segment.
    min_extreme_fraction : float, default 0.80
        Minimum extreme fraction within a segment.
    min_shared_variants : int, default 4
        Minimum direction-consistent shared extreme variants
        (``|LOW∩LOW| + |HIGH∩HIGH|``) for two cells to be grouped.
    min_cluster_size : int, default 5
        Minimum cells per initial cluster.
    min_final_clone_size : int, default 5
        Minimum cells per final ``<base> LOH-C<n> <arm>`` clone (unless
        *skip_size_filtering*).
    correlation_threshold : float, default 0.60
        Pearson r used for cluster merging, non-LOH expansion and pairwise
        homogeneity.
    restore_from_backup : bool or None, default None
        If True and ``Clone_backup`` exists, copy it into ``Clone`` first.
        Ignored when *clone_reference_labels* is given.
    skip_homogeneity_check : bool, default False
        Skip Step 9; the fourth output is then empty.
    min_homogeneity_fraction : float, default 0.70
        Fraction of within-clone pairwise r ≥ *correlation_threshold* needed
        to keep a clone.
    skip_size_filtering : bool, default False
        Defer the final clone-size gate (Phase 1 applies it later).
    cytobands_df : pd.DataFrame or None
        Only used to log how variants map onto cytobands.
    keep_incoherent_clusters : bool, default False
        Keep clusters that fail the arm-wide coherence check (diagnostics).
    verbose : bool, default True
        Print progress.
    clone_reference_labels : dict, pd.Series, sequence or None
        Frozen parent labels used instead of ``Clone_backup`` (e.g. final
        Phase-5 labels during the Phase-6 scan). A dict or a Series whose
        length differs from the Sample is matched by barcode (missing → "");
        otherwise labels are taken positionally.

    Returns
    -------
    df_loh_segments : pd.DataFrame
        One row per (LOH cell, segment): start/end variant and position,
        length, AF mean/median, extreme fraction, cluster-shared LOW/HIGH counts.
    df_loh_only : pd.DataFrame
        Rows of *df* for LOH cells plus ``loh_cluster``,
        ``assigned_by_correlation``, ``total_loh_*`` summaries and (after the
        homogeneity check) ``final_*`` clone statistics.
    df_correlations : pd.DataFrame
        LOH-clone vs same-base non-LOH-clone correlations from the expansion step.
    df_final_homogeneity : pd.DataFrame
        Per-clone homogeneity statistics (empty if the check is skipped).
        Every early exit returns four empty DataFrames, except when the
        homogeneity check removes all cells, which returns empty segment/cell
        frames with the correlation and homogeneity tables.

    Raises
    ------
    ValueError
        If *clone_reference_labels* cannot be aligned to ``Sample.dna`` rows.

    Notes
    -----
    Each arm is processed independently, and annotations accumulate in
    ``Clone``: the arm annotation is appended to the *current* label, while the
    base clone used for grouping comes from the reference labels.

    Clustering is a single greedy pass: each seed cell absorbs every later
    unassigned cell sharing ≥ *min_shared_variants* direction-consistent
    extreme variants with the seed. Clusters must then pass an arm-wide
    coherence check: ≥ 40 % of within-cluster pairwise r must be ≥ 0.30.
    Merges, expansion and homogeneity correlations require ≥ 10 shared
    finite variants.
    """
    vprint = _make_vprint(verbose)

    vprint("\n" + "=" * 60)
    vprint(f"PROCESSING: {target_chr}")
    vprint("=" * 60)

    sample_barcodes = pd.Series(Sample.dna.row_attrs["barcode"]).astype(str)
    reference_clones = _resolve_reference_clones(
        Sample, sample_barcodes, restore_from_backup, clone_reference_labels, vprint
    )

    _print_configuration(
        vprint,
        low_allele_threshold,
        high_allele_threshold,
        min_segment_length,
        min_extreme_fraction,
        min_shared_variants,
        min_cluster_size,
        min_final_clone_size,
        correlation_threshold,
        skip_size_filtering,
        skip_homogeneity_check,
        min_homogeneity_fraction,
    )

    # ── Step 1: Data preparation ──────────────────────────────────────
    vprint("\n  Data Preparation:")

    variants_chr_all = [
        v for v in grouped_by_chr_arm.get(target_chr, []) if v in df.columns
    ]
    variants_chr_sorted = sorted(
        variants_chr_all, key=lambda v: int(v.split(":")[1].split("-")[0])
    )
    vprint(f"    Variants on {target_chr}: {len(variants_chr_sorted)}")

    if len(variants_chr_sorted) == 0:
        vprint(f"    ⚠️  No variants found for {target_chr}")
        return _empty_result()

    X = df[variants_chr_sorted].values
    X_extreme = (
        (X <= low_allele_threshold) | (X >= high_allele_threshold)
    ).astype(float)
    X_low = (X <= low_allele_threshold).astype(float)
    X_high = (X >= high_allele_threshold).astype(float)

    # Cytoband maps are logged only; they never gate detection.
    var_idx_to_band, band_to_var_indices = build_cytoband_maps(
        variants_chr_sorted, cytobands_df, target_chr
    )
    if var_idx_to_band is not None:
        n_mapped = len(var_idx_to_band)
        n_bands = len(band_to_var_indices)
        vprint(
            f"    Cytoband mapping: {n_mapped}/{len(variants_chr_sorted)} variants "
            f"mapped to {n_bands} cytoband{'s' if n_bands != 1 else ''} (informational only)"
        )
    else:
        vprint("    Cytoband mapping: not available")

    vprint(f"    Cells: {X.shape[0]}")
    vprint(
        f"    Extreme values: {X_extreme.sum():.0f} "
        f"({100 * X_extreme.sum() / X_extreme.size:.1f}%)"
    )

    # ── Step 2: Per-cell contiguous LOH segments ──────────────────────
    vprint("\n  Spatial Pattern Detection:")

    cell_segments: dict = {}
    for i, cell_id in enumerate(df.index):
        segments = _find_loh_segments_1d(
            X_extreme[i, :],
            X[i, :],
            min_segment_length,
            min_extreme_fraction,
            low_allele_threshold,
            high_allele_threshold,
        )
        if segments:
            cell_segments[cell_id] = segments

    vprint(f"    Cells with LOH segments: {len(cell_segments)}")

    if len(cell_segments) == 0:
        vprint("    ⚠️  No segments found")
        return _empty_result()

    # ── Step 3: Group cells by shared directional variants ────────────
    vprint("\n  Grouping by Shared Directional Variants:")

    cell_low_variants, cell_high_variants = _directional_variant_sets(
        cell_segments, df, X_low, X_high
    )
    cell_to_cluster = _greedy_directional_clusters(
        list(cell_segments.keys()),
        cell_low_variants,
        cell_high_variants,
        min_shared_variants,
        min_cluster_size,
    )

    vprint(f"    Clustered cells: {len(cell_to_cluster)}")

    if len(cell_to_cluster) == 0:
        vprint("    ⚠️  No valid clusters found")
        return _empty_result()

    cluster_info = _build_cluster_info(
        cell_to_cluster, cell_low_variants, cell_high_variants
    )
    vprint(f"    Valid clusters: {len(cluster_info)}")

    # ── Step 3b: Intra-cluster coherence validation (full arm) ────────
    vprint("\n  Intra-cluster Coherence Validation (full-arm):")
    _drop_incoherent_clusters(
        cluster_info, cell_to_cluster, df, X, keep_incoherent_clusters, vprint
    )

    if len(cell_to_cluster) == 0:
        vprint("    ⚠️  No valid clusters after coherence check")
        return _empty_result()

    # ── Step 4: Cell ↔ barcode ↔ base-clone mapping ───────────────────
    barcodes = sample_barcodes.copy()
    cell_to_barcode = _map_cells_to_barcodes(cell_to_cluster, df, barcodes)
    barcode_to_backup_clone = dict(zip(barcodes, reference_clones.copy()))

    # ── Step 5: Correlation-based merging and expansion ───────────────
    df_correlations, additional_loh_assignments, cluster_info = _correlation_merge_and_expand(
        cell_to_cluster,
        cluster_info,
        cell_low_variants,
        cell_high_variants,
        df,
        X,
        cell_to_barcode,
        barcode_to_backup_clone,
        target_chr,
        correlation_threshold,
        vprint,
    )

    # ── Step 6: Optional size filtering ───────────────────────────────
    if not skip_size_filtering:
        filtered = _filter_by_final_clone_size(
            cell_to_cluster,
            cell_to_barcode,
            barcode_to_backup_clone,
            target_chr,
            min_final_clone_size,
            vprint,
        )
        if filtered is None:
            return _empty_result()
        cell_to_cluster = filtered
        cluster_info = _build_cluster_info(
            cell_to_cluster, cell_low_variants, cell_high_variants
        )
    else:
        vprint(
            "\n  Size filtering: DEFERRED (will be applied after cross-chromosome merging)"
        )

    # ── Step 7: Output DataFrames ─────────────────────────────────────
    vprint("\n  Creating output DataFrames...")
    df_loh_segments, df_loh_only = _build_output_frames(
        df,
        X,
        variants_chr_sorted,
        cell_segments,
        cell_to_cluster,
        cluster_info,
        additional_loh_assignments,
        target_chr,
        low_allele_threshold,
        high_allele_threshold,
    )
    vprint("    DataFrames created")

    # ── Step 8: Final clone label update (accumulate) ─────────────────
    barcodes = pd.Series(Sample.dna.row_attrs["barcode"]).astype(str)
    _append_loh_annotations(
        Sample, df_loh_only, cell_to_barcode, barcodes, target_chr, vprint
    )

    # ── Step 9: Optional homogeneity check ────────────────────────────
    if skip_homogeneity_check:
        vprint("\n  Homogeneity Check: SKIPPED")
        vprint(
            f"  ✅ {target_chr}: {len(df_loh_only)} cells with LOH (no filtering)"
        )
        return df_loh_segments, df_loh_only, df_correlations, pd.DataFrame()

    return _homogeneity_check(
        Sample,
        df,
        X,
        df_loh_segments,
        df_loh_only,
        df_correlations,
        cell_to_barcode,
        barcodes,
        target_chr,
        skip_size_filtering,
        min_final_clone_size,
        correlation_threshold,
        min_homogeneity_fraction,
        vprint,
    )
