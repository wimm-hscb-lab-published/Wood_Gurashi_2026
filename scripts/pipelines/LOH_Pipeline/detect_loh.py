"""
LOH Pipeline – Core detection function.

Contains ``detect_loh_and_update_clones()``, the per-arm LOH detection
engine that is called by Phase 1 for each chromosome arm.

This was previously a method inside CNV_Utils. It is now a standalone
module so it can be imported, tested, and versioned independently.

PATCH
-----
Detection no longer uses cytobands as a hard gate for segment discovery.
Contiguous LOH segments are detected arm-wide (by contiguous extreme runs),
and cytobands are used later for QC / annotation in Phase 1.
This prevents sparse amplicon panels (1–2 variants per band) from
suppressing true LOH blocks during the initial detection step.
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.ndimage import label as ndimage_label
from scipy.stats import pearsonr

from .loh_config import build_cytoband_maps, is_loh_label


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
    Detect LOH patterns and update clone labels for a specific chromosomal arm.

    IMPORTANT:
    - By default, LOH analysis is based on ``Clone_backup`` (the original
      base-clone annotations). Pass ``clone_reference_labels`` to supply a
      frozen alternative parent-label vector, for example final Phase-5
      annotations during a targeted post-pipeline subclone scan.
    - Each chromosome arm is processed independently.
    - LOH annotations are accumulated in Sample.dna.row_attrs['Clone'].

    NOTE ON CLUSTERING DIRECTIONAL CONSISTENCY (PATCHED):
    - Cells can share a mixed pattern across variants (e.g. HIGH HIGH LOW LOW).
    - Two cells are considered similar if they share >= min_shared_variants
      where BOTH are extreme AND the direction matches per-variant.
    - Implementation: shared = |LOW∩LOW| + |HIGH∩HIGH|.

    NOTE ON CYTOBANDS (PATCH):
    - Cytoband mappings are built and logged for downstream QC/annotation,
      but are NOT used to block initial LOH segment detection.
    - Segment detection is arm-wide contiguous-run detection based on
      extreme values only. This avoids false negatives on sparse panels.

    Returns
    -------
    df_loh_segments : pd.DataFrame
        Per-cell segment details (start/end variant, extreme fraction, etc.).
    df_loh_only : pd.DataFrame
        Subset of df for LOH cells, with cluster and correlation columns.
    df_correlations : pd.DataFrame
        Clone–clone correlation results from expansion step.
    df_final_homogeneity : pd.DataFrame
        Per-clone homogeneity statistics (empty if homogeneity check skipped).
    """

    # ── helpers ────────────────────────────────────────────────────────
    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    def extract_pos(variant_name: str) -> int:
        try:
            return int(variant_name.split(":")[1].split("-")[0])
        except Exception:
            return 0

    def find_loh_segments_1d(
        row_binary: np.ndarray,
        row_values: np.ndarray,
        min_length: int = min_segment_length,
        min_extreme_frac: float = min_extreme_fraction,
        var_idx_to_band: Optional[Dict[int, str]] = None,      # kept for API compatibility
        band_to_var_indices: Optional[Dict[str, List[int]]] = None,  # kept for API compatibility
    ) -> List[Tuple[int, int]]:
        """
        Detect contiguous LOH segments on one arm for one cell.

        PATCHED behavior:
        - Detect by contiguous runs of extreme variants (row_binary == 1)
        - Require run length >= min_length
        - Require segment-level extreme fraction >= min_extreme_frac
        - Do NOT require a single cytoband to contain >= min_length variants
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

    def get_base_clone(clone_label):
        return re.sub(r" LOH-C\d+ \S+", "", clone_label)

    # ── Reference parent labels / legacy backup handling ───────────────
    vprint("\n" + "=" * 60)
    vprint(f"PROCESSING: {target_chr}")
    vprint("=" * 60)

    sample_barcodes = pd.Series(Sample.dna.row_attrs["barcode"]).astype(str)

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

            reference_clones = pd.Series(
                Sample.dna.row_attrs["Clone_backup"]
            ).astype(str).reset_index(drop=True)
        except Exception as e:
            vprint(f"❌ Error with backup: {repr(e)}")
            raise
    else:
        # Post-pipeline refinement mode. Do not overwrite Clone_backup; caller
        # supplies a frozen parent-label vector for this discovery pass.
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

    # ── configuration summary ─────────────────────────────────────────
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = df[variants_chr_sorted].values
    X_extreme = (
        (X <= low_allele_threshold) | (X >= high_allele_threshold)
    ).astype(float)
    X_low = (X <= low_allele_threshold).astype(float)
    X_high = (X >= high_allele_threshold).astype(float)

    # Build cytoband maps for downstream logging / later QC (not for gating detection)
    _var_idx_to_band, _band_to_var_indices = build_cytoband_maps(
        variants_chr_sorted, cytobands_df, target_chr
    )
    if _var_idx_to_band is not None:
        n_mapped = len(_var_idx_to_band)
        n_bands = len(_band_to_var_indices)
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

    # ── Step 2: Identify spatial LOH patterns ─────────────────────────
    vprint("\n  Spatial Pattern Detection:")

    cell_segments: dict = {}
    segment_matrix = np.zeros_like(X_extreme)

    for i, cell_id in enumerate(df.index):
        # PATCH: do NOT use cytobands as hard gate here
        segments = find_loh_segments_1d(
            X_extreme[i, :],
            X[i, :],
            var_idx_to_band=None,
            band_to_var_indices=None,
        )
        if segments:
            cell_segments[cell_id] = segments
            for start, end in segments:
                segment_matrix[i, start : end + 1] = 1

    n_cells_with_segments = len(cell_segments)
    vprint(f"    Cells with LOH segments: {n_cells_with_segments}")

    if n_cells_with_segments == 0:
        vprint("    ⚠️  No segments found")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Step 3: Group cells by shared directional variants ────────────
    vprint("\n  Grouping by Shared Directional Variants:")

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

    cell_to_cluster: dict = {}
    cluster_counter = 0
    processed_cells: set = set()
    all_cells = list(cell_segments.keys())

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

    vprint(f"    Clustered cells: {len(cell_to_cluster)}")

    if len(cell_to_cluster) == 0:
        vprint("    ⚠️  No valid clusters found")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build cluster info
    cluster_info: dict = defaultdict(
        lambda: {
            "cells": [],
            "n_low": 0,
            "n_high": 0,
            "shared_low": set(),
            "shared_high": set(),
        }
    )
    for cell_id, cluster_id in cell_to_cluster.items():
        cluster_info[cluster_id]["cells"].append(cell_id)

    for cluster_id, info in cluster_info.items():
        cluster_cells = info["cells"]
        if len(cluster_cells) == 1:
            cell = cluster_cells[0]
            info["shared_low"] = cell_low_variants[cell]
            info["shared_high"] = cell_high_variants[cell]
        else:
            shared_low = cell_low_variants[cluster_cells[0]].copy()
            shared_high = cell_high_variants[cluster_cells[0]].copy()
            for cell in cluster_cells[1:]:
                shared_low &= cell_low_variants[cell]
                shared_high &= cell_high_variants[cell]
            info["shared_low"] = shared_low
            info["shared_high"] = shared_high
        info["n_low"] = len(info["shared_low"])
        info["n_high"] = len(info["shared_high"])

    vprint(f"    Valid clusters: {len(cluster_info)}")

    # ── Step 3b: Intra-cluster coherence validation ──────────────────
    # Compute pairwise full-arm correlations within each cluster.
    # Clusters whose cells don't genuinely resemble each other across
    # the FULL arm (not just the LOH block) are likely noise.
    # NOTE: We use a soft threshold here (r≥0.30) because localized LOH
    # clones may span cells from different base genotypes that only share
    # the LOH region.  This check is meant to catch completely random
    # clusters, not enforce high genome-wide similarity.
    vprint("\n  Intra-cluster Coherence Validation (full-arm):")

    coherence_r_threshold = 0.30
    coherence_min_frac = 0.40
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

        pairwise_rs: list = []
        for ii in range(len(cell_indices)):
            for jj in range(ii + 1, len(cell_indices)):
                profile_i = X[cell_indices[ii], :]
                profile_j = X[cell_indices[jj], :]
                valid_mask = ~(np.isnan(profile_i) | np.isnan(profile_j))
                if valid_mask.sum() < 3:
                    continue
                pi = profile_i[valid_mask]
                pj = profile_j[valid_mask]
                # If either profile has zero variance (all values identical),
                # Pearson is undefined.  Check if profiles are similar:
                std_i = float(np.std(pi))
                std_j = float(np.std(pj))
                if std_i < 1e-6 or std_j < 1e-6:
                    # Near-identical profiles → perfect agreement
                    # Different constant profiles → no agreement
                    mad = float(np.mean(np.abs(pi - pj)))
                    pairwise_rs.append(1.0 if mad < 15.0 else 0.0)
                    continue
                r, _ = pearsonr(pi, pj)
                if np.isfinite(r):
                    pairwise_rs.append(r)

        if not pairwise_rs:
            vprint(f"    Cluster {cluster_id} ({n_cc} cells): no valid pairs → DROP")
            clusters_to_drop.add(cluster_id)
            continue

        # Filter any remaining NaN
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

        if frac_good < coherence_min_frac:
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

    if len(cell_to_cluster) == 0:
        vprint("    ⚠️  No valid clusters after coherence check")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Step 4: First-pass clone update ───────────────────────────────
    backup_clones = reference_clones.copy()
    barcodes = sample_barcodes.copy()

    sample_index_match = list(cell_to_cluster.keys())[0] in barcodes.values
    if sample_index_match:
        cell_to_barcode = {cell: cell for cell in df.index}
    else:
        if "barcode" in df.columns:
            cell_to_barcode = df["barcode"].to_dict()
        else:
            cell_to_barcode = dict(zip(df.index, barcodes))

    barcode_to_backup_clone = dict(zip(barcodes, backup_clones))

    # ── Step 5: Correlation analysis ──────────────────────────────────
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
        base_clone_to_clones[get_base_clone(clone_label)].append(clone_label)

    temp_loh_clones: dict = {}
    for cell_id, cluster_id in cell_to_cluster.items():
        if cell_id in cell_to_backup_clone:
            base_clone = cell_to_backup_clone[cell_id]
            temp_loh_clones[cell_id] = (
                f"{base_clone} LOH-C{cluster_id} {target_chr}"
            )

    temp_clone_to_cells: dict = defaultdict(list)
    for cell_id, temp_clone in temp_loh_clones.items():
        temp_clone_to_cells[temp_clone].append(cell_id)

    for temp_clone in temp_clone_to_cells:
        base_name = get_base_clone(temp_clone)
        if temp_clone not in base_clone_to_clones[base_name]:
            base_clone_to_clones[base_name].append(temp_clone)

    all_clone_to_cells: dict = defaultdict(list)
    all_clone_to_cells.update(clone_to_cells)
    all_clone_to_cells.update(temp_clone_to_cells)

    loh_clones = set(temp_clone_to_cells.keys())

    correlation_results: list = []
    additional_loh_assignments: dict = {}
    cluster_merges: dict = {}
    resolved_merges: dict = {}

    if loh_clones:
        vprint(f"    Computing mean profiles for {len(all_clone_to_cells)} clones...")

        clone_mean_profiles: dict = {}
        for clone_label, cells in all_clone_to_cells.items():
            cell_indices = [df.index.get_loc(c) for c in cells if c in df.index]
            if cell_indices:
                clone_mean_profiles[clone_label] = np.nanmean(
                    X[cell_indices, :], axis=0
                )

        vprint(
            f"      Progress: {len(all_clone_to_cells)}/{len(all_clone_to_cells)} clones - DONE"
        )

        # Cluster merges
        vprint("    Checking for cluster merges...")
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
                    if valid_mask.sum() < 10:
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

        if cluster_merges:
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

        # Expansion to non-LOH clones
        vprint("    Expanding to non-LOH clones...")
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
                    if valid_mask.sum() < 10:
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

        if additional_loh_assignments:
            vprint("    Updating cell assignments...")
            for cell_id, cluster_id in additional_loh_assignments.items():
                cell_to_cluster[cell_id] = cluster_id
            vprint("      Assignments updated")

        # Rebuild cluster info
        vprint("    Rebuilding cluster info...")
        cluster_info = defaultdict(
            lambda: {
                "cells": [],
                "n_low": 0,
                "n_high": 0,
                "shared_low": set(),
                "shared_high": set(),
            }
        )
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
                shared_high = cell_high_variants[
                    cluster_cells_with_loh[0]
                ].copy()
                for cell in cluster_cells_with_loh[1:]:
                    shared_low &= cell_low_variants[cell]
                    shared_high &= cell_high_variants[cell]
                info["shared_low"] = shared_low
                info["shared_high"] = shared_high
            info["n_low"] = len(info["shared_low"])
            info["n_high"] = len(info["shared_high"])

        vprint("      Cluster info rebuilt")

    df_correlations = (
        pd.DataFrame(correlation_results)
        if correlation_results
        else pd.DataFrame()
    )

    # ── Step 6: Optional size filtering ───────────────────────────────
    if not skip_size_filtering:
        vprint("\n  Pre-checking Final Clone Sizes:")

        backup_clones = reference_clones.copy()
        barcode_to_backup_clone = dict(zip(barcodes, backup_clones))

        final_clone_counts: dict = defaultdict(int)
        for cell_id, cluster_id in cell_to_cluster.items():
            if cell_id in cell_to_barcode:
                barcode = cell_to_barcode[cell_id]
                if barcode in barcode_to_backup_clone:
                    base_clone = barcode_to_backup_clone[barcode]
                    final_clone_label = (
                        f"{base_clone} LOH-C{cluster_id} {target_chr}"
                    )
                    final_clone_counts[final_clone_label] += 1

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
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        filtered_cell_to_cluster: dict = {}
        for cell_id, cluster_id in cell_to_cluster.items():
            if cell_id in cell_to_barcode:
                barcode = cell_to_barcode[cell_id]
                if barcode in barcode_to_backup_clone:
                    base_clone = barcode_to_backup_clone[barcode]
                    final_clone_label = (
                        f"{base_clone} LOH-C{cluster_id} {target_chr}"
                    )
                    if final_clone_label in valid_final_clones:
                        filtered_cell_to_cluster[cell_id] = cluster_id

        cell_to_cluster = filtered_cell_to_cluster
        vprint(f"    Filtered cells: {len(cell_to_cluster)}")

        cluster_info = defaultdict(
            lambda: {
                "cells": [],
                "n_low": 0,
                "n_high": 0,
                "shared_low": set(),
                "shared_high": set(),
            }
        )
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
                shared_high = cell_high_variants[
                    cluster_cells_with_loh[0]
                ].copy()
                for cell in cluster_cells_with_loh[1:]:
                    shared_low &= cell_low_variants[cell]
                    shared_high &= cell_high_variants[cell]
                info["shared_low"] = shared_low
                info["shared_high"] = shared_high
            info["n_low"] = len(info["shared_low"])
            info["n_high"] = len(info["shared_high"])
    else:
        vprint(
            "\n  Size filtering: DEFERRED (will be applied after cross-chromosome merging)"
        )

    # ── Step 7: Create output DataFrames ──────────────────────────────
    vprint("\n  Creating output DataFrames...")

    sorted_cells: list = []
    for cluster_id in sorted(cluster_info.keys()):
        sorted_cells.extend(cluster_info[cluster_id]["cells"])

    segment_records: list = []
    for cell_id in sorted_cells:
        cluster_id = cell_to_cluster[cell_id]
        info = cluster_info[cluster_id]
        if cell_id in cell_segments:
            for start_idx, end_idx in cell_segments[cell_id]:
                start_var = variants_chr_sorted[start_idx]
                end_var = variants_chr_sorted[end_idx]
                pos_start = extract_pos(start_var)
                pos_end = extract_pos(end_var)
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

    vprint("    DataFrames created")

    # ── Step 8: Final clone label update (accumulate) ─────────────────
    vprint("\n  Updating final clone labels...")

    current_clones = pd.Series(Sample.dna.row_attrs["Clone"]).copy()
    barcodes = pd.Series(Sample.dna.row_attrs["barcode"]).astype(str)

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

    loh_count = sum(
        is_loh_label(label) for label in Sample.dna.row_attrs["Clone"]
    )
    loh_for_this_chr = (
        pd.Series(Sample.dna.row_attrs["Clone"])
        .str.contains(target_chr, na=False)
        .sum()
    )
    vprint(f"    Total cells with any LOH: {loh_count}")
    vprint(f"    Cells with LOH for {target_chr}: {loh_for_this_chr}")

    # ── Step 9: Optional homogeneity check ────────────────────────────
    if skip_homogeneity_check:
        vprint("\n  Homogeneity Check: SKIPPED")
        vprint(
            f"  ✅ {target_chr}: {len(df_loh_only)} cells with LOH (no filtering)"
        )
        return df_loh_segments, df_loh_only, df_correlations, pd.DataFrame()

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

    final_clones_to_remove: set = set()
    final_homogeneity_stats: list = []

    vprint(
        f"    Checking {len(final_clone_to_cells)} final clones for {target_chr}..."
    )

    clone_idx = 0
    for clone_idx, (final_label, clone_cells_list) in enumerate(
        final_clone_to_cells.items(), 1
    ):
        n_cells = len(clone_cells_list)
        if clone_idx % 5 == 0:
            vprint(
                f"      Progress: {clone_idx}/{len(final_clone_to_cells)} clones"
            )

        if not skip_size_filtering and n_cells < min_final_clone_size:
            vprint(
                f"      {final_label}: {n_cells} cells < {min_final_clone_size} threshold - REMOVE"
            )
            final_clones_to_remove.add(final_label)
            final_homogeneity_stats.append(
                {
                    "final_clone": final_label,
                    "n_cells": n_cells,
                    "mean_correlation": np.nan,
                    "min_correlation": np.nan,
                    "n_correlations": 0,
                    "fraction_good_correlations": np.nan,
                    "status": "REMOVED_SIZE",
                }
            )
            continue

        if n_cells < 2:
            final_homogeneity_stats.append(
                {
                    "final_clone": final_label,
                    "n_cells": n_cells,
                    "mean_correlation": 1.0,
                    "min_correlation": 1.0,
                    "n_correlations": 0,
                    "fraction_good_correlations": 1.0,
                    "status": "KEPT",
                }
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
                if valid_mask.sum() < 10:
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

    if final_clone_to_cells and clone_idx % 5 != 0:
        vprint(
            f"      Progress: {len(final_clone_to_cells)}/{len(final_clone_to_cells)} clones - DONE"
        )

    df_final_homogeneity = pd.DataFrame(final_homogeneity_stats)

    if final_clones_to_remove:
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
        cell_to_final_clone_map: dict = {}
        for cell_id in df_loh_only.index:
            if cell_id in cell_to_barcode:
                barcode = cell_to_barcode[cell_id]
                if barcode in barcode_to_clone:
                    cell_to_final_clone_map[cell_id] = barcode_to_clone[
                        barcode
                    ]

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
    loh_count = sum(
        is_loh_label(label) for label in Sample.dna.row_attrs["Clone"]
    )
    loh_for_this_chr = (
        pd.Series(Sample.dna.row_attrs["Clone"])
        .str.contains(target_chr, na=False)
        .sum()
    )
    vprint(
        f"  Final: Total LOH cells = {loh_count}, {target_chr} LOH cells = {loh_for_this_chr}"
    )

    return df_loh_segments, df_loh_only, df_correlations, df_final_homogeneity