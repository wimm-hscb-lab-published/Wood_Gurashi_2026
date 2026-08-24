"""
LOH Pipeline – Top-level wrapper.

Re-exports the phase entry points and provides ``build_config(...)`` so
all notebook-facing parameters are defined in one place.
"""
from __future__ import annotations

import pandas as pd

from .loh_config import (
    LOHConfig,
    PipelineState,
)  # noqa: F401 re-export
from .detect_loh import detect_loh_and_update_clones       # noqa: F401 re-export
from .phase0_setup import run_phase0                       # noqa: F401 re-export
from .phase1 import run_phase1                             # noqa: F401 re-export
from .phase2 import run_phase2                             # noqa: F401 re-export
from .phase3 import run_phase3                             # noqa: F401 re-export
from .phase4 import run_phase4                             # noqa: F401 re-export
from .phase5 import run_phase5                             # noqa: F401 re-export
from .phase6_subclone_scan import run_post_phase5_subclone_scan
from .ado_probability import run_ado_probability_qc            # noqa: F401 re-export
from .plot_loh_clones import plot_loh_clones               # noqa: F401 re-export


def _final_assignment_summary(
    state: PipelineState,
    initial_cell_ids,
) -> pd.DataFrame:
    """Return and print the final initial-cell-to-assignment breakdown.

    The summary reads labels directly from ``Sample.dna`` rather than only
    ``state.clone_to_cells``. This deliberately retains ADO-labelled cells
    that were removed from ``state.df`` for downstream AF calculations.
    """
    from .loh_config import ADO_LABEL, is_small_label, clone_category

    initial_cell_ids = list(initial_cell_ids)
    current_labels = pd.Series(state.Sample.dna.row_attrs["Clone"]).astype(str)

    records = []
    for raw_cell_id in initial_cell_ids:
        cell_id = str(raw_cell_id)
        barcode = str(
            state.cell_to_barcode.get(
                raw_cell_id,
                state.cell_to_barcode.get(cell_id, cell_id),
            )
        )
        sample_idx = state.sample_idx_for_barcode(barcode)

        if sample_idx is None or sample_idx >= len(current_labels):
            label = ""
        else:
            label = str(current_labels.iloc[sample_idx]).strip()

        if not label or label.lower() in {"nan", "none"}:
            category = "unassigned"
            label = label or "<unassigned>"
        elif label == ADO_LABEL:
            category = "ADO"
        elif is_small_label(label):
            category = "small"
        else:
            category = clone_category(label)

        records.append(
            {
                "cell_id": cell_id,
                "barcode": barcode,
                "final_label": label,
                "category": category,
            }
        )

    assignments = pd.DataFrame(records)
    initial_n_cells = len(assignments)

    if assignments.empty:
        summary = pd.DataFrame(
            columns=["category", "final_label", "n_cells", "percent_initial"]
        )
    else:
        summary = (
            assignments
            .groupby(["category", "final_label"], dropna=False)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        summary["percent_initial"] = (
            100.0 * summary["n_cells"] / max(initial_n_cells, 1)
        )

    category_order = [
        "LOH",
        "subclonal_LOH",
        "possible_LOH",
        "non_LOH",
        "ADO",
        "small",
        "unassigned",
    ]
    summary["_category_order"] = pd.Categorical(
        summary.get("category"),
        categories=category_order,
        ordered=True,
    )
    summary = summary.sort_values(
        ["_category_order", "n_cells", "final_label"],
        ascending=[True, False, True],
        kind="stable",
    ).drop(columns="_category_order").reset_index(drop=True)

    category_counts = (
        summary.groupby("category")["n_cells"].sum().to_dict()
        if not summary.empty else {}
    )

    print("\n" + "=" * 72)
    print("FINAL CELL ASSIGNMENT SUMMARY")
    print("=" * 72)
    print(f"  Initial cells: {initial_n_cells}")
    print(f"  Final labelled cells: {int(summary['n_cells'].sum()) if not summary.empty else 0}")
    print(
        "  Category totals: "
        + " | ".join(
            f"{category}={int(category_counts.get(category, 0))}"
            for category in category_order
            if int(category_counts.get(category, 0)) > 0
        )
    )

    for category in category_order:
        category_rows = summary.loc[summary["category"] == category]
        if category_rows.empty:
            continue

        heading = {
            "LOH": "Active LOH assignments",
            "subclonal_LOH": "Unresolved subclonal LOH assignments",
            "possible_LOH": "Possible-LOH / retained FP assignments",
            "non_LOH": "Non-LOH assignments",
            "ADO": "ADO assignments",
            "small": "Small / pruned assignments",
            "unassigned": "Unassigned cells",
        }[category]

        print(f"\n  ── {heading} ──")
        for _, row in category_rows.iterrows():
            print(
                f"    {int(row['n_cells']):>6}  "
                f"({float(row['percent_initial']):5.1f}%)  "
                f"{row['final_label']}"
            )

    if not summary.empty and int(summary["n_cells"].sum()) != initial_n_cells:
        print("  ⚠ Final assignment count does not equal the initial input count.")

    print("=" * 72)
    return summary


def build_config(
    # ── Phase 1 detection thresholds ──
    low_allele_threshold: float = 10.0,
    high_allele_threshold: float = 90.0,
    min_contiguous_variants_initial: int = 4,
    min_extreme_fraction: float = 0.90,
    min_shared_variants: int = 4,
    min_cluster_size: int = 5,
    min_final_clone_size: int = 5,
    correlation_threshold: float = 0.80,
    detect_loh_verbose: bool = False,
    block_edge_tolerance: int = 1,

    # ── Phase 6 primary-child expansion / nested-arm preservation ──
    phase6_expand_child_members: bool = True,
    phase6_expand_min_corr_to_representative: float = 0.85,
    phase6_expand_max_median_abs_delta: float = 15.0,
    phase6_expand_min_block_extreme_fraction: float = 0.75,
    phase6_expand_min_signed_agreement: float = 0.75,
    phase6_expand_min_compared_variants: int = 4,
    phase6_expand_constant_profile_max_abs_delta: float = 15.0,
    phase6_secondary_arm_label_min_fraction: float = 0.90,

    # ── Phase 1 display ──
    phase1_show_top_clones: bool = True,
    phase1_show_failed_clones: bool = False,

    # ── Phase 1 localized / cytoband QC knobs ──
    phase1_localized_min_cells: int = 5,
    phase1_localized_min_mean_r: float = 0.85,
    phase1_localized_min_good_frac: float = 0.85,
    phase1_localized_good_cell_r: float = 0.85,
    phase1_localized_min_extreme_variants_passing_bands: int = 4,

    # ── Phase 1 clone-size gate ──
    phase1_min_clone_cells_all: int = 5,

    # ── Phase 1 extreme-AF variance QC ──
    phase1_extreme_sd_qc_enabled: bool = True,
    phase1_extreme_sd_qc_min_cells: int = 5,
    phase1_extreme_sd_qc_max_median_sd: float = 10.0,
    phase1_extreme_sd_qc_max_p90_sd: float = 15.0,
    phase1_sd_dynamic_factor: float = 1.2,

    # ── Phase 2 p/q merge ──
    post_merge_correlation_threshold: float = 0.80,

    # ── Phase 3 LOH clone QC ──
    skip_homogeneity_check: bool = False,
    min_homogeneity_fraction: float = 0.80,
    phase3_outlier_k: float = 2.0,
    phase3_outlier_small_clone_max_cells: int | None = 10,
    phase3_outlier_small_clone_abs_corr_floor: float = 0.80,
    phase3_min_variants_for_cell_corr: int = 4,
    phase3_min_mean_corr_to_mean: float = 0.80,
    min_variants_per_arm_for_corr: int = 4,

    # ── Directional consistency check (Phase 1 QC + Phase 3 homogeneity) ──
    directional_check_enabled: bool = True,
    directional_min_per_variant_agreement: float = 0.95,
    directional_min_block_pass_frac: float = 1.00,
    directional_min_extreme_cells_per_variant: int = 5,

    # ── Phase 3 ADO/dropout-aware directional tolerance ──
    phase3_allow_single_ado_amplicon_direction_fail: bool = True,
    phase3_direction_ado_min_block_variants: int = 6,
    phase3_direction_ado_max_bad_amplicons: int = 1,
    phase3_direction_ado_min_good_variants: int = 5,
    phase3_split_directional_mixed: bool = True,
    phase3_directional_split_min_cells: int | None = None,
    phase3_directional_split_min_discordant_variants: int = 1,

    # ── Phase 3a high-similarity multi-arm directional decomposition ──
    phase3_multi_arm_homogeneity_per_arm: bool = True,
    phase3_multi_arm_require_all_arms_pass: bool = True,
    phase3_multi_arm_min_block_variants: int = 4,
    phase3_multi_arm_partial_fail_use_sub_label: bool = True,

    # ── Phase 3b automatic nested parent-child LOH rescue ──
    phase3_nested_loh_rescue_enabled: bool = True,
    phase3_nested_loh_rescue_require_single_child_only_arm: bool = True,
    phase3_nested_loh_rescue_min_block_variants: int = 4,
    phase3_nested_loh_rescue_min_child_extreme_fraction: float = 0.85,
    phase3_nested_loh_rescue_min_shared_arm_r: float = 0.80,
    phase3_nested_loh_rescue_max_shared_arm_median_abs_diff: float = 15.0,
    phase3_nested_loh_rescue_min_shared_arm_extreme_match: float = 0.75,
    phase3_nested_loh_rescue_min_child_only_arm_r: float = 0.85,
    phase3_nested_loh_rescue_max_child_only_arm_median_abs_diff: float = 15.0,
    phase3_nested_loh_rescue_min_child_only_arm_extreme_match: float = 0.80,
    phase3_nested_loh_rescue_min_child_only_arm_called_variants: int = 4,
    phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr: float = 20.0,
    phase3_nested_loh_rescue_min_retained_parent_cells: int | None = None,

    # ── Phase 4 merge / rescue ──
    phase4_preserve_nested_loh_lineages: bool = True,

    # ── Phase 4d iterative cell reassignment ──
    phase4_iterate_cell_merge: bool = True,
    phase4_cell_merge_max_iters: int = 10,
    phase4_cell_merge_detect_cycles: bool = True,

    merge_min_extreme_frac: float = 0.85,
    phase4_percell_rescue: bool = True,
    phase4_percell_min_corr: float = 0.80,
    phase4_percell_min_extreme: float = 0.85,
    phase4_rescue_use_loh_block_variants: bool = True,
    phase4_rescue_expected_amplicon_ado: float = 0.10,
    phase4_rescue_max_ado_amplicons: int | None = 1,
    phase4_rescue_min_amplicons_to_allow_ado: int = 5,
    phase4_rescue_min_required_amplicons: int = 3,
    phase4_rescue_amplicon_variant_extreme_frac: float = 1.0,
    phase4_rescue_require_direction_match: bool = True,
    phase4_rescue_min_corr_variants: int = 4,
    phase4_rescue_constant_profiles_are_match: bool = False,
    phase4_rescue_constant_profile_max_abs_diff: float = 15.0,
    phase4_rescue_allow_amplicon_support_without_corr: bool = False,
    phase4_rescue_exclude_somatic_from_amp_support: bool = True,
    phase4_percell_trigger_mean_corr: float = 0.70,
    phase4_percell_trigger_any_arm_corr: float = 0.85,

    # ── Phase 4c broad rescue / second-pass expansion ──
    phase4_arm_rescue_enabled: bool = False,
    phase4_arm_rescue_min_cells: int = 5,
    phase4_arm_expansion_enabled: bool = False,
    phase4_arm_expansion_min_corr: float = 0.80,
    phase4_arm_expansion_min_extreme: float = 0.85,
    phase4_arm_expansion_unanimous_frac: float = 0.95,
    phase4_arm_expansion_min_cells: int = 8,

    # ── Phase 4b WT/Het → Hem/Hom somatic-amplicon ADO check ──
    ado_check_enabled: bool = True,
    ado_min_corr: float = 0.80,
    ado_max_median_abs_diff: float = 15.0,
    ado_min_comparison_variants: int = 20,
    ado_exclude_somatic_amplicon: bool = True,
    ado_include_hom_targets: bool = True,
    ado_wt_compare_all_hemhom_loh_targets: bool = True,
    # Backward-compatible, unused by new Phase 4b logic.
    ado_min_extreme_frac: float = 0.80,

    # ── Phase 4c cross-LOH clone similarity diagnostic ──
    phase4_cross_loh_similarity_enabled: bool = True,
    phase4_cross_loh_min_clone_size: int | None = None,
    phase4_cross_loh_min_shared_variants: int = 4,
    phase4_cross_loh_suggest_corr: float = 0.95,
    phase4_cross_loh_suggest_median_abs_delta: float = 10.0,
    phase4_cross_loh_suggest_extreme_frac: float = 0.85,
    phase4_cross_loh_min_block_overlap: float = 0.50,
    phase4_cross_loh_require_same_base: bool = False,
    phase4_cross_loh_constant_profiles_are_match: bool = True,

    # ── Phase 4d automatic small-LOH → larger-LOH merge ──
    phase4_merge_small_loh_after_annotation_enabled: bool = True,
    phase4_merge_small_loh_max_source_cells: int | None = None,
    phase4_merge_small_loh_min_target_cells: int | None = None,
    phase4_merge_small_loh_require_same_base: bool = True,
    phase4_merge_small_loh_require_source_arms_subset: bool = True,
    phase4_merge_small_loh_compare_all_target_arms: bool = True,
    phase4_merge_small_loh_loh_to_loh_require_larger_target: bool = True,
    phase4_merge_small_loh_loh_to_loh_require_arm_overlap: bool = True,
    phase4_merge_small_loh_min_shared_variants: int = 4,
    phase4_merge_small_loh_min_corr: float = 0.95,
    phase4_merge_small_loh_max_median_abs_delta: float = 10.0,
    phase4_merge_small_loh_min_extreme_frac: float = 0.85,
    phase4_merge_small_loh_min_block_overlap: float = 0.50,
    phase4_merge_small_loh_ambiguous_corr_delta: float = 0.02,
    phase4_merge_small_loh_constant_profiles_are_match: bool = True,
    phase4_merge_small_loh_include_base_non_loh_target: bool = False,
    phase4_merge_small_loh_include_base_non_loh_sources: bool = False,
    phase4_merge_small_loh_base_source_min_cells: int | None = None,
    phase4_merge_base_to_loh_min_corr: float = 0.80,
    phase4_merge_base_to_loh_max_median_abs_delta: float = 15.0,
    phase4_merge_base_to_loh_min_cell_extreme_frac: float = 0.70,
    phase4_merge_base_to_loh_min_target_extreme_frac: float = 0.85,
    phase4_merge_base_to_loh_require_direction_match: bool = True,
    phase4_merge_base_to_loh_min_matching_arms_frac: float = 1.00,
    phase4_merge_base_to_loh_allow_het_to_hem: bool = True,
    phase4_merge_small_loh_base_target_require_low_extreme: bool = False,
    phase4_merge_small_loh_base_target_max_cell_extreme_frac: float = 0.50,
    phase4_merge_small_loh_base_target_max_target_extreme_frac: float = 0.50,
    phase4_promote_fp_to_loh: bool = False,
    phase4_promote_fp_min_source_cells: int = 1,
    phase4_promote_fp_require_exact_arm_set: bool = True,

    # ── Phase 4e residual base-clone LOH discovery ──
    phase4_residual_base_loh_discovery_enabled: bool = False,
    phase4_residual_base_loh_min_candidate_cells: int = 5,
    phase4_residual_base_loh_min_contiguous_variants: int = 4,
    phase4_residual_base_loh_min_extreme_fraction: float = 0.90,
    phase4_residual_base_loh_min_shared_variants: int = 4,
    phase4_residual_base_loh_min_mean_corr: float = 0.80,
    phase4_residual_base_loh_max_cell_median_abs_diff: float = 15.0,

    # ── Amplicon validation ──
    validate_segment_length: bool = True,
    min_contiguous_variants_final: int = 4,
    min_different_amplicons: int = 3,

    # ── Somatic/non-informative variant block handling ──
    enforce_somatic_block_flanking: bool = True,
    min_informative_variants_flanking_somatic: int = 3,
    require_informative_on_both_sides_of_somatic: bool = True,

    # ── ADO-combination probability QC ──
    ado_probability_qc_enabled: bool = False,
    amplicon_ado_rates=None,
    ado_probability_rate_col: str = "smooth_ado_any_rate",
    ado_probability_amplicon_col: str = "amplicon_id",
    ado_probability_default_amplicon_rate: float = 0.01,
    ado_probability_required_amplicons: int | None = None,
    ado_probability_pvalue_threshold: float = 0.01,
    ado_probability_require_rates: bool = False,

    # ── Phase 5 QC alerting ──
    phase5_alert_three_amplicon_loh_blocks: bool = True,
    phase5_qc_alert_amplicons: int = 3,
    phase5_qc_alert_explain_ado_plausibility: bool = True,
    phase5_qc_alert_ado_rate_source: str = "empirical_background",
    phase5_qc_alert_ado_rate_per_amplicon: float = 0.10,
    phase5_qc_alert_empirical_min_background_cells: int = 50,
    phase5_qc_alert_empirical_use_dp: bool = True,
    phase5_qc_alert_empirical_min_depth: int = 10,
    phase5_qc_alert_empirical_amplicon_extreme_frac: float = 1.0,
    phase5_qc_alert_empirical_beta_alpha: float = 1.0,
    phase5_qc_alert_empirical_beta_beta: float = 9.0,
    phase5_qc_alert_ado_unlikely_pvalue: float = 0.01,

    # ── Plotting / references ──
    informative_variants: list | None = None,
    cytobands: "pd.DataFrame | None" = None,
    output_dir: str | None = None,
    save_pdf: bool = True,

    # ── Verbosity ──
    verbose: bool = True,

    # ── Diagnostic ──
    plot_dropped_clusters: bool = False,
) -> LOHConfig:
    """Build an ``LOHConfig`` from keyword arguments."""
    return LOHConfig(
        # Phase 1 core
        low_allele_threshold=low_allele_threshold,
        high_allele_threshold=high_allele_threshold,
        min_contiguous_variants_initial=min_contiguous_variants_initial,
        min_extreme_fraction=min_extreme_fraction,
        min_shared_variants=min_shared_variants,
        min_cluster_size=min_cluster_size,
        min_final_clone_size=min_final_clone_size,
        correlation_threshold=correlation_threshold,
        detect_loh_verbose=detect_loh_verbose,
        block_edge_tolerance=block_edge_tolerance,

        # Phase 6 primary-child expansion / nested-arm preservation
        phase6_expand_child_members=phase6_expand_child_members,
        phase6_expand_min_corr_to_representative=phase6_expand_min_corr_to_representative,
        phase6_expand_max_median_abs_delta=phase6_expand_max_median_abs_delta,
        phase6_expand_min_block_extreme_fraction=phase6_expand_min_block_extreme_fraction,
        phase6_expand_min_signed_agreement=phase6_expand_min_signed_agreement,
        phase6_expand_min_compared_variants=phase6_expand_min_compared_variants,
        phase6_expand_constant_profile_max_abs_delta=phase6_expand_constant_profile_max_abs_delta,
        phase6_secondary_arm_label_min_fraction=phase6_secondary_arm_label_min_fraction,

        # Phase 1 display
        phase1_show_top_clones=phase1_show_top_clones,
        phase1_show_failed_clones=phase1_show_failed_clones,

        # Phase 1 localized/cytoband QC
        phase1_localized_min_cells=phase1_localized_min_cells,
        phase1_localized_min_mean_r=phase1_localized_min_mean_r,
        phase1_localized_min_good_frac=phase1_localized_min_good_frac,
        phase1_localized_good_cell_r=phase1_localized_good_cell_r,
        phase1_localized_min_extreme_variants_passing_bands=phase1_localized_min_extreme_variants_passing_bands,

        # Phase 1 size gates
        phase1_min_clone_cells_all=phase1_min_clone_cells_all,

        # Phase 1 SD QC
        phase1_extreme_sd_qc_enabled=phase1_extreme_sd_qc_enabled,
        phase1_extreme_sd_qc_min_cells=phase1_extreme_sd_qc_min_cells,
        phase1_extreme_sd_qc_max_median_sd=phase1_extreme_sd_qc_max_median_sd,
        phase1_extreme_sd_qc_max_p90_sd=phase1_extreme_sd_qc_max_p90_sd,
        phase1_sd_dynamic_factor=phase1_sd_dynamic_factor,

        # Phase 2
        post_merge_correlation_threshold=post_merge_correlation_threshold,

        # Phase 3
        skip_homogeneity_check=skip_homogeneity_check,
        min_homogeneity_fraction=min_homogeneity_fraction,
        phase3_outlier_k=phase3_outlier_k,
        phase3_outlier_small_clone_max_cells=phase3_outlier_small_clone_max_cells,
        phase3_outlier_small_clone_abs_corr_floor=phase3_outlier_small_clone_abs_corr_floor,
        phase3_min_variants_for_cell_corr=phase3_min_variants_for_cell_corr,
        phase3_min_mean_corr_to_mean=phase3_min_mean_corr_to_mean,
        min_variants_per_arm_for_corr=min_variants_per_arm_for_corr,

        # Directional consistency
        directional_check_enabled=directional_check_enabled,
        directional_min_per_variant_agreement=directional_min_per_variant_agreement,
        directional_min_block_pass_frac=directional_min_block_pass_frac,
        directional_min_extreme_cells_per_variant=directional_min_extreme_cells_per_variant,

        # Phase 3 ADO/dropout-aware directional tolerance
        phase3_allow_single_ado_amplicon_direction_fail=phase3_allow_single_ado_amplicon_direction_fail,
        phase3_direction_ado_min_block_variants=phase3_direction_ado_min_block_variants,
        phase3_direction_ado_max_bad_amplicons=phase3_direction_ado_max_bad_amplicons,
        phase3_direction_ado_min_good_variants=phase3_direction_ado_min_good_variants,
        phase3_split_directional_mixed=phase3_split_directional_mixed,
        phase3_directional_split_min_cells=phase3_directional_split_min_cells,
        phase3_directional_split_min_discordant_variants=phase3_directional_split_min_discordant_variants,

        # Phase 3a high-similarity multi-arm directional decomposition
        phase3_multi_arm_homogeneity_per_arm=phase3_multi_arm_homogeneity_per_arm,
        phase3_multi_arm_require_all_arms_pass=phase3_multi_arm_require_all_arms_pass,
        phase3_multi_arm_min_block_variants=phase3_multi_arm_min_block_variants,
        phase3_multi_arm_partial_fail_use_sub_label=phase3_multi_arm_partial_fail_use_sub_label,

        # Phase 3b automatic nested parent-child LOH rescue
        phase3_nested_loh_rescue_enabled=phase3_nested_loh_rescue_enabled,
        phase3_nested_loh_rescue_require_single_child_only_arm=phase3_nested_loh_rescue_require_single_child_only_arm,
        phase3_nested_loh_rescue_min_block_variants=phase3_nested_loh_rescue_min_block_variants,
        phase3_nested_loh_rescue_min_child_extreme_fraction=phase3_nested_loh_rescue_min_child_extreme_fraction,
        phase3_nested_loh_rescue_min_shared_arm_r=phase3_nested_loh_rescue_min_shared_arm_r,
        phase3_nested_loh_rescue_max_shared_arm_median_abs_diff=phase3_nested_loh_rescue_max_shared_arm_median_abs_diff,
        phase3_nested_loh_rescue_min_shared_arm_extreme_match=phase3_nested_loh_rescue_min_shared_arm_extreme_match,
        phase3_nested_loh_rescue_min_child_only_arm_r=phase3_nested_loh_rescue_min_child_only_arm_r,
        phase3_nested_loh_rescue_max_child_only_arm_median_abs_diff=phase3_nested_loh_rescue_max_child_only_arm_median_abs_diff,
        phase3_nested_loh_rescue_min_child_only_arm_extreme_match=phase3_nested_loh_rescue_min_child_only_arm_extreme_match,
        phase3_nested_loh_rescue_min_child_only_arm_called_variants=phase3_nested_loh_rescue_min_child_only_arm_called_variants,
        phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr=phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr,
        phase3_nested_loh_rescue_min_retained_parent_cells=phase3_nested_loh_rescue_min_retained_parent_cells,

        # Phase 4
        phase4_preserve_nested_loh_lineages=phase4_preserve_nested_loh_lineages,
        phase4_iterate_cell_merge=phase4_iterate_cell_merge,
        phase4_cell_merge_max_iters=phase4_cell_merge_max_iters,
        phase4_cell_merge_detect_cycles=phase4_cell_merge_detect_cycles,
        merge_min_extreme_frac=merge_min_extreme_frac,
        phase4_percell_rescue=phase4_percell_rescue,
        phase4_percell_min_corr=phase4_percell_min_corr,
        phase4_percell_min_extreme=phase4_percell_min_extreme,
        phase4_rescue_use_loh_block_variants=phase4_rescue_use_loh_block_variants,
        phase4_rescue_expected_amplicon_ado=phase4_rescue_expected_amplicon_ado,
        phase4_rescue_max_ado_amplicons=phase4_rescue_max_ado_amplicons,
        phase4_rescue_min_required_amplicons=phase4_rescue_min_required_amplicons,
        phase4_rescue_amplicon_variant_extreme_frac=phase4_rescue_amplicon_variant_extreme_frac,
        phase4_rescue_require_direction_match=phase4_rescue_require_direction_match,
        phase4_rescue_min_corr_variants=phase4_rescue_min_corr_variants,
        phase4_rescue_constant_profiles_are_match=phase4_rescue_constant_profiles_are_match,
        phase4_rescue_constant_profile_max_abs_diff=phase4_rescue_constant_profile_max_abs_diff,
        phase4_rescue_allow_amplicon_support_without_corr=phase4_rescue_allow_amplicon_support_without_corr,
        phase4_rescue_exclude_somatic_from_amp_support=phase4_rescue_exclude_somatic_from_amp_support,
        phase4_rescue_min_amplicons_to_allow_ado=phase4_rescue_min_amplicons_to_allow_ado,
        phase4_percell_trigger_mean_corr=phase4_percell_trigger_mean_corr,
        phase4_percell_trigger_any_arm_corr=phase4_percell_trigger_any_arm_corr,
        phase4_arm_rescue_enabled=phase4_arm_rescue_enabled,
        phase4_arm_rescue_min_cells=phase4_arm_rescue_min_cells,
        phase4_arm_expansion_enabled=phase4_arm_expansion_enabled,
        phase4_arm_expansion_min_corr=phase4_arm_expansion_min_corr,
        phase4_arm_expansion_min_extreme=phase4_arm_expansion_min_extreme,
        phase4_arm_expansion_unanimous_frac=phase4_arm_expansion_unanimous_frac,
        phase4_arm_expansion_min_cells=phase4_arm_expansion_min_cells,

        # Phase 4b ADO/somatic-amplicon similarity check
        ado_check_enabled=ado_check_enabled,
        ado_min_corr=ado_min_corr,
        ado_max_median_abs_diff=ado_max_median_abs_diff,
        ado_min_comparison_variants=ado_min_comparison_variants,
        ado_exclude_somatic_amplicon=ado_exclude_somatic_amplicon,
        ado_include_hom_targets=ado_include_hom_targets,
        ado_wt_compare_all_hemhom_loh_targets=ado_wt_compare_all_hemhom_loh_targets,
        ado_min_extreme_frac=ado_min_extreme_frac,

        # Phase 4c cross-LOH diagnostic
        phase4_cross_loh_similarity_enabled=phase4_cross_loh_similarity_enabled,
        phase4_cross_loh_min_clone_size=phase4_cross_loh_min_clone_size,
        phase4_cross_loh_min_shared_variants=phase4_cross_loh_min_shared_variants,
        phase4_cross_loh_suggest_corr=phase4_cross_loh_suggest_corr,
        phase4_cross_loh_suggest_median_abs_delta=phase4_cross_loh_suggest_median_abs_delta,
        phase4_cross_loh_suggest_extreme_frac=phase4_cross_loh_suggest_extreme_frac,
        phase4_cross_loh_min_block_overlap=phase4_cross_loh_min_block_overlap,
        phase4_cross_loh_require_same_base=phase4_cross_loh_require_same_base,
        phase4_cross_loh_constant_profiles_are_match=phase4_cross_loh_constant_profiles_are_match,

        # Phase 4d automatic small-LOH merge
        phase4_merge_small_loh_after_annotation_enabled=phase4_merge_small_loh_after_annotation_enabled,
        phase4_merge_small_loh_max_source_cells=phase4_merge_small_loh_max_source_cells,
        phase4_merge_small_loh_min_target_cells=phase4_merge_small_loh_min_target_cells,
        phase4_merge_small_loh_require_same_base=phase4_merge_small_loh_require_same_base,
        phase4_merge_small_loh_require_source_arms_subset=phase4_merge_small_loh_require_source_arms_subset,
        phase4_merge_small_loh_compare_all_target_arms=phase4_merge_small_loh_compare_all_target_arms,
        phase4_merge_small_loh_loh_to_loh_require_larger_target=phase4_merge_small_loh_loh_to_loh_require_larger_target,
        phase4_merge_small_loh_loh_to_loh_require_arm_overlap=phase4_merge_small_loh_loh_to_loh_require_arm_overlap,
        phase4_merge_small_loh_min_shared_variants=phase4_merge_small_loh_min_shared_variants,
        phase4_merge_small_loh_min_corr=phase4_merge_small_loh_min_corr,
        phase4_merge_small_loh_max_median_abs_delta=phase4_merge_small_loh_max_median_abs_delta,
        phase4_merge_small_loh_min_extreme_frac=phase4_merge_small_loh_min_extreme_frac,
        phase4_merge_small_loh_min_block_overlap=phase4_merge_small_loh_min_block_overlap,
        phase4_merge_small_loh_ambiguous_corr_delta=phase4_merge_small_loh_ambiguous_corr_delta,
        phase4_merge_small_loh_constant_profiles_are_match=phase4_merge_small_loh_constant_profiles_are_match,
        phase4_merge_small_loh_include_base_non_loh_target=phase4_merge_small_loh_include_base_non_loh_target,
        phase4_merge_small_loh_include_base_non_loh_sources=phase4_merge_small_loh_include_base_non_loh_sources,
        phase4_merge_small_loh_base_source_min_cells=phase4_merge_small_loh_base_source_min_cells,
        phase4_merge_base_to_loh_min_corr=phase4_merge_base_to_loh_min_corr,
        phase4_merge_base_to_loh_max_median_abs_delta=phase4_merge_base_to_loh_max_median_abs_delta,
        phase4_merge_base_to_loh_min_cell_extreme_frac=phase4_merge_base_to_loh_min_cell_extreme_frac,
        phase4_merge_base_to_loh_min_target_extreme_frac=phase4_merge_base_to_loh_min_target_extreme_frac,
        phase4_merge_base_to_loh_require_direction_match=phase4_merge_base_to_loh_require_direction_match,
        phase4_merge_base_to_loh_min_matching_arms_frac=phase4_merge_base_to_loh_min_matching_arms_frac,
        phase4_merge_base_to_loh_allow_het_to_hem=phase4_merge_base_to_loh_allow_het_to_hem,
        phase4_merge_small_loh_base_target_require_low_extreme=phase4_merge_small_loh_base_target_require_low_extreme,
        phase4_merge_small_loh_base_target_max_cell_extreme_frac=phase4_merge_small_loh_base_target_max_cell_extreme_frac,
        phase4_merge_small_loh_base_target_max_target_extreme_frac=phase4_merge_small_loh_base_target_max_target_extreme_frac,
        phase4_promote_fp_to_loh=phase4_promote_fp_to_loh,
        phase4_promote_fp_min_source_cells=phase4_promote_fp_min_source_cells,
        phase4_promote_fp_require_exact_arm_set=phase4_promote_fp_require_exact_arm_set,

        # Phase 4e residual base-clone LOH discovery
        phase4_residual_base_loh_discovery_enabled=phase4_residual_base_loh_discovery_enabled,
        phase4_residual_base_loh_min_candidate_cells=phase4_residual_base_loh_min_candidate_cells,
        phase4_residual_base_loh_min_contiguous_variants=phase4_residual_base_loh_min_contiguous_variants,
        phase4_residual_base_loh_min_extreme_fraction=phase4_residual_base_loh_min_extreme_fraction,
        phase4_residual_base_loh_min_shared_variants=phase4_residual_base_loh_min_shared_variants,
        phase4_residual_base_loh_min_mean_corr=phase4_residual_base_loh_min_mean_corr,
        phase4_residual_base_loh_max_cell_median_abs_diff=phase4_residual_base_loh_max_cell_median_abs_diff,

        # Amplicon validation
        validate_segment_length=validate_segment_length,
        min_contiguous_variants_final=min_contiguous_variants_final,
        min_different_amplicons=min_different_amplicons,

        # Somatic/non-informative block handling
        enforce_somatic_block_flanking=enforce_somatic_block_flanking,
        min_informative_variants_flanking_somatic=min_informative_variants_flanking_somatic,
        require_informative_on_both_sides_of_somatic=require_informative_on_both_sides_of_somatic,

        # ADO-combination probability QC
        ado_probability_qc_enabled=ado_probability_qc_enabled,
        amplicon_ado_rates=amplicon_ado_rates,
        ado_probability_rate_col=ado_probability_rate_col,
        ado_probability_amplicon_col=ado_probability_amplicon_col,
        ado_probability_default_amplicon_rate=ado_probability_default_amplicon_rate,
        ado_probability_required_amplicons=ado_probability_required_amplicons,
        ado_probability_pvalue_threshold=ado_probability_pvalue_threshold,
        ado_probability_require_rates=ado_probability_require_rates,

        # Phase 5 QC alerting
        phase5_alert_three_amplicon_loh_blocks=phase5_alert_three_amplicon_loh_blocks,
        phase5_qc_alert_amplicons=phase5_qc_alert_amplicons,
        phase5_qc_alert_explain_ado_plausibility=phase5_qc_alert_explain_ado_plausibility,
        phase5_qc_alert_ado_rate_source=phase5_qc_alert_ado_rate_source,
        phase5_qc_alert_ado_rate_per_amplicon=phase5_qc_alert_ado_rate_per_amplicon,
        phase5_qc_alert_empirical_min_background_cells=phase5_qc_alert_empirical_min_background_cells,
        phase5_qc_alert_empirical_use_dp=phase5_qc_alert_empirical_use_dp,
        phase5_qc_alert_empirical_min_depth=phase5_qc_alert_empirical_min_depth,
        phase5_qc_alert_empirical_amplicon_extreme_frac=phase5_qc_alert_empirical_amplicon_extreme_frac,
        phase5_qc_alert_empirical_beta_alpha=phase5_qc_alert_empirical_beta_alpha,
        phase5_qc_alert_empirical_beta_beta=phase5_qc_alert_empirical_beta_beta,
        phase5_qc_alert_ado_unlikely_pvalue=phase5_qc_alert_ado_unlikely_pvalue,

        # Plotting
        informative_variants=informative_variants,
        cytobands=cytobands,
        output_dir=output_dir,
        save_pdf=save_pdf,

        # Verbose
        verbose=verbose,

        # Diagnostic
        plot_dropped_clusters=plot_dropped_clusters,
    )


def _print_state(
    label: str,
    state: PipelineState,
    *,
    results: dict | None = None,
    summary_df: pd.DataFrame | None = None,
) -> None:
    print(f"\n✅ {label} complete.")
    if results is not None:
        print(f"   Result keys: {sorted(results.keys())}")
    print(f"   Active LOH cells:   {state.loh_cell_count()}")
    print(f"   Active LOH labels:  {state.loh_clone_labels()}")
    from .loh_config import is_fp_label, is_sub_label
    fp_counts = state.current_clones.astype(str).loc[
        state.current_clones.astype(str).apply(is_fp_label)
    ].value_counts()
    sub_counts = state.current_clones.astype(str).loc[
        state.current_clones.astype(str).apply(is_sub_label)
    ].value_counts()
    print(f"   Unresolved SUB:      {len(sub_counts)} clone(s), {int(sub_counts.sum())} cell(s)")
    print(f"   Possible-LOH (FP):  {len(fp_counts)} clone(s), {int(fp_counts.sum())} cell(s)")
    if summary_df is not None:
        print(f"   Clones summarised: {len(summary_df)}")
        if "is_loh" in summary_df.columns:
            print(f"   LOH clones:        {int(summary_df['is_loh'].sum())}")


def run(
    *,
    Sample,
    df: pd.DataFrame,
    grouped_by_chr_arm: dict,
    cfg: LOHConfig,
    df_dp: pd.DataFrame | None = None,
    chr_arms: list[str] | None = None,
    detect_loh_fn=detect_loh_and_update_clones,
    run_plotting: bool = True,
    informative_variants: list | None = None,
    cytobands: "pd.DataFrame | None" = None,
    output_dir: str | None = None,
    output_prefix: str = "All_clones_AF",
    include_non_loh: bool = True,
    include_possible_loh: bool = True,
    plot_separate_clones: bool = True,
    split_by_cytobands: bool = False,
    save_figures: bool = True,
    save_pdf: bool | None = None,
    visualise_fn=None,
    print_progress: bool = True,
    # Optional targeted refinement within the same run: scan final Phase-5
    # parent labels for additional LOH children before final accounting/plots.
    post_phase5_subclone_scan_labels: list[str] | str | None = None,
    post_phase5_subclone_scan_chr_arms: list[str] | None = None,
    post_phase5_subclone_scan_min_parent_cells: int | None = None,
    post_phase5_subclone_scan_merge_back: bool = True,
    post_phase5_subclone_scan_auto_exclude_labels: list[str] | None = None,
    post_phase5_subclone_scan_auto_include_wt: bool = False,
) -> dict:
    """
    Run the full LOH pipeline (Phase 0→5, plus optional final-label refinement)
    with optional plotting.

    Parameters mirror the notebook workflow while providing a single
    callable entry point. ``include_possible_loh`` retains QC-rejected
    ``FP: ...`` clones in the plotting dataframe as a third category.
    Pass df_dp when available so Phase 5 empirical
    ADO-like QC is coverage-aware (DP < min_depth is missingness, not ADO).

    ``post_phase5_subclone_scan_labels`` optionally activates the final-label
    subclone scan *inside this run*. The scan freezes the Phase-5 labels,
    searches the requested non-LOH parent clone(s), validates any LOH children,
    optionally merges newly split children back into larger compatible clones,
    refreshes the Phase-5 summary, and then proceeds directly to final assignment
    accounting and plotting. It never reopens active LOH/SUB/FP/small labels.

    Pass ``post_phase5_subclone_scan_labels="auto"`` to scan eligible non-WT
    parents that still retain a possible LOH block (a per-cell screen, so
    subclonal blocks present in only a fraction of a parent are caught), or
    ``"all"`` to scan eligible non-WT parents regardless of the screen. The exact
    dominant ``"WT"`` label is excluded by default; set
    ``post_phase5_subclone_scan_auto_include_wt=True`` only for an intentional
    WT-focused investigation. For focused analyses, pass an explicit list such
    as ``["TP53_R248W_Hem"]``. With
    ``post_phase5_subclone_scan_merge_back=True`` (default), strict consolidation
    reabsorbs a smaller same-base child into a larger compatible same-arm sibling
    after all normal profile, delta, direction, and block-overlap gates pass.
    Phase 6 also retains a strict primary LOH family when only a subset carries
    an additional arm: the nested arm is recorded in row metadata rather than
    splitting the primary family into separately pruned flat labels.

    Returns
    -------
    dict
        {
            "state": PipelineState,
            "phase1_results": dict,
            "phase2_results": dict,
            "phase3_results": dict,
            "phase4_results": dict,
            "phase6_results": dict,
            "phase7_summary_df": pd.DataFrame | None,
            "summary_df": pd.DataFrame,
            "df_plot_all": pd.DataFrame | None,
        }
    """
    # Phase 0 — pre-checks and state construction
    state = run_phase0(
        Sample=Sample,
        df=df,
        grouped_by_chr_arm=grouped_by_chr_arm,
        chr_arms=chr_arms,
        cfg=cfg,
        df_dp=df_dp,
    )
    if state is None:
        raise RuntimeError("Phase 0 returned None — no arms to process.")

    # Immutable denominator for the final assignment accounting. This must be
    # captured before Phase 4b may remove ADO-labelled cells from state.df.
    initial_cell_ids = list(state.df.index)

    if print_progress:
        print("\n✅ Phase 0 complete.")
        print(f"   Arms to process: {len(state.chr_arms)}")
        print(f"   Variant columns: {len(state.variant_cols)}")
        print(f"   Cells in df:     {len(state.df)}")

    # Phase 1 — per-arm LOH detection
    phase1_results = run_phase1(state=state, detect_loh_fn=detect_loh_fn)
    if print_progress:
        _print_state("Phase 1", state, results=phase1_results)

    # Phase 2 — p/q arm integration is intentionally disabled for this
    # workflow. Keeping an explicit Phase-2 log makes downstream numbering
    # unambiguous: Phase 3 always follows Phase 2, rather than Phase 1.
    phase2_results = {"status": "SKIPPED", "reason": "p_q_arm_merging_disabled"}
    if print_progress:
        print("\n" + "=" * 60)
        print("PHASE 2: P/Q ARM INTEGRATION (SKIPPED)")
        print("=" * 60)
        print("  Arm calls remain independent; no p/q merging is applied.")

    # Phase 3 — clone consolidation
    phase3_results = run_phase3(state)
    if print_progress:
        _print_state("Phase 3", state, results=phase3_results)

    # Phase 4 — non-LOH ↔ LOH correlation/merge
    phase4_results = run_phase4(state)
    if print_progress:
        _print_state("Phase 4", state, results=phase4_results)

    # Phase 5 — LOH clone summary and HEM-TP53 warning.
    # ADO-combination probability QC has been removed from Phase 5.
    summary_df = run_phase5(state)
    ado_probability_qc = pd.DataFrame()
    if print_progress:
        _print_state("Phase 5", state, summary_df=summary_df)

    # Optional Phase 6 — targeted refinement of the *final Phase-5* parent
    # annotations. This is deliberately inside run() so final accounting and
    # plots automatically reflect discovered children.
    phase6_results = {}
    if post_phase5_subclone_scan_labels is not None:
        phase6_results = run_post_phase5_subclone_scan(
            state=state,
            clone_labels=post_phase5_subclone_scan_labels,
            chr_arms=post_phase5_subclone_scan_chr_arms,
            min_parent_cells=post_phase5_subclone_scan_min_parent_cells,
            merge_back_after_scan=post_phase5_subclone_scan_merge_back,
            auto_exclude_labels=post_phase5_subclone_scan_auto_exclude_labels,
            auto_include_wt=post_phase5_subclone_scan_auto_include_wt,
            # Keep phase ordering explicit in the wrapper: Phase 6 finishes
            # first; the refreshed post-refinement clone summary is Phase 7.
            rerun_phase5_summary=False,
            verbose=print_progress,
        )
        if print_progress:
            _print_state("Phase 6", state, results=phase6_results)

        summary_df = run_phase5(
            state,
            heading="PHASE 7: FINAL LOH CLONE SUMMARY AFTER PHASE 6",
        )
        phase6_results["summary_df"] = summary_df
        state.results["phase7_final_summary"] = summary_df
        if print_progress:
            _print_state("Phase 7", state, summary_df=summary_df)

    # Final accounting is deliberately emitted after every reassignment/pruning
    # phase and immediately before plotting.
    final_assignment_summary = _final_assignment_summary(
        state=state,
        initial_cell_ids=initial_cell_ids,
    )
    state.results["final_assignment_summary"] = final_assignment_summary

    # Plot LOH clones
    df_plot_all = None
    if run_plotting:
        df_plot_all = plot_loh_clones(
            state,
            informative_variants=informative_variants,
            cytobands=cytobands,
            output_dir=output_dir,
            output_prefix=output_prefix,
            include_non_loh=include_non_loh,
            include_possible_loh=include_possible_loh,
            plot_separate_clones=plot_separate_clones,
            split_by_cytobands=split_by_cytobands,
            save_figures=save_figures,
            save_pdf=save_pdf,
            visualise_fn=visualise_fn,
        )
        if print_progress:
            print("\n✅ Plotting complete.")

    return {
        "state": state,
        "phase1_results": phase1_results,
        "phase2_results": phase2_results,
        "phase3_results": phase3_results,
        "phase4_results": phase4_results,
        "phase6_results": phase6_results,
        "phase7_summary_df": state.results.get("phase7_final_summary"),
        "ado_probability_qc": ado_probability_qc,
        "summary_df": summary_df,
        "final_assignment_summary": final_assignment_summary,
        "df_plot_all": df_plot_all,
    }
