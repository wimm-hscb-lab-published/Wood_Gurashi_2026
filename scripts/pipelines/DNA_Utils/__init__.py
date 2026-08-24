"""DNA_pipeline – DNA variant analysis utilities.

v3 initializer exposing the notebook-facing germline utilities.
Copy this file to the package as ``__init__.py`` together with the updated
``germline.py`` module. The downloadable module may be named ``germline.py``;
inside the package it should be copied/renamed to ``germline.py`` unless you
also change the import below to ``from .germline import ...``.
"""

# ── Utilities ─────────────────────────────────────────────────────────────────
from .utils import (
    extract_chr,
    extract_chr_num,
    extract_position,
    chrom_key,
    sanitize_row_attrs,
    drop_scalar_row_attrs,
    heatmap_shape,
    binomial_pval_right_tail,

    # Varsome annotation cache maintenance
    VARSOME_DB_COLUMNS,
    VARSOME_N_FIELDS,
    VARSOME_VALID_GENOMES,
    varsome_db_path,
    varsome_db_lock,
    inspect_varsome_db,
    repair_varsome_db,
    reset_varsome_db,
    annotate_with_repair,

    # Annotation / subsetting / metadata / matrix export
    annotate_assay_in_chunks,
    subset_assay_to_cells,
    variant_metadata_frame,
    export_assay_layers,
    export_cell_subset_matrices,
)

# ── QC filters ────────────────────────────────────────────────────────────────
from .qc_filters import (
    flag_amplicon_loss_cells,
    flag_spurious_gain_cells,
    piecewise_elbow_threshold,
    flag_doublet_change_cells,
    define_doublet_cutoff_per_donor,
    apply_doublet_calls_with_corr_reassign,
    apply_low_quality_qc,
    doublets_example,
    fast_af_heatmap,
    find_remaining_cells_like_mixed,
    visualise_mixed_like_rescue,
)

# ── Germline / CNV marker workflow ────────────────────────────────────────────
from .germline import (
    # Original germline helpers
    get_real_somatic_variants,
    find_germline_variants,
    visualize_found_germline_variants,
    validate_germline_by_af,
    review_failed_variants,
    genotype_af_cutoffs,
    compute_genotype_structure,
    variant_level_ngt_af_dp_profile,
    build_and_save_all_variant_germline_profile,
    run_all_variant_germline_profile_export,
    build_wt_cell_subset,
    population_structure_table,
    add_population_concordance,

    # Cohort calibration helpers
    resolve_calibration_columns,
    load_germline_cohort,
    germline_recurrence,
    germline_anchor_frame,
    germline_percentile_table,
    propose_germline_thresholds,
    evaluate_min_frac_het,
    germline_practical_anchor_frame,
    calibrate_practical_germline_thresholds,
    apply_calibrated_germline_rules,
    germline_counts_per_sample,
    calibrate_germline_thresholds,

    # Notebook-facing v2/v3 wrappers
    get_empirical_cohort_germline_thresholds,
    run_empirical_all_variant_germline_export,
    add_wt_concordance_and_finalize_germline,
    build_all_variant_matrices_from_step2_clone_labels,
    run_all_variant_cnv_baf_audit_pipeline,
    run_step2_barcode_cnv_audit_pipeline,
    check_significant_variants_against_all_variant_germline_df,
    apply_final_no_loss_cnv_marker_rules,
    run_af_clustering_only,
    rename_af_clusters_and_update_filtered_germline,

    # Parameter dictionaries
    FINAL_NO_LOSS_CNV_MARKER_PARAMETERS,
    AF_CLUSTER_ONLY_DEFAULT_PARAMS,
    DEFAULT_AF_CLUSTER_RENAME_MAP,
)

# ── Somatic ───────────────────────────────────────────────────────────────────
from .somatic import (
    find_somatic_variants,
    visualize_found_differential_variants,
    visualize_found_informative_variants,
)

# ── Clones ────────────────────────────────────────────────────────────────────
from .clones import (
    assign_patients_from_whitelist,
    clones_to_dataframe,
    get_VAF,
    get_clones_size,
)

# ── Clone / variant plots ────────────────────────────────────────────────────
from .plot_clones import (
    plot_individual_clone_umaps,
    plot_somatic_variants,
)

# ── Outlier filtering ────────────────────────────────────────────────────────
from .filter_outliers import (
    filter_somatic_variant_outliers,
)

__all__ = [
    # utils
    "extract_chr",
    "extract_chr_num",
    "extract_position",
    "chrom_key",
    "sanitize_row_attrs",
    "drop_scalar_row_attrs",
    "heatmap_shape",
    "binomial_pval_right_tail",

    # varsome cache maintenance
    "VARSOME_DB_COLUMNS",
    "VARSOME_N_FIELDS",
    "VARSOME_VALID_GENOMES",
    "varsome_db_path",
    "varsome_db_lock",
    "inspect_varsome_db",
    "repair_varsome_db",
    "reset_varsome_db",
    "annotate_with_repair",

    # annotation / subsetting / metadata / export
    "annotate_assay_in_chunks",
    "subset_assay_to_cells",
    "variant_metadata_frame",
    "export_assay_layers",
    "export_cell_subset_matrices",

    # qc
    "flag_amplicon_loss_cells",
    "flag_spurious_gain_cells",
    "piecewise_elbow_threshold",
    "flag_doublet_change_cells",
    "define_doublet_cutoff_per_donor",
    "apply_doublet_calls_with_corr_reassign",
    "apply_low_quality_qc",
    "doublets_example",
    "fast_af_heatmap",
    "find_remaining_cells_like_mixed",
    "visualise_mixed_like_rescue",

    # original germline helpers
    "get_real_somatic_variants",
    "find_germline_variants",
    "visualize_found_germline_variants",
    "validate_germline_by_af",
    "review_failed_variants",
    "genotype_af_cutoffs",
    "compute_genotype_structure",
    "variant_level_ngt_af_dp_profile",
    "build_and_save_all_variant_germline_profile",
    "run_all_variant_germline_profile_export",
    "build_wt_cell_subset",
    "population_structure_table",
    "add_population_concordance",

    # cohort calibration helpers
    "resolve_calibration_columns",
    "load_germline_cohort",
    "germline_recurrence",
    "germline_anchor_frame",
    "germline_percentile_table",
    "propose_germline_thresholds",
    "evaluate_min_frac_het",
    "germline_practical_anchor_frame",
    "calibrate_practical_germline_thresholds",
    "apply_calibrated_germline_rules",
    "germline_counts_per_sample",
    "calibrate_germline_thresholds",

    # v2/v3 notebook-facing wrappers
    "get_empirical_cohort_germline_thresholds",
    "run_empirical_all_variant_germline_export",
    "add_wt_concordance_and_finalize_germline",
    "build_all_variant_matrices_from_step2_clone_labels",
    "run_all_variant_cnv_baf_audit_pipeline",
    "run_step2_barcode_cnv_audit_pipeline",
    "check_significant_variants_against_all_variant_germline_df",
    "apply_final_no_loss_cnv_marker_rules",
    "run_af_clustering_only",
    "rename_af_clusters_and_update_filtered_germline",

    # v2/v3 parameter dictionaries
    "FINAL_NO_LOSS_CNV_MARKER_PARAMETERS",
    "AF_CLUSTER_ONLY_DEFAULT_PARAMS",
    "DEFAULT_AF_CLUSTER_RENAME_MAP",

    # somatic
    "find_somatic_variants",
    "visualize_found_differential_variants",
    "visualize_found_informative_variants",

    # clones
    "assign_patients_from_whitelist",
    "clones_to_dataframe",
    "get_VAF",
    "get_clones_size",

    # plots
    "plot_individual_clone_umaps",
    "plot_somatic_variants",

    # outlier filtering
    "filter_somatic_variant_outliers",
]