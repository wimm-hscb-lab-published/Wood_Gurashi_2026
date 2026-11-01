"""DNA_Utils: single-cell DNA genotype processing for Tapestri / Mosaic samples.

Modules
-------
utils           Variant-ID parsing, assay helpers, Varsome annotation-cache maintenance, matrix export.
qc_filters      Cell QC: doublet scoring and calls, low-quality (amplicon-loss) cells, AF heatmaps.
filter_outliers Removal of outlier cells for somatic variants.
germline        Germline heterozygous variant detection, genotype profiles, CNV/BAF audit pipelines.
somatic         Somatic variant selection and plots.
clones          Clone tables, VAF and clone sizes.
plot_clones     Clone UMAPs and somatic-variant plots.

Everything listed in ``__all__`` is importable directly from ``DNA_Utils``. See README.md for details.
"""

# ── Utilities ─────────────────────────────────────────────────────────────────
from .utils import (
    extract_chr,
    extract_chr_num,
    extract_position,
    chrom_key,
    sanitize_row_attrs,
    drop_scalar_row_attrs,

    # Varsome annotation cache maintenance
    VARSOME_DB_COLUMNS,
    VARSOME_N_FIELDS,
    VARSOME_VALID_GENOMES,
    varsome_db_path,
    varsome_db_lock,
    repair_varsome_db,
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
    fast_af_heatmap,
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
    apply_calibrated_germline_rules,

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

    # varsome cache maintenance
    "VARSOME_DB_COLUMNS",
    "VARSOME_N_FIELDS",
    "VARSOME_VALID_GENOMES",
    "varsome_db_path",
    "varsome_db_lock",
    "repair_varsome_db",
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
    "fast_af_heatmap",

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
    "apply_calibrated_germline_rules",

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
    "clones_to_dataframe",
    "get_VAF",
    "get_clones_size",

    # plots
    "plot_individual_clone_umaps",
    "plot_somatic_variants",

    # outlier filtering
    "filter_somatic_variant_outliers",
]