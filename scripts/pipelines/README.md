# pipelines

Python modules imported by the notebooks in `notebooks/`. They run inside the `mosaic.sif` container.

The **Notebook** column shows which notebook step calls each function:

- **01:** `notebooks/01_preprocessing/01_Preprocessing_<experiment>.ipynb`
- **03:** `notebooks/03_sample_analysis/03_Sample_Analysis_<experiment>_<donor>.ipynb`

The `02_protein_filtering` notebooks don't call these modules. Modules with no functions listed are used internally by other modules.

## DNA_Utils: single-cell genotype processing

| Module | Purpose | Functions called | Notebook |
|--------|---------|------------------|----------|
| `qc_filters.py` | Cell-level QC: amplicon loss, spurious gain, doublet-like AF shifts | `apply_low_quality_qc`, `define_doublet_cutoff_per_donor`, `apply_doublet_calls_with_corr_reassign`, `fast_af_heatmap` | 01 |
| `germline.py` | Germline variant discovery, blacklist filtering, AF clustering and visualisation | `get_empirical_cohort_germline_thresholds`, `run_empirical_all_variant_germline_export`, `run_step2_barcode_cnv_audit_pipeline`, `run_af_clustering_only`, `rename_af_clusters_and_update_filtered_germline`, `add_wt_concordance_and_finalize_germline`, `visualize_found_germline_variants` | 03 |
| `somatic.py` | Somatic variant discovery (correlation + Wilcoxon) and heatmaps | `find_somatic_variants`, `visualize_found_differential_variants`, `visualize_found_informative_variants` | 03 |
| `filter_outliers.py` | Somatic variant outlier filtering within carrier cells | `filter_somatic_variant_outliers` | 03 |
| `clones.py` | Clone tables, VAF and clone sizes | `clones_to_dataframe`, `get_VAF`, `get_clones_size` | 03 |
| `plot_clones.py` | UMAPs of individual clones and somatic variants | `plot_individual_clone_umaps`, `plot_somatic_variants` | 03 |
| `utils.py` | Shared helpers (VarSome cache, exports) | `inspect_varsome_db`, `repair_varsome_db`, `export_cell_subset_matrices`, `extract_chr_num` | 03 |

## CNV_Utils: copy number analysis

| Module | Purpose | Functions called | Notebook |
|--------|---------|------------------|----------|
| `cnv_extraction.py` | Per-amplicon read depth extraction for CNV calling | `extract_cnv_data` | 03 |
| `informativeness.py` | Testing which heterozygous variants are informative for copy number/allelic imbalance | `test_variant_informativeness`, `expand_informative_variants` | 03 |
| `variant_grouping.py` | Grouping of informative variants by chromosome arm | `group_informative_variants_by_chr_arm` | 03 |
| `explore_gene_cnv.py` | Gene-level ploidy plots using RefSeq exons | `load_refseq_gtf_exons`, `prepare_ploidy_region_plot_inputs` | 03 |
| `visualization.py` | Integrated CNV/BAF plots per clone | `plot_integrated_cnv_baf_with_significance`, `visualise_variants_by_clone` | 03 |
| `cytoband_mapping.py` | Mapping of amplicons to cytobands | | |

## LOH_Pipeline: loss of heterozygosity detection and clone refinement

Run from the 03 notebooks through `LOH_Pipeline.build_config` and `LOH_Pipeline.run` (in `loh_wrapper.py`), which execute the phases below. See [`LOH_Pipeline/README.md`](LOH_Pipeline/README.md) for full documentation and [`LOH_Pipeline/VALIDATION.md`](LOH_Pipeline/VALIDATION.md) for validation.

| Module | Purpose |
|--------|---------|
| `loh_wrapper.py` | Top-level entry point (`build_config`, `run`) |
| `loh_config.py` | Configuration, shared state and helpers (`LOH_TAG` imported in 03) |
| `phase0_setup.py` | Phase 0: baseline clone labels and variant availability check |
| `phase1.py` | Phase 1: per-arm LOH detection |
| `phase2.py` | Phase 2: p/q arm merging |
| `phase3.py` | Phase 3: LOH clone QC (outlier removal, homogeneity, rescue, reconciliation) |
| `phase4.py` | Phase 4: non-LOH ↔ LOH clone correlation and merge |
| `phase5.py` | Phase 5: LOH clone summary and haemizygous TP53 warning |
| `phase6_subclone_scan.py` | Phase 6: targeted LOH subclone discovery |
| `detect_loh.py` | Core LOH detection function |
| `ado_probability.py` | Allelic dropout combination probability scoring |
| `plot_loh_clones.py` | LOH clone plots |

## PRO_Utils: protein (ADT) analysis

| Module | Purpose | Functions called | Notebook |
|--------|---------|------------------|----------|
| `update_labels.py` | Cell-type label updates | `update_protein_labels_from_dataframe` | 03 |
| `plot_umap.py` | Protein UMAPs | `plot_protein_umap` | 03 |
| `homogeneity_silhouette.py` | Cluster homogeneity and silhouette scores | `plot_protein_homogeneity_silhouette` | 03 |
| `hto_calling.py` | Single-hashtag HTO present/absent calling | `call_hto_present_absent` | 03 (MDS011 A/B, pMDS001 only) |
| `plot_umap_per_label.py` | Per-label UMAPs | | |
| `colors.py` | Colour maps for protein UMAPs | | |