# scripts

Code used for the analyses in the manuscript, split into reusable pipeline modules (Python) and figure scripts (R Markdown).

```
scripts/
├── pipelines/      # Custom scripts called in the notebooks
│   ├── DNA_Utils/
│   ├── CNV_Utils/
│   ├── LOH_Pipeline/
│   └── PRO_Utils/
└── manuscript/     # R Markdown figure scripts (step 5)
```

## pipelines/

Python modules run inside the `mosaic.sif` container and imported by the notebooks in `notebooks/`.

### DNA_Utils: single-cell genotype processing

| Module | Purpose |
|--------|---------|
| `qc_filters.py` | Cell and variant QC filters |
| `filter_outliers.py` | Removal of outlier cells |
| `germline.py` | Identification of germline heterozygous variants |
| `somatic.py` | Somatic variant selection |
| `clones.py` | Clonotype calling from single-cell genotypes |
| `plot_clones.py` | Clone plots |
| `utils.py` | Shared helpers |

### CNV_Utils: copy number analysis

| Module | Purpose |
|--------|---------|
| `cnv_extraction.py` | Extraction of per-amplicon read depth for CNV calling |
| `cytoband_mapping.py` | Mapping of amplicons to cytobands |
| `variant_grouping.py` | Grouping of variants by genomic region |
| `informativeness.py` | Selection of informative regions |
| `explore_gene_cnv.py` | Per-gene CNV exploration |
| `visualization.py` | CNV plots |

### LOH_Pipeline: loss of heterozygosity detection

Detects LOH in single cells from germline heterozygous variants, in sequential phases. See [`pipelines/LOH_Pipeline/README.md`](pipelines/LOH_Pipeline/README.md) for full documentation and [`VALIDATION.md`](pipelines/LOH_Pipeline/VALIDATION.md) for validation.

| Module | Purpose |
|--------|---------|
| `loh_config.py` | Parameters and thresholds |
| `loh_wrapper.py` | Entry point that runs all phases |
| `phase0_setup.py` – `phase5.py` | Pipeline phases |
| `phase6_subclone_scan.py` | Subclonal LOH scan |
| `ado_probability.py` | Allelic dropout probability model |
| `detect_loh.py` | LOH calling |
| `plot_loh_clones.py` | LOH clone plots |

### PRO_Utils: protein (ADT) analysis

| Module | Purpose |
|--------|---------|
| `hto_calling.py` | Hashtag (HTO) calling |
| `update_labels.py` | Cell-type label updates |
| `homogeneity_silhouette.py` | Cluster homogeneity and silhouette scores |
| `plot_umap.py`, `plot_umap_per_label.py` | UMAP plots |
| `colors.py` | Colour palettes |

## manuscript/

R Markdown scripts that generate the manuscript figures, run inside the R container. Each file is named after the panels it produces.

| Script | Panels |
|--------|--------|
| `Figure_1C.Rmd` | Fig. 1C |
| `Figure_1D_S1.Rmd` | Fig. 1D, Fig. S1 |
| `Figure_1E_1F_S2.Rmd` | Fig. 1E, 1F, Fig. S2 |
| `Figure_3D_S7.Rmd` | Fig. 3D, Fig. S7 |
