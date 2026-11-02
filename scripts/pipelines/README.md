# pipelines

Python packages used by the analysis notebooks in `notebooks/`. They run inside the Mosaic container (`mosaic.sif`), or the equivalent `mosaic` conda environment.

```
pipelines/
├── DNA_Utils/      # single-cell genotype processing
├── CNV_Utils/      # copy number and allelic imbalance
├── LOH_Pipeline/   # loss-of-heterozygosity detection
├── PRO_Utils/      # protein (antibody) analysis
├── docs/           # scripts that draw the README figures
└── tests/          # unit tests
```

Each package has its own README describing every function, with schematics of the main methods and a list of known quirks.

## Packages

| Package | What it does | Used in | Main functions called by the notebooks |
|---|---|---|---|
| [`DNA_Utils`](DNA_Utils/README.md) | Cell QC (doublets, low-quality cells), germline and somatic variant identification, clones, variant allele frequencies | 01, 03 | `define_doublet_cutoff_per_donor`, `apply_doublet_calls_with_corr_reassign`, `apply_low_quality_qc`, `fast_af_heatmap`, `find_germline_variants`, `find_somatic_variants`, `filter_somatic_variant_outliers`, `get_VAF`, `get_clones_size` |
| [`CNV_Utils`](CNV_Utils/README.md) | Informative variants between clones, per-amplicon ploidy, CNV and allele-frequency plots | 03 | `test_variant_informativeness`, `extract_cnv_data`, `prepare_ploidy_region_plot_inputs`, `plot_integrated_cnv_baf_with_significance`, `visualise_variants_by_clone` |
| [`LOH_Pipeline`](LOH_Pipeline/README.md) | Detects loss of heterozygosity per chromosome arm and refines clones, in phases | 03 | `build_config`, `run` |
| [`PRO_Utils`](PRO_Utils/README.md) | Protein cell-type labels, UMAPs, cluster quality, hashtag calling | 03 | `update_protein_labels_from_dataframe`, `plot_protein_umap`, `plot_protein_homogeneity_silhouette`, `call_hto_present_absent` |

## How the notebooks load them

The notebooks add `scripts/` and `scripts/pipelines/` to the Python path, import the four packages, and reload all their submodules, so edits to the code are picked up without restarting the kernel:

```python
sys.path.insert(0, os.path.join(wd, "scripts"))
sys.path.insert(0, os.path.join(wd, "scripts", "pipelines"))
import DNA_Utils, PRO_Utils, CNV_Utils, LOH_Pipeline
```

Functions are then called as `DNA_Utils.find_germline_variants(...)`, `LOH_Pipeline.run(...)` and so on.

## Resource files

Functions that read shared reference files (common variants, blacklists) or write cohort tables (germline patterns) use the project's `resources/` folder. The project folder is found from the package location (`<project>/scripts/pipelines/`); set the environment variable `TAPESTRI_PROJECT_ROOT` to point elsewhere.

## Tests

From this folder, with the `mosaic` conda environment:

```bash
~/miniconda3/envs/mosaic/bin/python -m pytest -c tests/pytest.ini tests
```

Tests run in under a minute on synthetic data and need no input files or network access. A few tests are marked as expected failures (`xfail`): each documents a known quirk in the code that has been left unchanged.

## README figures

The schematics in each package's `docs/` folder are drawn by `docs/make_<package>_figures.py`, which share the style in `docs/_style.py`:

```bash
~/miniconda3/envs/mosaic/bin/python docs/make_DNA_Utils_figures.py
```

## Links from the READMEs to the code

In each package README, every function is a collapsible block with a link to the line where it is defined (for example `cnv_extraction.py:309`). After editing the code, update those line numbers with:

```bash
~/miniconda3/envs/mosaic/bin/python docs/readme_toggles.py refresh DNA_Utils CNV_Utils LOH_Pipeline PRO_Utils
```

## Note on allele frequencies

Mosaic stores allele frequencies (the `AF` layer) in percent, 0–100. Most functions detect this and scale their thresholds; the exceptions are noted in the package READMEs (see the doublet scoring in `DNA_Utils`).
