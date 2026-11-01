# Single-cell multiomic profiling reveals distinct clonal architectures in TP53-mutated myeloid neoplasms

![Python](https://img.shields.io/badge/Python-mosaic-blue?style=flat&logo=python)
![R](https://img.shields.io/badge/R-4.x-blue?style=flat&logo=R)
[![GEO](https://img.shields.io/badge/GEO-GSE344978-lightgrey?style=flat)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE344978)

Analysis code for the single-cell DNA and protein (Tapestri) profiling of clonal architecture in TP53-mutated myelodysplastic syndrome (MDS).

## Summary

Biallelic TP53 mutation is associated with progression to AML, lack of response to chemotherapy, and a devastating overall survival of less than 6 months. Early detection of biallelic TP53 clones, and understanding how they evolve, is vital for refining diagnostic classification, improving patient outcomes, and advancing scientific understanding of this disease. This project uses single-cell DNA sequencing to accurately identify mono- and biallelic TP53 clones and their associated copy number alterations, and to decipher their clonal architecture and evolution.

## Contributors

- **Amy Wood** (first author) · TBD
- **Kristian Gurashi** (co-first author, computational analysis) · kristian.gurashi@ndcls.ox.ac.uk
- **Adam J. Mead** (senior and corresponding author)
- **Onima Chowdhury** (senior and corresponding author)

## Citation

> Wood, A.\*, Gurashi, K.\*, ..., Mead, A. J.†, Chowdhury, O.† *Single-cell multiomic profiling reveals distinct clonal architectures in TP53-mutated myeloid neoplasms.* (in preparation)
>
> \*These authors contributed equally. †These authors contributed equally as senior authors.

## Graphical Abstract

<!-- TODO: add image -->

---

## Study design

Patients were profiled across Tapestri runs (experiments). Where several patients share a run, cells were assigned back to each patient by genotype before downstream analysis. Donor labels (A, B, C) are local to each experiment; the Patient ID is the identifier used in the manuscript. Patients sampled more than once appear under several experiments.

| Experiment | Donor | Patient ID | Timepoint |
|------------|-------|------------|-----------|
| MDS001 | Donor A | Patient 03 | Baseline |
| MDS001 | Donor B | Patient 02 | Baseline |
| MDS001 | Donor C | Patient 01 | Baseline |
| MDS002 | Donor A | Patient 06 | Baseline |
| MDS002 | Donor B | Patient 08 | Baseline |
| MDS002 | Donor C | Patient 07 | Baseline |
| MDS003 | Donor A | Patient 05 | Baseline |
| MDS003 | Donor B | Patient 04 | Baseline |
| MDS005 | Donor A | Patient 10 | Baseline |
| MDS005 | Donor B | Patient 09 | Baseline |
| MDS006 | Donor A | Patient 11 | Baseline |
| MDS006 | Donor B | Patient 12 | Baseline |
| MDS007 | Donor A | Patient 13 | Baseline |
| MDS008 | Donor A | Patient 14 | Baseline |
| MDS009 | Donor A | Patient 16 | Baseline |
| MDS009 | Donor B | Patient 15 | Baseline |
| MDS010 | Donor A | Patient 17 | Baseline |
| MDS010 | Donor B | Patient 18 | Baseline |
| MDS011 | Donor A | Patient 15 | FU |
| MDS011 | Donor B | Patient 19 | Baseline |
| pMDS001 | Donor A | Patient 20 | Baseline |
| pMDS002 | Donor A | Patient 20 | FU |

FU, follow-up.

## Analyses

1. **Preprocessing**: alignment, barcode calling and genotyping (Tapestri Pipeline, run online).
2. **Demultiplexing**: assignment of cells in pooled runs back to each patient by germline genotype.
3. **Filtering**: cell, variant and protein QC.
4. **Patient data analysis**: per-patient TP53 allelic state, copy number, clonotypes and phylogeny.
5. **Integration and visualisation**: cohort-level integration and manuscript figures.

## Repository

```
notebooks/
  01_preprocessing/                   # 1–2. preprocessing and demultiplexing (method notes: 01_Preprocessing_Docs.md)
  02_protein_filtering/               # 3. filtering
  03_sample_analysis/                 # 4. patient data analysis
scripts/
  01_Run_preprocessing_notebooks.sh   # runs the preprocessing notebooks inside the container (SLURM)
  pipelines/                          # Python packages used by the notebooks (DNA_Utils, CNV_Utils, LOH_Pipeline, PRO_Utils; each has a README)
  manuscript/                         # 5. integration and figures (named by figure panel)
metadata/                             # per-experiment configuration
cohort_metadata/                      # cohort-level summary tables
envs/containers/                      # Mosaic container (image not tracked; see Requirements)
```

## Data

- **GEO:** [GSE344978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE344978)
  - Raw: FASTQs per Tapestri library (pooled runs contain several patients).
  - Processed: one multiplexed H5 file per Tapestri run (DNA + protein), plus demultiplexed H5 files per patient.

All coordinates reference **hg19**.

## Requirements

Python with Mission Bio Mosaic (`missionbio.mosaic-base`) and R. Analysis can be reproduced starting from the multiplexed H5 files.

The Python environment used in the analysis is provided as an Apptainer/Singularity container (`mosaic.sif`), available on Zenodo: [10.5281/zenodo.xxxxxxx](https://doi.org/10.5281/zenodo.xxxxxxx) *(pending)*.

The R environment used in the analysis is provided as an Apptainer/Singularity container (`r_env.sif`), available on Zenodo: [10.5281/zenodo.xxxxxxx](https://doi.org/10.5281/zenodo.xxxxxxx) *(pending)*.

## Computing environment

Python notebooks were executed interactively with `scripts/01_Run_preprocessing_notebooks.sh` (papermill, inside `mosaic.sif`) on the <cluster name> high-performance computing cluster (SLURM, partition `long`), using 8 CPU cores and 160 GB RAM job session on AMD EPYC 7713 (64-core) nodes. Numerical libraries were limited to 8 threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`) and `PYTHONHASHSEED=42`. Random seeds are fixed in the notebooks (`random_state=1337` for PCA, UMAP and clustering). Each run writes its node, CPU model and thread settings to `notebooks/<step>/logs/<notebook>.log`.

R scripts were run with `r_env.sif` on <machine / cluster>.

### Reproducibility note

Results can differ slightly between computers, even with the same code, software and settings. When re-running the analysis on different hardware, check that each cell cluster is still assigned to the correct patient before exporting.

The demultiplexed per-patient H5 files deposited on GEO allow the downstream analyses (filtering, patient analysis and figures) to be reproduced exactly, without re-running demultiplexing.

## License

CC BY 4.0. See `LICENSE`.