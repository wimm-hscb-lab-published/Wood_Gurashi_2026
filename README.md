# TP53 MDS Single-Cell Multi-Omics

## Quick start

```bash
bash Initialise_project.sh          # once
# edit Sample_sheet.csv
bash Create_project_directories.sh  # as many times as needed
```

## Project structure

```
├── raw_data/
│   ├── bam/{experiment}/
│   └── fastq/{experiment}/
│       ├── multiplexed/                    # only if multiple patients per experiment
│       └── demultiplexed/{patient}/
│       OR
│       └── {patient}/                      # flat if single-patient experiment
│
├── processed_data/
│   └── h5/{experiment}/                    # same auto-detection as fastq
│
├── metadata/
│   ├── SAMPLES_MANIFEST.tsv
│   └── {experiment}/{patient}/
│       ├── sample_config.yml
│       └── whitelist.txt
│
├── results/
│   ├── demultiplexing/
│   ├── per_sample/{experiment}/{patient}/
│   │   ├── dna_cnv/
│   │   │   ├── variants/
│   │   │   │   ├── somatic_variants/
│   │   │   │   ├── germline_variants/
│   │   │   │   ├── informative_variants/
│   │   │   │   └── whitelist_variants/
│   │   │   ├── phylogeny/
│   │   │   ├── clonotypes/
│   │   │   │   ├── allele_frequencies/
│   │   │   │   └── ploidy/
│   │   │   └── matrices/
│   │   └── protein/
│   │       ├── matrices/
│   │       └── annotation/
│   ├── integrated/
│   ├── figures/
│   ├── summaries/
│   └── general/
│
├── notebooks/
│   ├── 01_preprocessing/
│   ├── 02_protein_filtering/
│   ├── 03_sample_analysis/
│   └── 04_integration/
│
├── scripts/
│   ├── general/
│   ├── pipelines/
│   └── templates/
│
├── envs/
├── docs/
└── logs/
```

## Multiplexed vs single-sample experiments

The script automatically detects whether an experiment is multiplexed
by counting how many patients share the same experiment name in
Sample_sheet.csv.

- **Multiple patients** (e.g. MDS001 with Patient_ID01, ID02, ID03):
  fastq and h5 directories get `multiplexed/` + `demultiplexed/{patient}/`
  subdirectories. The `demultiplexed/` folders are GEO-ready.

- **Single patient** (e.g. HD001_FACS with only Donor_HD01):
  fastq and h5 get a flat `{patient}/` directory — no split needed.

## Experiment naming

Sorting methods (FACS, MACS) are part of the experiment name, not separate
subdirectories — some experiments have both multiplexed together.

## Genome build

All coordinates reference **hg19**.

## Adding new samples

1. Add rows to `Sample_sheet.csv`
2. Run `bash Create_project_directories.sh`
3. Commit: `git add -A && git commit -m "Add new samples"`

Nothing is ever overwritten.

## Python helper

```python
import sys; sys.path.insert(0, 'scripts')
from general.manifest import load_manifest, get_sample, sample_data_path

manifest = load_manifest()
sample   = get_sample("Patient_ID01", "MDS001")
path     = sample_data_path("Patient_ID01", "MDS001", "processed", "h5")
```
