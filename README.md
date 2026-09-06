# Single-cell multiomic profiling reveals distinct clonal architectures in TP53-mutated myeloid neoplasms

Project information:
Coded in/requires:
Python (conda env `mosaic`, mission bio missionbio.mosaic-base), R, Bash, SLURM

General Description:
First Author: Kristian Gurashi ([kristian.gurashi@ndcls.ox.ac.uk](mailto:kristian.gurashi@ndcls.ox.ac.uk))
Correspondance:
TBD
Disease: Myelodysplastic Syndrome (MDS)
Graphical Abstract:
TBD
Publication link:
TBD
Content:
This is the code repository for the TP53 MDS single-cell multi-omics manuscript (Wood, Gurashi et al., 2026). It contains the pipeline used to process Tapestri single-cell DNA + protein data, from raw demultiplexing through variant calling, clonotype/phylogeny inference, and protein annotation.

Instructions on reproducibility:

1. Clone the GitHub repository as follows:

```
git clone https://github.com/wimm-hscb-lab-published/Wood_Gurashi_2026.git
```

2. Access the cloned directory, initialise the project structure and populate the sample sheet:

```
cd Wood_Gurashi_2026/
bash Initialise_project.sh          # once
# edit Sample_sheet.csv
bash Create_project_directories.sh  # as many times as needed
```

3. Download and decompress the processed input data deposited on GEO/NCBI:
[Link to repository](TBD)

4. To ensure reproducibility run as follows:

* Please verify that your Python and R package versions match those used to generate figures in the manuscript - [Check environment specification](envs/)
* Bash/SLURM scripts are only needed if processing raw sequencing files deposited on GEO

Notes on multiplexed vs single-sample experiments: experiment folders are automatically split into `multiplexed/` + `demultiplexed/{patient}/` when multiple patients share an experiment in `Sample_sheet.csv` (GEO-ready), or left as a flat `{patient}/` directory for single-patient experiments. Sorting methods (FACS, MACS) are part of the experiment name, not separate subdirectories. All coordinates reference **hg19**.

Citation:
If using any code here published and/or data from this publication please remember to cite us:

* APA Style:

* Vancouver Style: