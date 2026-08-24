#!/bin/bash
#SBATCH --job-name=Conda_installation_Mosaic
#SBATCH --output=Report-%x.%j.out
#SBATCH --error=Report-%x.%j.err
#SBATCH --time=26:00:00
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email@institution.ac.uk>

# ── User-configurable paths ──
# Set these to match your own environment before running
CONDA_BASE_DIR="${CONDA_BASE_DIR:-$HOME/miniconda3}"
ENVIRONMENT_DIR="${ENVIRONMENT_DIR:-./environments/Mosaic}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-mosaic}"

source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"

# Remove any existing/corrupted environment
rm -rf "$CONDA_BASE_DIR/envs/$ENVIRONMENT_NAME"
conda env remove -n "$ENVIRONMENT_NAME" -y

# Clean conda cache (removes corrupted package files)
conda clean --all -y

if [ -f "$ENVIRONMENT_DIR/Mosaic.yml" ]; then
    echo "Creating environment from file…"
    conda env create -f "$ENVIRONMENT_DIR/Mosaic.yml"
else
    echo "Creating Mosaic environment from scratch…"
    conda create -n "$ENVIRONMENT_NAME" -c missionbio -c conda-forge \
             python=3.10 \
             missionbio.mosaic-base=3.12.2 \
             plotly=5.* \
             python-kaleido=0.2.1 -y

    CONDA_PREFIX="$CONDA_BASE_DIR/envs/$ENVIRONMENT_NAME"

    conda activate "$ENVIRONMENT_NAME"
    "$CONDA_PREFIX/bin/python3" -m pip install --upgrade pip
    "$CONDA_PREFIX/bin/python3" -m pip install notebook==6.5.4
    "$CONDA_PREFIX/bin/python3" -m pip install umap-learn==0.5.6
    "$CONDA_PREFIX/bin/python3" -m pip install seaborn==0.13.2
    "$CONDA_PREFIX/bin/python3" -m pip install ipykernel==6.30.1
    "$CONDA_PREFIX/bin/python3" -m pip install scanpy==1.11.4
    "$CONDA_PREFIX/bin/python3" -m pip install itables==2.4.4
    "$CONDA_PREFIX/bin/python3" -m pip install scikit-learn==1.7.1
    "$CONDA_PREFIX/bin/python3" -m pip install harmonypy==0.0.10
    "$CONDA_PREFIX/bin/python3" -m pip install chrov==0.0.3
    "$CONDA_PREFIX/bin/python3" -m pip install xgboost==3.0.3
    "$CONDA_PREFIX/bin/python3" -m pip install leidenalg==0.10.2
    "$CONDA_PREFIX/bin/python3" -m pip install venny4py==1.0.3
    "$CONDA_PREFIX/bin/python3" -m pip install netcal==1.3.6
    "$CONDA_PREFIX/bin/python3" -m pip install adjustText==1.3.0
    "$CONDA_PREFIX/bin/python3" -m pip install panel==1.7.5
    "$CONDA_PREFIX/bin/python3" -m pip install dash==3.2.0
    "$CONDA_PREFIX/bin/python3" -m pip install pyucell==0.5.0
    "$CONDA_PREFIX/bin/python3" -m pip install git+https://github.com/uom-eoh-lab-published/2024__EspressoPro.git
    "$CONDA_PREFIX/bin/python3" -m pip install pysam==0.23.3
    "$CONDA_PREFIX/bin/python3" -m pip install sinto==0.10.1
fi