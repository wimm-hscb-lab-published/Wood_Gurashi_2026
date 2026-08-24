#!/bin/bash

set -euo pipefail

# ============================================================
# User-configurable paths
# ============================================================

CONDA_BASE_DIR="${CONDA_BASE_DIR:-$HOME/miniconda3}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-mosaic}"

# ============================================================
# Activate Conda
# ============================================================

source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"

conda activate "$ENVIRONMENT_NAME"

# ============================================================
# Confirm correct environment
# ============================================================

echo "============================================"
echo "Conda environment"
echo "============================================"

echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV}"

which python
python --version

which pip
python -m pip --version

echo


# ============================================================
# HUNTRESS directory
# ============================================================

HUNTRESS_DIR="${HUNTRESS_DIR:-$CONDA_BASE_DIR/envs/$ENVIRONMENT_NAME/lib/python3.10/site-packages/HUNTRESS}"


if [[ ! -d "${HUNTRESS_DIR}" ]]; then
    echo "ERROR: HUNTRESS directory not found:"
    echo "${HUNTRESS_DIR}"
    exit 1
fi


cd "${HUNTRESS_DIR}"

echo "============================================"
echo "HUNTRESS directory"
echo "============================================"

pwd

echo

# ============================================================
# Check requirements
# ============================================================

if [[ ! -f "requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found in:"
    pwd
    exit 1
fi

# ============================================================
# Install requirements
# ============================================================

echo "============================================"
echo "Installing HUNTRESS requirements"
echo "============================================"

python -m pip install -r requirements.txt


# ============================================================
# Final check
# ============================================================

echo
echo "============================================"
echo "Installation complete"
echo "============================================"

echo "Environment: ${CONDA_DEFAULT_ENV}"
echo "Python:"
which python
python --version

echo
echo "Pip:"
python -m pip --version