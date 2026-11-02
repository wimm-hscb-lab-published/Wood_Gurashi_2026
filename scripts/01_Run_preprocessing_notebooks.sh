#!/usr/bin/env bash
#SBATCH --job-name=01_Run_preprocessing_notebooks
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=01_Run_preprocessing_notebooks_%A_%a.log
##SBATCH --constraint=<cpu-feature>   # uncomment to pin one CPU type if the partition is mixed
#
# Execute the preprocessing notebooks inside the Mosaic container and render each to HTML.
#
# Usage:
#   sbatch 01_Run_preprocessing_notebooks.sh                # every notebook in NOTEBOOKS, one after another
#   sbatch --array=0-N 01_Run_preprocessing_notebooks.sh    # one SLURM task per NOTEBOOKS entry (N = count - 1)
#   sbatch 01_Run_preprocessing_notebooks.sh --all          # every *.ipynb in NB_DIR
#   bash   01_Run_preprocessing_notebooks.sh --local        # run on this machine instead of through SLURM
#
# Reproducibility: clustering depends on the CPU type and thread count. Keep the partition,
# --cpus-per-task and THREADS unchanged between runs; each log records the node and CPU used.
#
# Outputs per notebook (in OUTDIR; default: next to the notebook):
#   <name>.html                    rendered notebook, also written when a run fails
#   logs/<name>.executed.ipynb     executed notebook (saved after every cell when papermill is available)
#   logs/<name>.log                run record (node, CPU, threads) and cell-by-cell progress

set -uo pipefail

# ── Settings ──
PROJECT=/project/meadlab/shared/02__Projects/02c__Test
SIF="${PROJECT}/envs/containers/mosaic.sif"
NB_DIR="${PROJECT}/notebooks/01_preprocessing"
OUTDIR=""      # empty = same folder as each notebook
THREADS=8      # threads for numerical libraries; keep equal to --cpus-per-task and constant between runs
HASHSEED=42    # PYTHONHASHSEED, set before Python starts

# ── Notebooks to run (uncomment to enable) ──
NOTEBOOKS=(
    01_Preprocessing_MDS001.ipynb
    # 01_Preprocessing_MDS002.ipynb
    # 01_Preprocessing_MDS003.ipynb
    # 01_Preprocessing_MDS005.ipynb
    # 01_Preprocessing_MDS006.ipynb
    # 01_Preprocessing_MDS007.ipynb
    # 01_Preprocessing_MDS008.ipynb
    # 01_Preprocessing_MDS009.ipynb
    # 01_Preprocessing_MDS010.ipynb
    # 01_Preprocessing_MDS011.ipynb
    # 01_Preprocessing_MDS012.ipynb
    # 01_Preprocessing_pMDS001.ipynb
    # 01_Preprocessing_pMDS002.ipynb
)

# ── Options ──
ALL=0; LOCAL=0
for arg in "$@"; do
    case "${arg}" in
        --all)   ALL=1 ;;
        --local) LOCAL=1 ;;
        *) echo "ERROR: unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" && ${LOCAL} -eq 0 ]]; then
    echo "ERROR: not running under SLURM. Submit with 'sbatch $0', or add --local to run here." >&2
    exit 2
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "${THREADS}" ]]; then
    echo "WARNING: SLURM allocated ${SLURM_CPUS_PER_TASK} CPUs but THREADS=${THREADS}; results may not match earlier runs." >&2
fi

if (( ALL )); then
    mapfile -t NOTEBOOKS < <(find "${NB_DIR}" -maxdepth 1 -name '*.ipynb' -printf '%f\n' | sort)
fi

# SLURM array: run only the entry for this task
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NOTEBOOKS=("${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}")
fi

# ── Container runtime ──
[[ -f "${SIF}" ]] || { echo "ERROR: container not found: ${SIF}" >&2; exit 1; }

RUNTIME=$(command -v apptainer || command -v singularity || true)
if [[ -z "${RUNTIME}" ]]; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    RUNTIME=$(command -v apptainer || command -v singularity || true)
fi
[[ -n "${RUNTIME}" ]] || { echo "ERROR: apptainer/singularity not available" >&2; exit 1; }

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/nbrun_XXXXXX")
trap 'rm -rf "${SCRATCH}"' EXIT

ENV_ARGS=(
    --env JUPYTER_RUNTIME_DIR="${SCRATCH}"
    --env JUPYTER_DATA_DIR="${SCRATCH}"
    --env IPYTHONDIR="${SCRATCH}/ipython"
    --env MPLCONFIGDIR="${SCRATCH}/mpl"
    --env PYTHONHASHSEED="${HASHSEED}"
    --env PYTHONUNBUFFERED=1
    --env OMP_NUM_THREADS="${THREADS}"
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMBA_NUM_THREADS="${THREADS}"
)

cexec() {
    "${RUNTIME}" exec --bind /project --bind "${SCRATCH}" "${ENV_ARGS[@]}" "${SIF}" "$@"
}

# ── Run record (written at the top of each notebook log) ──
CPU_MODEL=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
run_record() {
    echo "# host        : $(hostname)"
    echo "# cpu model   : ${CPU_MODEL:-unknown}"
    echo "# slurm job   : ${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-none}  cpus=${SLURM_CPUS_PER_TASK:-n/a}"
    echo "# threads     : ${THREADS}   PYTHONHASHSEED=${HASHSEED}"
    echo "# container   : ${SIF}"
    echo "# started     : $(date '+%F %T')"
}

# ── Pre-flight: annotation API must be reachable from this node ──
echo "Checking access to https://api.missionbio.io ..."
if ! cexec python - <<'EOF'
import sys, urllib.request, urllib.error
try:
    urllib.request.urlopen("https://api.missionbio.io", timeout=20)
except urllib.error.HTTPError:
    pass  # any HTTP response means the server is reachable
except Exception as exc:
    sys.exit(f"cannot reach https://api.missionbio.io ({exc})")
EOF
then
    echo "ERROR: variant annotation needs internet access to api.missionbio.io from this node." >&2
    exit 1
fi

if cexec python -c "import papermill" 2>/dev/null; then ENGINE=papermill; else ENGINE=nbconvert; fi

# ── Run one notebook ──
run_notebook() {
    local nb="$1"
    [[ "${nb}" = /* ]] || nb="${NB_DIR}/${nb}"
    [[ -f "${nb}" ]] || { echo "ERROR: notebook not found: ${nb}" >&2; return 1; }

    local name dir out logdir executed log rc start
    name=$(basename "${nb}" .ipynb)
    dir=$(dirname "${nb}")
    out="${OUTDIR:-${dir}}"
    logdir="${out}/logs"
    executed="${logdir}/${name}.executed.ipynb"
    log="${logdir}/${name}.log"
    mkdir -p "${logdir}"
    rm -f "${executed}"
    start=$(date +%s)

    run_record > "${log}"
    echo "── ${name} ── started $(date '+%F %T')   (log: ${log})"
    if [[ "${ENGINE}" == papermill ]]; then
        ( cd "${dir}" && cexec papermill "${nb}" "${executed}" \
              --kernel python3 --cwd "${dir}" \
              --log-output --no-progress-bar --request-save-on-cell-execute ) 2>&1 \
          | tee -a "${log}" | grep --line-buffered -E "Executing Cell|Error|Traceback"
        rc=${PIPESTATUS[0]}
    else
        ( cd "${dir}" && cexec jupyter nbconvert --to notebook --execute \
              --ExecutePreprocessor.timeout=-1 \
              --ExecutePreprocessor.kernel_name=python3 \
              --log-level=DEBUG \
              --output "${name}.executed" --output-dir "${logdir}" "${nb}" ) 2>&1 \
          | tee -a "${log}" | grep --line-buffered -E -A1 "Executing cell|CellExecutionError|Traceback"
        rc=${PIPESTATUS[0]}
    fi

    if [[ -f "${executed}" ]]; then
        cexec jupyter nbconvert --to html --output "${name}" --output-dir "${out}" "${executed}" \
            > /dev/null 2>&1 || echo "  ⚠ could not render HTML from ${executed}"
    fi

    echo "── ${name} ── finished $(date '+%F %T') after $(( ($(date +%s) - start) / 60 )) min (exit ${rc}) → ${out}/${name}.html"
    return ${rc}
}

# ── Main ──
echo "Runtime   : ${RUNTIME}"
echo "Container : ${SIF}"
echo "Engine    : ${ENGINE}"
echo "Node      : $(hostname)  (${CPU_MODEL:-unknown CPU})"
echo "Threads   : ${THREADS}   PYTHONHASHSEED: ${HASHSEED}"
echo "Notebooks : ${#NOTEBOOKS[@]}"

FAILED=()
for nb in "${NOTEBOOKS[@]}"; do
    run_notebook "${nb}" || FAILED+=("${nb}")
done

echo
echo "Done: $(( ${#NOTEBOOKS[@]} - ${#FAILED[@]} ))/${#NOTEBOOKS[@]} succeeded"
if (( ${#FAILED[@]} )); then
    printf 'Failed: %s\n' "${FAILED[@]}"
    exit 1
fi