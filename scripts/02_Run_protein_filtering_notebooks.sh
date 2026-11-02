#!/usr/bin/env bash
#SBATCH --job-name=02_Run_protein_filtering_notebooks
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=02_Run_protein_filtering_notebooks_%A_%a.log
#
# Execute the protein-filtering notebooks inside the Mosaic container and render each to HTML.
#
# Usage:
#   sbatch 02_Run_protein_filtering_notebooks.sh                # every notebook in NOTEBOOKS, one after another
#   sbatch --array=0-N 02_Run_protein_filtering_notebooks.sh    # one SLURM task per NOTEBOOKS entry (N = count - 1)
#   sbatch 02_Run_protein_filtering_notebooks.sh --all          # every *.ipynb in NB_DIR
#   bash   02_Run_protein_filtering_notebooks.sh --local        # run on this machine instead of through SLURM
#   bash   02_Run_protein_filtering_notebooks.sh --check        # only run the input checks, then stop
#
# Requires the 01 preprocessing exports. For multiplexed experiments, BAM demultiplexing runs only if
# the run's BAM file is in raw_data/bam/<EXPERIMENT>/ (otherwise the notebook skips that step); it then
# needs pysam in the container and sinto at the notebook's SINTO_BIN path.
#
# Outputs per notebook (in OUTDIR; default: next to the notebook):
#   <name>.html                    rendered notebook, also written when a run fails
#   logs/<name>.executed.ipynb     executed notebook (saved after every cell when papermill is available)
#   logs/<name>.log                run record (node, CPU, threads) and cell-by-cell progress

set -uo pipefail

# ── Settings ──
PROJECT=/project/meadlab/shared/02__Projects/02c__Test
SIF="${PROJECT}/envs/containers/mosaic.sif"
NB_DIR="${PROJECT}/notebooks/02_protein_filtering"
OUTDIR=""      # empty = same folder as each notebook
THREADS=8      # threads for numerical libraries; keep equal to --cpus-per-task
HASHSEED=42    # PYTHONHASHSEED, set before Python starts

# ── Notebooks to run (uncomment to enable) ──
NOTEBOOKS=(
    02_Protein_Filtering_MDS001.ipynb
    # 02_Protein_Filtering_MDS002.ipynb
    # 02_Protein_Filtering_MDS003.ipynb
    # 02_Protein_Filtering_MDS005.ipynb
    # 02_Protein_Filtering_MDS006.ipynb
    # 02_Protein_Filtering_MDS007.ipynb
    # 02_Protein_Filtering_MDS008.ipynb
    # 02_Protein_Filtering_MDS009.ipynb
    # 02_Protein_Filtering_MDS010.ipynb
    # 02_Protein_Filtering_MDS011.ipynb
    # 02_Protein_Filtering_MDS012.ipynb
    # 02_Protein_Filtering_pMDS001.ipynb
    # 02_Protein_Filtering_pMDS002.ipynb
)

# ── Options ──
ALL=0; LOCAL=0; CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --all)   ALL=1 ;;
        --local) LOCAL=1 ;;
        --check) CHECK_ONLY=1; LOCAL=1 ;;
        *) echo "ERROR: unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" && ${LOCAL} -eq 0 ]]; then
    echo "ERROR: not running under SLURM. Submit with 'sbatch $0', or add --local to run here." >&2
    exit 2
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

# ── Pre-flight: inputs each notebook needs ──
NB_PATHS=()
for nb in "${NOTEBOOKS[@]}"; do
    [[ "${nb}" = /* ]] && NB_PATHS+=("${nb}") || NB_PATHS+=("${NB_DIR}/${nb}")
done

echo "Checking inputs ..."
cexec env PROJECT="${PROJECT}" NB_LIST="${NB_PATHS[*]}" python - <<'EOF'
import glob, importlib.util, json, os, re
import yaml

project = os.environ["PROJECT"]
n_warn = 0

def report(ok, message):
    global n_warn
    n_warn += 0 if ok else 1
    print(("  ok    " if ok else "  WARN  ") + message)

pysam_ok = importlib.util.find_spec("pysam") is not None
template = os.path.join(project, "scripts", "templates", "03_Sample_Analysis_Template.ipynb")

for nb in os.environ["NB_LIST"].split():
    print(os.path.basename(nb))
    if not os.path.exists(nb):
        report(False, f"notebook not found: {nb}")
        continue
    src = "\n".join("".join(c["source"]) for c in json.load(open(nb))["cells"] if c["cell_type"] == "code")
    exp = re.search(r'^EXPERIMENT = "([^"]+)"', src, re.M).group(1)
    sinto = re.search(r'^SINTO_BIN = "([^"]+)"', src, re.M)

    cfg = os.path.join(project, "metadata", exp, "experiment_config.yml")
    if not os.path.exists(cfg):
        report(False, f"config missing: {cfg}")
        continue
    multiplexed = yaml.safe_load(open(cfg)).get("multiplexed", False)

    h5_base = os.path.join(project, "processed_data", "h5", exp, *(["demultiplexed"] if multiplexed else []))
    patients = sorted(d for d in os.listdir(h5_base) if os.path.isdir(os.path.join(h5_base, d))) if os.path.isdir(h5_base) else []
    with_h5 = [d for d in patients if glob.glob(os.path.join(h5_base, d, "*.h5"))]
    report(bool(with_h5), f"{exp}: patient folders with H5 in {h5_base}: {with_h5 or 'none (run 01 first)'}")

    if multiplexed:
        bam_dir = os.path.join(project, "raw_data", "bam", exp)
        bams = glob.glob(os.path.join(bam_dir, "multiplexed", "*.bam")) or glob.glob(os.path.join(bam_dir, "*.bam"))
        print(("  ok    " if bams else "  info  ") + f"{exp}: multiplexed BAM in {bam_dir}"
              + ("" if bams else " — not found, BAM demultiplexing will be skipped"))
        if bams:
            report(bool(sinto) and os.path.exists(sinto.group(1)), f"{exp}: sinto at {sinto.group(1) if sinto else '?'}")
            report(pysam_ok, f"{exp}: pysam importable in the container")

report(os.path.exists(template), f"03 template: {template}" + ("" if os.path.exists(template) else " — missing, no 03 notebooks will be created"))
print(f"Input check: {n_warn} warning(s)")
EOF

(( CHECK_ONLY )) && exit 0

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