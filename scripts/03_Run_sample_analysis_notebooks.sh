#!/usr/bin/env bash
#SBATCH --job-name=03_Run_sample_analysis_notebooks
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=03_Run_sample_analysis_notebooks_%A_%a.log
##SBATCH --constraint=<cpu-feature>   # uncomment to pin one CPU type if the partition is mixed
#
# Execute the per-patient sample-analysis notebooks inside the Mosaic container and render each to HTML.
#
# Usage:
#   sbatch 03_Run_sample_analysis_notebooks.sh                # every notebook in NOTEBOOKS, one after another
#   sbatch --array=0-N 03_Run_sample_analysis_notebooks.sh    # one SLURM task per NOTEBOOKS entry (N = count - 1)
#   sbatch 03_Run_sample_analysis_notebooks.sh --all          # every *.ipynb in NB_DIR
#   bash   03_Run_sample_analysis_notebooks.sh --local        # run on this machine instead of through SLURM
#   bash   03_Run_sample_analysis_notebooks.sh --check        # only run the network and input checks, then stop
#
# Requires the 02 protein-filtered H5 files, the resource files in <PROJECT>/resources/ listed in
# notebooks/03_sample_analysis/README.md, and internet access to the Mission Bio annotation service.
# The EspressoPro models come with the container (Hugging Face is only needed if they are missing).
#
# Reproducibility: these notebooks run UMAP and clustering, which depend on the CPU type and thread count.
# Keep the partition, --cpus-per-task and THREADS unchanged between runs; each log records the node and CPU used.
#
# Outputs per notebook (in OUTDIR; default: next to the notebook):
#   <name>.html                    rendered notebook, also written when a run fails
#   logs/<name>.executed.ipynb     executed notebook (saved after every cell when papermill is available)
#   logs/<name>.log                run record (node, CPU, threads) and cell-by-cell progress

set -uo pipefail

# ── Settings ──
PROJECT=/project/meadlab/shared/02__Projects/02c__Test
SIF="${PROJECT}/envs/containers/mosaic.sif"
NB_DIR="${PROJECT}/notebooks/03_sample_analysis"
OUTDIR=""      # empty = same folder as each notebook
THREADS=8      # threads for numerical libraries; keep equal to --cpus-per-task and constant between runs
HASHSEED=42    # PYTHONHASHSEED, set before Python starts

# ── Notebooks to run (uncomment to enable) ──
NOTEBOOKS=(
    03_Sample_Analysis_MDS001_Donor_A.ipynb
    # 03_Sample_Analysis_MDS001_Donor_B.ipynb
    # 03_Sample_Analysis_MDS001_Donor_C.ipynb
    # 03_Sample_Analysis_MDS002_Donor_A.ipynb
    # 03_Sample_Analysis_MDS002_Donor_B.ipynb
    # 03_Sample_Analysis_MDS002_Donor_C.ipynb
    # 03_Sample_Analysis_MDS003_Donor_A.ipynb
    # 03_Sample_Analysis_MDS003_Donor_B.ipynb
    # 03_Sample_Analysis_MDS005_Donor_A.ipynb
    # 03_Sample_Analysis_MDS005_Donor_B.ipynb
    # 03_Sample_Analysis_MDS006_Donor_A.ipynb
    # 03_Sample_Analysis_MDS006_Donor_B.ipynb
    # 03_Sample_Analysis_MDS007_Donor_A.ipynb
    # 03_Sample_Analysis_MDS008_Donor_A.ipynb
    # 03_Sample_Analysis_MDS008_Donor_B.ipynb
    # 03_Sample_Analysis_MDS009_Donor_A.ipynb
    # 03_Sample_Analysis_MDS009_Donor_B.ipynb
    # 03_Sample_Analysis_MDS010_Donor_A.ipynb
    # 03_Sample_Analysis_MDS010_Donor_B.ipynb
    # 03_Sample_Analysis_MDS011_Donor_A.ipynb
    # 03_Sample_Analysis_MDS011_Donor_B.ipynb
    # 03_Sample_Analysis_pMDS001_Donor_A.ipynb
    # 03_Sample_Analysis_pMDS002_Donor_A.ipynb
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

# ── Pre-flight 1: network ──
echo "Checking network access ..."
if ! cexec python - <<'EOF'
import sys, urllib.request, urllib.error

def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=20)
    except urllib.error.HTTPError:
        pass  # any HTTP response means the server is reachable
    except Exception as exc:
        return str(exc)
    return None

err = reachable("https://api.missionbio.io")
if err:
    sys.exit(f"cannot reach https://api.missionbio.io ({err}); variant annotation needs it")
print("  ok    https://api.missionbio.io")
err = reachable("https://huggingface.co")
print(("  ok    " if not err else "  info  ") + "https://huggingface.co"
      + ("" if not err else f" ({err}) — only needed if the EspressoPro models are missing from the container"))
EOF
then
    echo "ERROR: variant annotation needs internet access to api.missionbio.io from this node." >&2
    exit 1
fi

# ── Pre-flight 2: inputs each notebook needs ──
NB_PATHS=()
for nb in "${NOTEBOOKS[@]}"; do
    [[ "${nb}" = /* ]] && NB_PATHS+=("${nb}") || NB_PATHS+=("${NB_DIR}/${nb}")
done

echo "Checking inputs ..."
cexec env PROJECT="${PROJECT}" NB_LIST="${NB_PATHS[*]}" python - <<'EOF'
import glob, json, os, re
import yaml

project = os.environ["PROJECT"]
resources = os.path.join(project, "resources")
n_warn = 0

def report(ok, message):
    global n_warn
    n_warn += 0 if ok else 1
    print(("  ok    " if ok else "  WARN  ") + message)

shared = {
    "cytobands": os.path.join(resources, "references", "cytobands.tsv"),
    "RefSeq GTF (hg19)": os.path.join(resources, "genome", "hg19", "hg19.ncbiRefSeq.gtf"),
    "dbSNP common variants": os.path.join(resources, "common_mutations", "dbsnp_v155_hg19_common_vars.txt"),
    "gnomAD common variants": os.path.join(resources, "common_mutations", "gnomad_genomes_r2.1.1_hg19_common_vars.txt"),
    "blacklist (ENCODE v2)": os.path.join(resources, "blacklists", "hg19-blacklist.v2.bed"),
    "blacklist (Kundaje merged)": os.path.join(resources, "blacklists", "hg19_blacklist_MergedList_Anshul Kundaje_Stanford.csv"),
    "germline pattern folder (written)": os.path.join(resources, "variants_patterns_tapestri"),
}
print("Shared resources")
for label, path in shared.items():
    report(os.path.exists(path), f"{label}: {path}")

for nb in os.environ["NB_LIST"].split():
    print(os.path.basename(nb))
    if not os.path.exists(nb):
        report(False, f"notebook not found: {nb}")
        continue
    src = "\n".join("".join(c["source"]) for c in json.load(open(nb))["cells"] if c["cell_type"] == "code")
    exp = re.search(r'^EXPERIMENT = "([^"]+)"', src, re.M)
    pat = re.search(r'^PATIENT_ID = "([^"]+)"', src, re.M)
    if not (exp and pat):
        report(False, "could not read EXPERIMENT / PATIENT_ID from the notebook")
        continue
    exp, pat = exp.group(1), pat.group(1)

    cfg_path = os.path.join(project, "metadata", exp, "experiment_config.yml")
    if not os.path.exists(cfg_path):
        report(False, f"config missing: {cfg_path}")
        continue
    cfg = yaml.safe_load(open(cfg_path)) or {}
    multiplexed = cfg.get("multiplexed", False)

    h5_dir = os.path.join(project, "processed_data", "h5", exp, *(["demultiplexed"] if multiplexed else []), pat)
    h5 = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    filtered = [f for f in h5 if "protein_filtered" in os.path.basename(f).lower()]
    report(bool(filtered), f"{exp}/{pat}: protein-filtered H5 in {h5_dir}"
           + ("" if filtered else (" — only unfiltered H5 found (run 02 first)" if h5 else " — missing (run 01 and 02 first)")))

    default_wl = re.search(r'config\.get\(\s*"germline_whitelist",\s*"([^"]+)"', src)
    wl = cfg.get("germline_whitelist", default_wl.group(1) if default_wl else None)
    if wl:
        report(os.path.exists(os.path.join(project, wl)), f"{exp}/{pat}: germline whitelist {wl}")

    hard_coded = sorted(set(re.findall(r'/project/[^"\'\s]+', src)))
    report(not hard_coded, f"{exp}/{pat}: no hard-coded /project paths"
           + ("" if not hard_coded else f" — found {hard_coded[:3]}"))

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