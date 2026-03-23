#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

VLLM_HOST_FILE="${BV_MCODE_DIR}/vllm_host.txt"

if [[ ! -f "${VLLM_HOST_FILE}" ]]; then
  echo "ERROR: vLLM host file not found. Run ./start-vllm.sh first." >&2
  exit 1
fi

VLLM_HOST="$(cat "${VLLM_HOST_FILE}")"
OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

NAMESPACE=${NAMESPACE:-swebench}
TASK_IDS_FILE=${TASK_IDS_FILE:-${BV_RESULTS_DIR}/rerun-tasks.json}

echo "=== SWE-bench Lite Rerun (failed tasks only) ==="
echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Model:       ${MODEL}"
echo "Task IDs:    ${TASK_IDS_FILE}"
echo ""

LOG_DIR="${BV_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

EXEC_HOST_FLAG=""
if [[ -n "${BV_EXEC_HOST:-}" ]]; then
    EXEC_HOST_FLAG="-m ${BV_EXEC_HOST}"
fi
bsub -G "${BV_GROUP}" \
    -J "rerun-mega" \
    -q "${BV_QUEUE}" \
    -n 1 \
    ${EXEC_HOST_FLAG} \
    -R "span[hosts=1]" \
    -W 12:00 \
    -o "${LOG_DIR}/rerun-mega.log" \
    -e "${LOG_DIR}/rerun-mega.log" \
    bash -c '
set -e

# Podman storage: graphroot on proj (large), runroot on /tmp (fast, small)
PD_RUN=/tmp/podman-mcode-$(id -u)
PD_GRAPH='"${BV_PODMAN_ROOT}"'/graphroot
rm -rf ${PD_RUN} 2>/dev/null || true
mkdir -p ${PD_GRAPH} ${PD_RUN}/runroot
export XDG_RUNTIME_DIR=${PD_RUN}
SOCK=${PD_RUN}/podman.sock
podman --cgroup-manager=cgroupfs --storage-driver=overlay \
    --storage-opt ignore_chown_errors=true \
    --root=${PD_GRAPH} --runroot=${PD_RUN}/runroot \
    system service --time=0 unix://${SOCK} &
PODMAN_PID=$!
sleep 3
export DOCKER_HOST=unix://${SOCK}

trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT

source '"${BV_MCODE_DIR}"'/.venv/bin/activate
export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
export MCODE_CONTEXT_WINDOW='"'${VLLM_MAX_MODEL_LEN}'"'
export MCODE_REACT_TIMEOUT=450
export MCODE_KEEP_IMAGES=1

TASK_IDS_FILE='"${TASK_IDS_FILE}"'
LOG_DIR='"${LOG_DIR}"'
RESULTS_DIR='"${BV_RESULTS_DIR}"'

echo "=== Pre-pulling rerun images ==="
date
# Pull only the images we need
python3 << PYEOF
import json, docker
from swebench.harness.test_spec.test_spec import make_test_spec
from datasets import load_dataset

task_ids = set(json.load(open("${TASK_IDS_FILE}")))
ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
tasks = [t for t in ds if t["instance_id"] in task_ids]
print(f"Pre-pulling images for {len(tasks)} tasks...")

client = docker.from_env()
for i, task in enumerate(tasks):
    spec = make_test_spec(task, namespace="swebench")
    name = spec.instance_image_key
    fq = name if "/" in name and "." in name.split("/")[0] else f"docker.io/{name}"
    try:
        client.images.get(name)
        print(f"  [{i+1}/{len(tasks)}] cached {name}")
        continue
    except Exception:
        pass
    try:
        client.images.get(fq)
        print(f"  [{i+1}/{len(tasks)}] cached {fq}")
        continue
    except Exception:
        pass
    try:
        print(f"  [{i+1}/{len(tasks)}] pulling {fq}...", flush=True)
        for line in client.api.pull(fq, stream=True, decode=True):
            if "error" in line:
                raise RuntimeError(line["error"])
    except Exception as e:
        print(f"  [{i+1}/{len(tasks)}] FAILED {fq}: {e}", flush=True)
PYEOF
echo "=== Pre-pull complete ==="
date

run_experiment() {
    local EXP_NAME=$1
    shift

    for var in "$@"; do
        export "${var}"
    done

    local LOCAL_DB=/tmp/rerun-${EXP_NAME}.db
    echo "[${EXP_NAME}] starting on $(hostname)" >> ${LOG_DIR}/rerun-${EXP_NAME}.log
    date >> ${LOG_DIR}/rerun-${EXP_NAME}.log

    python -m mcode bench swebench-lite \
        --backend '"'${BACKEND}'"' \
        --model '"'${MODEL}'"' \
        --loop-budget ${LOOP_BUDGET} \
        --timeout '"${TIMEOUT_S}"' \
        --namespace '"${NAMESPACE}"' \
        --task-ids ${TASK_IDS_FILE} \
        --db ${LOCAL_DB} \
        >> ${LOG_DIR}/rerun-${EXP_NAME}.log 2>&1
    local RC=$?

    cp ${LOCAL_DB} ${RESULTS_DIR}/rerun-${EXP_NAME}.db 2>/dev/null || true
    echo "[${EXP_NAME}] done (exit=${RC})" >> ${LOG_DIR}/rerun-${EXP_NAME}.log
    date >> ${LOG_DIR}/rerun-${EXP_NAME}.log
}

echo "=== Running 5 experiments in parallel ==="
date

run_experiment "baseline" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
run_experiment "A-prompt" "MCODE_EXPLORE_PROMPT=1" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
run_experiment "B-warning" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=1" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
run_experiment "C-budget25" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=25" &
run_experiment "D-nudge" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=1" "LOOP_BUDGET=15" &

echo "All 5 experiments launched, waiting..."
wait
echo "=== All experiments complete ==="
date

# Print summary
echo ""
echo "=== RERUN RESULTS ==="
python3 << PYEOF
import sqlite3, glob
for db in sorted(glob.glob("'"${BV_RESULTS_DIR}"'/rerun-*.db")):
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT task_id, passed FROM task_results ORDER BY task_id").fetchall()
    except Exception as e:
        print(f"{db}: ERROR {e}")
        continue
    passed = sum(1 for _, p in rows if p)
    total = len(rows)
    pct = 100*passed/total if total else 0
    print(f"{db}: {passed}/{total} = {pct:.1f}%")
    conn.close()
PYEOF
'

echo ""
echo "=== Rerun job submitted ==="
echo "Monitor:  bjobs -w | grep rerun"
echo "Logs:     ls ${LOG_DIR}/rerun-*.log"
echo "Results:  tail -20 ${LOG_DIR}/rerun-mega.log"
