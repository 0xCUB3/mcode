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

LIMIT=${LIMIT:-300}

echo "=== SWE-bench Live Lite (RAW model, no mellea) ==="
echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Model:       ${MODEL}"
echo "Shards:      ${SHARD_COUNT}"
echo "Tasks:       ${LIMIT}"
echo ""

LOG_DIR="${BV_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

rm -f "${BV_RESULTS_DIR}"/live-raw-*.db "${LOG_DIR}"/live-raw-*.log

EXEC_HOST_FLAG=""
if [[ -n "${BV_EXEC_HOST:-}" ]]; then
    EXEC_HOST_FLAG="-m ${BV_EXEC_HOST}"
fi
bsub -G "${BV_GROUP}" \
    -J "live-raw" \
    -q "${BV_QUEUE}" \
    -n 1 \
    ${EXEC_HOST_FLAG} \
    -R "span[hosts=1]" \
    -W 24:00 \
    -o "${LOG_DIR}/live-raw-mega.log" \
    -e "${LOG_DIR}/live-raw-mega.log" \
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

source '"${BV_MCODE_DIR}"'/venv/bin/activate
export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
export MCODE_CONTEXT_WINDOW='"'${VLLM_MAX_MODEL_LEN}'"'
export MCODE_REACT_TIMEOUT=120
export MCODE_KEEP_IMAGES=1

SHARD_COUNT='"${SHARD_COUNT}"'
LIMIT='"${LIMIT}"'
LOG_DIR='"${LOG_DIR}"'
RESULTS_DIR='"${BV_RESULTS_DIR}"'

echo "=== Phase 0: Reuse cached Live images (already pulled) ==="
date

run_shard() {
    local SHARD_IDX=$1

    local LOCAL_DB=/tmp/live-raw-shard${SHARD_IDX}.db
    echo "[raw-s${SHARD_IDX}] starting on $(hostname)" >> ${LOG_DIR}/live-raw-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/live-raw-shard${SHARD_IDX}.log

    python -m mcode bench swebench-live \
        --backend '"'${BACKEND}'"' \
        --model '"'${MODEL}'"' \
        --split lite \
        --strategy raw \
        --loop-budget 1 \
        --timeout '"${TIMEOUT_S}"' \
        --limit ${LIMIT} \
        --shard-count ${SHARD_COUNT} \
        --shard-index ${SHARD_IDX} \
        --db ${LOCAL_DB} \
        >> ${LOG_DIR}/live-raw-shard${SHARD_IDX}.log 2>&1
    local RC=$?

    cp ${LOCAL_DB} ${RESULTS_DIR}/live-raw-shard${SHARD_IDX}.db 2>/dev/null || true
    echo "[raw-s${SHARD_IDX}] done (exit=${RC})" >> ${LOG_DIR}/live-raw-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/live-raw-shard${SHARD_IDX}.log
}

echo "=== Phase 1: ${SHARD_COUNT} shards in parallel (raw, no mellea) ==="
date

for i in $(seq 0 $((SHARD_COUNT - 1))); do
    run_shard $i &
done

echo "All ${SHARD_COUNT} shards launched, waiting..."
wait
echo "=== All shards complete ==="
date

# Merge shards
echo "Merging..."
python -m mcode merge-shards --force \
    --out ${RESULTS_DIR}/live-raw.db \
    ${RESULTS_DIR}/live-raw-shard*.db 2>&1 || echo "  MERGE FAILED"

# Print summary
echo ""
echo "=== RAW MODEL RESULTS ==="
echo "Model: '"${MODEL}"'"
echo "Strategy: raw (single-shot, no agent loop)"
echo ""
python3 << PYEOF
import sqlite3
db = "'"${BV_RESULTS_DIR}"'/live-raw.db"
conn = sqlite3.connect(db)
try:
    rows = conn.execute("SELECT task_id, passed FROM task_results ORDER BY task_id").fetchall()
except Exception as e:
    print(f"ERROR: {e}")
    exit()
passed = sum(1 for _, p in rows if p)
total = len(rows)
pct = 100*passed/total if total else 0
print(f"Raw model: {passed}/{total} = {pct:.1f}%")
conn.close()
PYEOF
'

echo ""
echo "=== Raw model job submitted ==="
echo "Strategy: raw (single-shot diff generation, no mellea agent)"
echo "Shards: ${SHARD_COUNT} parallel"
echo ""
echo "Monitor:  bjobs -w | grep live-raw"
echo "Logs:     ls ${LOG_DIR}/live-raw-*.log"
echo "Results:  tail -30 ${LOG_DIR}/live-raw-mega.log"
