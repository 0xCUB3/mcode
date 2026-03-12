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

# Ablation config
LIMIT=${LIMIT:-300}
NAMESPACE=${NAMESPACE:-swebench}

echo "=== SWE-bench Lite Ablation Suite ==="
echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Model:       ${MODEL}"
echo "Shards:      ${SHARD_COUNT}"
echo "Tasks:       ${LIMIT}"
echo ""

LOG_DIR="${BV_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Clean old ablation results
rm -f "${BV_RESULTS_DIR}"/ablation-*.db "${LOG_DIR}"/ablation-*.log

# Single mega job: one podman service, sequential image pull, then parallel experiments
# Pin to a specific host if BV_EXEC_HOST is set (reuses cached images in /tmp).
EXEC_HOST_FLAG=""
if [[ -n "${BV_EXEC_HOST:-}" ]]; then
    EXEC_HOST_FLAG="-m ${BV_EXEC_HOST}"
fi
bsub -G "${BV_GROUP}" \
    -J "ablation-mega" \
    -q "${BV_QUEUE}" \
    -n 1 \
    ${EXEC_HOST_FLAG} \
    -R "span[hosts=1]" \
    -W 24:00 \
    -o "${LOG_DIR}/ablation-mega.log" \
    -e "${LOG_DIR}/ablation-mega.log" \
    bash -c '
set -e

# Single podman service on local /tmp
PD=/tmp/podman-ablation-$(id -u)
# Preserve graphroot to reuse cached images across runs.
# Only clean runroot (transient state) and socket.
rm -rf ${PD}/runroot ${PD}/podman.sock 2>/dev/null || true
mkdir -p ${PD}/graphroot ${PD}/runroot
export XDG_RUNTIME_DIR=${PD}
SOCK=${PD}/podman.sock
podman --cgroup-manager=cgroupfs --storage-driver=overlay \
    --storage-opt ignore_chown_errors=true \
    --root=${PD}/graphroot --runroot=${PD}/runroot \
    system service --time=0 unix://${SOCK} &
PODMAN_PID=$!
sleep 3
export DOCKER_HOST=unix://${SOCK}
# Use anonymous pulls to avoid per-user rate limit exhaustion.
# Authenticated limit (200/6h) may already be exhausted; anonymous
# limit (100/6h) is per-IP and usually has capacity.
# export REGISTRY_AUTH_FILE=${HOME}/.config/containers/auth.json

trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT

source '"${BV_MCODE_DIR}"'/venv/bin/activate
export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
export MCODE_CONTEXT_WINDOW='"'${VLLM_MAX_MODEL_LEN}'"'
export MCODE_REACT_TIMEOUT=450
export MCODE_KEEP_IMAGES=1

SHARD_COUNT='"${SHARD_COUNT}"'
LIMIT='"${LIMIT}"'
LOG_DIR='"${LOG_DIR}"'
RESULTS_DIR='"${BV_RESULTS_DIR}"'

SKIP_PREPULL='"${SKIP_PREPULL:-0}"'
if [[ "${SKIP_PREPULL}" != "1" ]]; then
    echo "=== Phase 0: Pre-pull all SWE-bench Lite images ==="
    date
    python3 '"${BV_MCODE_DIR}"'/deploy/bluevela/prepull_images.py '"${NAMESPACE}"' ${LIMIT}
    echo "=== Phase 0 complete ==="
    date
else
    echo "=== Phase 0: SKIPPED (SKIP_PREPULL=1) ==="
fi

run_shard() {
    local EXP_NAME=$1
    local SHARD_IDX=$2
    shift 2

    for var in "$@"; do
        export "${var}"
    done

    local LOCAL_DB=/tmp/ablation-${EXP_NAME}-shard${SHARD_IDX}.db
    echo "[${EXP_NAME}-s${SHARD_IDX}] starting on $(hostname)" >> ${LOG_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.log

    python -m mcode bench swebench-lite \
        --backend '"'${BACKEND}'"' \
        --model '"'${MODEL}'"' \
        --loop-budget ${LOOP_BUDGET} \
        --timeout '"${TIMEOUT_S}"' \
        --namespace '"${NAMESPACE}"' \
        --limit ${LIMIT} \
        --shard-count ${SHARD_COUNT} \
        --shard-index ${SHARD_IDX} \
        --db ${LOCAL_DB} \
        >> ${LOG_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.log 2>&1
    local RC=$?

    cp ${LOCAL_DB} ${RESULTS_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.db 2>/dev/null || true
    echo "[${EXP_NAME}-s${SHARD_IDX}] done (exit=${RC})" >> ${LOG_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/ablation-${EXP_NAME}-shard${SHARD_IDX}.log
}

echo "=== Phase 1: All 5 experiments in parallel (images cached) ==="
date

# Launch all experiments, all shards
for i in $(seq 0 $((SHARD_COUNT - 1))); do
    run_shard "baseline" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
    run_shard "A-prompt" $i "MCODE_EXPLORE_PROMPT=1" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
    run_shard "B-warning" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=1" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
    run_shard "C-budget25" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=25" &
    run_shard "D-nudge" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=1" "LOOP_BUDGET=15" &
done

echo "All 35 shards launched, waiting..."
wait
echo "=== All experiments complete ==="
date

# Merge shards per experiment
for EXP in baseline A-prompt B-warning C-budget25 D-nudge; do
    echo "Merging ${EXP}..."
    python -m mcode merge-shards --force \
        --out ${RESULTS_DIR}/ablation-${EXP}.db \
        ${RESULTS_DIR}/ablation-${EXP}-shard*.db 2>&1 || echo "  MERGE FAILED for ${EXP}"
done

# Print summary
echo ""
echo "=== ABLATION RESULTS ==="
echo "Model: '"${MODEL}"'"
echo ""
python3 << PYEOF
import sqlite3, glob
for db in sorted(glob.glob("'"${BV_RESULTS_DIR}"'/ablation-*.db")):
    if "shard" in db:
        continue
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
echo "=== Ablation mega job submitted ==="
echo "Phase 0: Sequential pre-pull of all ~300 Docker images"
echo "Phase 1: 5 experiments x ${SHARD_COUNT} shards = $((5 * SHARD_COUNT)) parallel (images cached)"
echo ""
echo "Monitor:  bjobs -w | grep ablation"
echo "Logs:     ls ${LOG_DIR}/ablation-*.log"
echo "Results:  tail -20 ${LOG_DIR}/ablation-mega.log"
