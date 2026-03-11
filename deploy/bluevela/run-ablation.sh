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
rm -f "${BV_RESULTS_DIR}"/ablation-*.db "${BV_RESULTS_DIR}"/ablation-*.log

# Submit a single mega job that:
# 1. Starts one shared podman service on local /tmp
# 2. Runs all 5 experiments x N shards in parallel, sharing images
# 3. Merges and summarizes
bsub -G "${BV_GROUP}" \
    -J "ablation-mega" \
    -q "${BV_QUEUE}" \
    -n 1 \
    -R "span[hosts=1]" \
    -W 12:00 \
    -o "${LOG_DIR}/ablation-mega.log" \
    -e "${LOG_DIR}/ablation-mega.log" \
    bash -c '
set -e

# Single shared podman service on local /tmp (no NFS contention)
PD=/tmp/podman-ablation-$(id -u)
mkdir -p ${PD}/graphroot ${PD}/runroot
export XDG_RUNTIME_DIR=${PD}
SOCK=${PD}/podman.sock
rm -f ${SOCK}
podman --cgroup-manager=cgroupfs --storage-driver=overlay \
    --root=${PD}/graphroot --runroot=${PD}/runroot \
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
export MCODE_REACT_TIMEOUT=450
export MCODE_KEEP_IMAGES=1

SHARD_COUNT='"${SHARD_COUNT}"'
LIMIT='"${LIMIT}"'

echo "=== Phase 1: Baseline (pulls images into local cache) ==="
date

# Run baseline shards staggered to avoid pull storms
export MCODE_EXPLORE_PROMPT=0
export MCODE_BUDGET_WARNING=0
export MCODE_READ_NUDGE=0
export LOOP_BUDGET=15

for i in $(seq 0 $((SHARD_COUNT - 1))); do
    echo "[baseline-s${i}] starting"
    python -m mcode bench swebench-lite \
        --backend '"'${BACKEND}'"' \
        --model '"'${MODEL}'"' \
        --loop-budget 15 \
        --timeout '"${TIMEOUT_S}"' \
        --namespace '"${NAMESPACE}"' \
        --limit ${LIMIT} \
        --shard-count ${SHARD_COUNT} \
        --shard-index ${i} \
        --db /tmp/ablation-baseline-shard${i}.db \
        > /tmp/ablation-baseline-shard${i}.log 2>&1 &
    sleep 30
done

echo "[baseline] All shards launched (staggered), waiting..."
wait
echo "=== Phase 1 complete (baseline done + images cached) ==="
date

# Copy baseline DBs to results
for i in $(seq 0 $((SHARD_COUNT - 1))); do
    cp /tmp/ablation-baseline-shard${i}.db '"${BV_RESULTS_DIR}"'/ablation-baseline-shard${i}.db 2>/dev/null || true
done

echo "=== Phase 2: Remaining experiments (images cached) ==="

run_experiment() {
    local EXP_NAME=$1
    shift
    for var in "$@"; do
        export "${var}"
    done
    for i in $(seq 0 $((SHARD_COUNT - 1))); do
        (
            echo "[${EXP_NAME}-s${i}] starting"
            python -m mcode bench swebench-lite \
                --backend '"'${BACKEND}'"' \
                --model '"'${MODEL}'"' \
                --loop-budget ${LOOP_BUDGET} \
                --timeout '"${TIMEOUT_S}"' \
                --namespace '"${NAMESPACE}"' \
                --limit ${LIMIT} \
                --shard-count ${SHARD_COUNT} \
                --shard-index ${i} \
                --db /tmp/ablation-${EXP_NAME}-shard${i}.db \
                > /tmp/ablation-${EXP_NAME}-shard${i}.log 2>&1
            echo "[${EXP_NAME}-s${i}] done (exit=$?)"
        ) &
    done
}

run_experiment "A-prompt" "MCODE_EXPLORE_PROMPT=1" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15"
run_experiment "B-warning" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=1" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15"
run_experiment "C-budget25" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=25"
run_experiment "D-nudge" "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=1" "LOOP_BUDGET=15"

echo "=== Phase 2: All experiments launched, waiting ==="
wait
echo "=== All experiments complete ==="
date

# Copy all DBs to results
for EXP in A-prompt B-warning C-budget25 D-nudge; do
    for i in $(seq 0 $((SHARD_COUNT - 1))); do
        cp /tmp/ablation-${EXP}-shard${i}.db '"${BV_RESULTS_DIR}"'/ablation-${EXP}-shard${i}.db 2>/dev/null || true
    done
done

# Merge shards per experiment
for EXP in baseline A-prompt B-warning C-budget25 D-nudge; do
    echo "Merging ${EXP}..."
    python -m mcode merge-shards --force \
        --out '"${BV_RESULTS_DIR}"'/ablation-${EXP}.db \
        '"${BV_RESULTS_DIR}"'/ablation-${EXP}-shard*.db 2>&1 || echo "  MERGE FAILED for ${EXP}"
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
echo "Phase 1: baseline (${SHARD_COUNT} shards, staggered 30s, pulls images)"
echo "Phase 2: 4 experiments x ${SHARD_COUNT} shards = $((4 * SHARD_COUNT)) parallel (images cached)"
echo ""
echo "Monitor:  bjobs -w | grep ablation"
echo "Progress: ssh \$(cat ${VLLM_HOST_FILE}) 'tail -f /tmp/ablation-baseline-shard0.log'"
echo "Results:  cat ${LOG_DIR}/ablation-mega.log | tail -20"
