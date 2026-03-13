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

echo "=== SWE-bench Live Lite ==="
echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Model:       ${MODEL}"
echo "Shards:      ${SHARD_COUNT}"
echo "Tasks:       ${LIMIT}"
echo ""

LOG_DIR="${BV_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

rm -f "${BV_RESULTS_DIR}"/live-*.db "${LOG_DIR}"/live-*.log

EXEC_HOST_FLAG=""
if [[ -n "${BV_EXEC_HOST:-}" ]]; then
    EXEC_HOST_FLAG="-m ${BV_EXEC_HOST}"
fi
bsub -G "${BV_GROUP}" \
    -J "live-mega" \
    -q "${BV_QUEUE}" \
    -n 1 \
    ${EXEC_HOST_FLAG} \
    -R "span[hosts=1]" \
    -W 24:00 \
    -o "${LOG_DIR}/live-mega.log" \
    -e "${LOG_DIR}/live-mega.log" \
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
export MCODE_REACT_TIMEOUT=450
export MCODE_KEEP_IMAGES=1

SHARD_COUNT='"${SHARD_COUNT}"'
LIMIT='"${LIMIT}"'
LOG_DIR='"${LOG_DIR}"'
RESULTS_DIR='"${BV_RESULTS_DIR}"'

echo "=== Phase 0: Pre-pull SWE-bench Live Lite images ==="
date
python3 << PYEOF
import docker
from datasets import load_dataset

ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split="lite")
tasks = list(ds)[:${LIMIT}]
print(f"Pre-pulling images for {len(tasks)} tasks...")

client = docker.from_env()

# Build local tag cache
local_tags = set()
for img in client.images.list():
    for tag in img.tags or []:
        local_tags.add(tag)
        if tag.startswith("docker.io/"):
            local_tags.add(tag[len("docker.io/"):])
        else:
            local_tags.add(f"docker.io/{tag}")

pulled = 0
failed = 0
cached = 0
for i, task in enumerate(tasks):
    iid = task["instance_id"]
    sanitized = iid.replace("__", "_1776_").lower()
    name = f"docker.io/starryzhang/sweb.eval.x86_64.{sanitized}"
    if name in local_tags:
        cached += 1
        continue
    try:
        print(f"  [{i+1}/{len(tasks)}] pulling {name}...", flush=True)
        for line in client.api.pull(name, stream=True, decode=True):
            if "error" in line:
                raise RuntimeError(line["error"])
        pulled += 1
    except Exception as e:
        print(f"  [{i+1}/{len(tasks)}] FAILED {name}: {e}", flush=True)
        failed += 1

print(f"Pre-pull done: {pulled} pulled, {failed} failed, {cached} cached")
PYEOF
echo "=== Phase 0 complete ==="
date

run_shard() {
    local EXP_NAME=$1
    local SHARD_IDX=$2
    shift 2

    for var in "$@"; do
        export "${var}"
    done

    local LOCAL_DB=/tmp/live-${EXP_NAME}-shard${SHARD_IDX}.db
    echo "[${EXP_NAME}-s${SHARD_IDX}] starting on $(hostname)" >> ${LOG_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.log

    python -m mcode bench swebench-live \
        --backend '"'${BACKEND}'"' \
        --model '"'${MODEL}'"' \
        --split lite \
        --loop-budget ${LOOP_BUDGET} \
        --timeout '"${TIMEOUT_S}"' \
        --limit ${LIMIT} \
        --shard-count ${SHARD_COUNT} \
        --shard-index ${SHARD_IDX} \
        --db ${LOCAL_DB} \
        >> ${LOG_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.log 2>&1
    local RC=$?

    cp ${LOCAL_DB} ${RESULTS_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.db 2>/dev/null || true
    echo "[${EXP_NAME}-s${SHARD_IDX}] done (exit=${RC})" >> ${LOG_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.log
    date >> ${LOG_DIR}/live-${EXP_NAME}-shard${SHARD_IDX}.log
}

echo "=== Phase 1: 3 experiments x ${SHARD_COUNT} shards in parallel ==="
date

for i in $(seq 0 $((SHARD_COUNT - 1))); do
    run_shard "baseline" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
    run_shard "B-warning" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=1" "MCODE_READ_NUDGE=0" "LOOP_BUDGET=15" &
    run_shard "D-nudge" $i "MCODE_EXPLORE_PROMPT=0" "MCODE_BUDGET_WARNING=0" "MCODE_READ_NUDGE=1" "LOOP_BUDGET=15" &
done

echo "All $((3 * SHARD_COUNT)) shards launched, waiting..."
wait
echo "=== All experiments complete ==="
date

# Merge shards per experiment
for EXP in baseline B-warning D-nudge; do
    echo "Merging ${EXP}..."
    python -m mcode merge-shards --force \
        --out ${RESULTS_DIR}/live-${EXP}.db \
        ${RESULTS_DIR}/live-${EXP}-shard*.db 2>&1 || echo "  MERGE FAILED for ${EXP}"
done

# Print summary
echo ""
echo "=== LIVE RESULTS ==="
echo "Model: '"${MODEL}"'"
echo ""
python3 << PYEOF
import sqlite3, glob
for db in sorted(glob.glob("'"${BV_RESULTS_DIR}"'/live-*.db")):
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
echo "=== Live mega job submitted ==="
echo "Phase 0: Sequential pre-pull of ~300 Docker images"
echo "Phase 1: 3 experiments x ${SHARD_COUNT} shards = $((3 * SHARD_COUNT)) parallel"
echo ""
echo "Monitor:  bjobs -w | grep live"
echo "Logs:     ls ${LOG_DIR}/live-*.log"
echo "Results:  tail -30 ${LOG_DIR}/live-mega.log"
