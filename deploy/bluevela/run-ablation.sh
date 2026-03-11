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
echo "Podman root: ${BV_PODMAN_ROOT}"
echo ""

LOG_DIR="${BV_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Clean old ablation results
rm -f "${BV_RESULTS_DIR}"/ablation-*.db "${BV_RESULTS_DIR}"/ablation-*.log

ALL_MERGE_JOBS=""

# Experiment definitions: "name|env_vars"
EXPERIMENTS=(
    "baseline|MCODE_EXPLORE_PROMPT=0,MCODE_BUDGET_WARNING=0,MCODE_READ_NUDGE=0,LOOP_BUDGET=15"
    "A-prompt|MCODE_EXPLORE_PROMPT=1,MCODE_BUDGET_WARNING=0,MCODE_READ_NUDGE=0,LOOP_BUDGET=15"
    "B-warning|MCODE_EXPLORE_PROMPT=0,MCODE_BUDGET_WARNING=1,MCODE_READ_NUDGE=0,LOOP_BUDGET=15"
    "C-budget25|MCODE_EXPLORE_PROMPT=0,MCODE_BUDGET_WARNING=0,MCODE_READ_NUDGE=0,LOOP_BUDGET=25"
    "D-nudge|MCODE_EXPLORE_PROMPT=0,MCODE_BUDGET_WARNING=0,MCODE_READ_NUDGE=1,LOOP_BUDGET=15"
)

for EXPDEF in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_NAME ENV_STR <<< "${EXPDEF}"
    echo "--- Experiment: ${EXP_NAME} ---"

    # Build export statements from comma-separated env vars
    ENV_EXPORTS=""
    IFS=',' read -ra ENVPAIRS <<< "${ENV_STR}"
    for pair in "${ENVPAIRS[@]}"; do
        ENV_EXPORTS="export ${pair}; ${ENV_EXPORTS}"
    done

    # Submit shard array job (1-based LSF index, 0-based shard index)
    SHARD_JOBS=""
    JOB_PREFIX="abl-${EXP_NAME}"

    bsub -G "${BV_GROUP}" \
        -J "${JOB_PREFIX}[1-${SHARD_COUNT}]" \
        -q "${BV_QUEUE}" \
        -n 4 \
        -R "span[hosts=1]" \
        -R "rusage[mem=16000]" \
        -W 8:00 \
        -o "${LOG_DIR}/ablation-${EXP_NAME}-%I.log" \
        -e "${LOG_DIR}/ablation-${EXP_NAME}-%I.log" \
        bash -c '
            '"${ENV_EXPORTS}"'

            # Shared podman storage (NFS, images pulled once)
            PD='"${BV_PODMAN_ROOT}"'
            mkdir -p ${PD}/graphroot ${PD}/runroot

            # Per-shard podman socket (avoid conflicts)
            export XDG_RUNTIME_DIR=/tmp/podman-$(id -u)-'"${EXP_NAME}"'-${LSB_JOBINDEX}
            mkdir -p ${XDG_RUNTIME_DIR}
            SOCK=${XDG_RUNTIME_DIR}/podman.sock
            rm -f ${SOCK}
            podman --cgroup-manager=cgroupfs --storage-driver=overlay \
                --root=${PD}/graphroot --runroot=${PD}/runroot \
                system service --time=0 unix://${SOCK} &
            PODMAN_PID=$!
            sleep 3

            source '"${BV_MCODE_DIR}"'/venv/bin/activate
            export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
            export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
            export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
            export MCODE_CONTEXT_WINDOW='"'${VLLM_MAX_MODEL_LEN}'"'
            export MCODE_REACT_TIMEOUT=450
            export MCODE_KEEP_IMAGES=1
            export DOCKER_HOST=unix://${SOCK}

            SHARD_INDEX=$((LSB_JOBINDEX - 1))

            echo "['"${EXP_NAME}"'-s${SHARD_INDEX}] starting on $(hostname)"
            date

            python -m mcode bench swebench-lite \
                --backend '"'${BACKEND}'"' \
                --model '"'${MODEL}'"' \
                --loop-budget ${LOOP_BUDGET} \
                --timeout '"${TIMEOUT_S}"' \
                --namespace '"${NAMESPACE}"' \
                --limit '"${LIMIT}"' \
                --shard-count '"${SHARD_COUNT}"' \
                --shard-index ${SHARD_INDEX} \
                --db '"${BV_RESULTS_DIR}"'/ablation-'"${EXP_NAME}"'-shard${SHARD_INDEX}.db

            echo "['"${EXP_NAME}"'-s${SHARD_INDEX}] done (exit=$?)"
            date
            kill ${PODMAN_PID} 2>/dev/null || true
        '

    echo "  -> ${SHARD_COUNT} shards submitted"

    # Merge job: runs after all shards finish
    MERGE_NAME="abl-merge-${EXP_NAME}"
    bsub -G "${BV_GROUP}" \
        -J "${MERGE_NAME}" \
        -q "${BV_QUEUE}" \
        -n 1 \
        -W 0:10 \
        -w "done(\"${JOB_PREFIX}\")" \
        -o "${LOG_DIR}/ablation-${EXP_NAME}-merge.log" \
        -e "${LOG_DIR}/ablation-${EXP_NAME}-merge.log" \
        bash -c '
            source '"${BV_MCODE_DIR}"'/venv/bin/activate
            cd '"${BV_MCODE_DIR}"'
            python -m mcode merge-shards --force \
                --out '"${BV_RESULTS_DIR}"'/ablation-'"${EXP_NAME}"'.db \
                '"${BV_RESULTS_DIR}"'/ablation-'"${EXP_NAME}"'-shard*.db
            echo "MERGE_DONE: '"${EXP_NAME}"'"
        '

    if [[ -z "${ALL_MERGE_JOBS}" ]]; then
        ALL_MERGE_JOBS="done(\"${MERGE_NAME}\")"
    else
        ALL_MERGE_JOBS="${ALL_MERGE_JOBS} && done(\"${MERGE_NAME}\")"
    fi
done

# Summary job: runs after all merges
bsub -G "${BV_GROUP}" \
    -J "abl-summary" \
    -q "${BV_QUEUE}" \
    -n 1 \
    -W 0:10 \
    -w "${ALL_MERGE_JOBS}" \
    -o "${LOG_DIR}/ablation-summary.log" \
    -e "${LOG_DIR}/ablation-summary.log" \
    bash -c '
        source '"${BV_MCODE_DIR}"'/venv/bin/activate
        cd '"${BV_MCODE_DIR}"'
        python3 << PYEOF
import sqlite3, glob
print("=== ABLATION RESULTS ===")
print(f"Model: '"${MODEL}"'")
print()
for db in sorted(glob.glob("results/ablation-*.db")):
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
echo "=== Ablation suite submitted ==="
echo "5 experiments x ${SHARD_COUNT} shards = $((5 * SHARD_COUNT)) shard jobs"
echo "+ 5 merge jobs + 1 summary job = $((5 * SHARD_COUNT + 6)) total"
echo ""
echo "Monitor:  bjobs -w | grep abl"
echo "Results:  cat ${LOG_DIR}/ablation-summary.log"
