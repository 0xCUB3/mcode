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

# SWE-bench Live config
SWB_SPLIT=${SWB_SPLIT:-verified}
SWB_TIMEOUT=${SWB_TIMEOUT:-1800}
SWB_MEM_LIMIT=${SWB_MEM_LIMIT:-4g}
SWB_PIDS_LIMIT=${SWB_PIDS_LIMIT:-512}
SWB_LIMIT=${SWB_LIMIT:-}

echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Benchmark:   swebench-live (${SWB_SPLIT})"
echo "Model:       ${MODEL}"
echo "Shards:      ${SHARD_COUNT}"

LOG_DIR="${BV_MCODE_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

LIMIT_FLAG=""
if [[ -n "${SWB_LIMIT}" ]]; then
  LIMIT_FLAG="--limit ${SWB_LIMIT}"
fi

bsub -q "${BV_QUEUE}" \
  -G "${BV_GROUP}" \
  -J "swb-live[1-${SHARD_COUNT}]" \
  -n 8 \
  -R "span[hosts=1]" \
  -R "rusage[mem=16000]" \
  -o "${LOG_DIR}/swb-live-%I.log" \
  -e "${LOG_DIR}/swb-live-%I.log" \
  bash -c '
    PD='"${BV_PODMAN_ROOT}"'
    mkdir -p ${PD}/graphroot ${PD}/runroot

    export XDG_RUNTIME_DIR=/tmp/podman-$(id -u)-swb-${LSB_JOBINDEX}
    mkdir -p ${XDG_RUNTIME_DIR}
    SOCK=${XDG_RUNTIME_DIR}/podman.sock
    rm -f ${SOCK}
    podman --cgroup-manager=cgroupfs --storage-driver=overlay \
        --root=${PD}/graphroot --runroot=${PD}/runroot \
        system service --time=0 unix://${SOCK} &
    PODMAN_PID=$!
    sleep 2

    source '"${BV_MCODE_DIR}"'/venv/bin/activate
    export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
    export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
    export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
    export DOCKER_HOST="unix://${SOCK}"

    SHARD_INDEX=$((LSB_JOBINDEX - 1))

    mcode bench swebench-live \
      --model '"'${MODEL}'"' \
      --backend '"'${BACKEND}'"' \
      --loop-budget '"${LOOP_BUDGET}"' \
      --timeout '"${SWB_TIMEOUT}"' \
      --strategy '"${STRATEGY}"' \
      --split '"${SWB_SPLIT}"' \
      --mem-limit '"${SWB_MEM_LIMIT}"' \
      --pids-limit '"${SWB_PIDS_LIMIT}"' \
      --shard-count '"${SHARD_COUNT}"' \
      --shard-index ${SHARD_INDEX} \
      --db '"${BV_RESULTS_DIR}"'/swebench-live-shard-${SHARD_INDEX}.db \
      '"${LIMIT_FLAG}"'

    kill ${PODMAN_PID} 2>/dev/null || true
  '

echo "Array job submitted. Monitor with: bjobs -J swb-live"
