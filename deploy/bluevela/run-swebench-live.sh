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
    set -e

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
    export DOCKER_HOST="unix://${SOCK}"

    trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT

    cd '"${BV_MCODE_DIR}"'

    for attempt in $(seq 1 30); do
      if uv run python - <<'"'"'PYEOF'"'"' >/dev/null 2>&1; then
import docker
client = docker.from_env()
client.ping()
PYEOF
        break
      fi
      sleep 1
      if [[ ${attempt} -eq 30 ]]; then
        echo "Docker socket did not become ready" >&2
        exit 1
      fi
    done

    export OPENAI_BASE_URL='"'${OPENAI_BASE_URL}'"'
    export OPENAI_API_KEY='"'${OPENAI_API_KEY}'"'
    export MCODE_MAX_NEW_TOKENS='"'${MCODE_MAX_NEW_TOKENS}'"'
    export HF_HOME='"'${HF_HOME}'"'
    export HF_HUB_CACHE='"'${HF_HUB_CACHE}'"'
    export HF_DATASETS_CACHE='"'${HF_DATASETS_CACHE}'"'
    if [[ -n '"'${HF_TOKEN:-}'"' ]]; then
      export HF_TOKEN='"'${HF_TOKEN}'"'
      export HUGGINGFACE_HUB_TOKEN='"'${HF_TOKEN}'"'
    fi

    SHARD_INDEX=$((LSB_JOBINDEX - 1))

    uv run mcode bench swebench-live \
      --model '"'${MODEL}'"' \
      --backend '"'${BACKEND}'"' \
      --loop-budget '"'${LOOP_BUDGET}'"' \
      --timeout '"'${SWB_TIMEOUT}'"' \
      --split '"'${SWB_SPLIT}'"' \
      --mem-limit '"'${SWB_MEM_LIMIT}'"' \
      --pids-limit '"'${SWB_PIDS_LIMIT}'"' \
      --shard-count '"'${SHARD_COUNT}'"' \
      --shard-index ${SHARD_INDEX} \
      --db '"'${BV_RESULTS_DIR}'"'/swebench-live-shard-${SHARD_INDEX}.db \
      '"'${LIMIT_FLAG}'"'
  '

echo "Array job submitted. Monitor with: bjobs -J swb-live"
