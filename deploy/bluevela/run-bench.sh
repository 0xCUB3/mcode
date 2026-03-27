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

echo "vLLM server: ${OPENAI_BASE_URL}"
echo "Benchmark:   ${BENCHMARK}"
echo "Model:       ${MODEL}"
echo "Shards:      ${SHARD_COUNT}"

LOG_DIR="${BV_MCODE_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

bsub -q "${BV_QUEUE}" \
  -G "${BV_GROUP}" \
  -J "bench[1-${SHARD_COUNT}]" \
  -n 4 \
  -R "span[hosts=1]" \
  -o "${LOG_DIR}/bench-%I.log" \
  -e "${LOG_DIR}/bench-%I.log" \
  bash -c '
    set -e

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
    export DOCKER_HOST=unix://${SOCK}

    trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT

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

    SHARD_INDEX=$((LSB_JOBINDEX - 1))

    uv run python -m mcode bench '"${BENCHMARK}"' \
      --model '"'${MODEL}'"' \
      --backend '"'${BACKEND}'"' \
      --loop-budget '"${LOOP_BUDGET}"' \
      --timeout '"${TIMEOUT_S}"' \
      --strategy '"${STRATEGY}"' \
      --sandbox '"${SANDBOX}"' \
      --shard-count '"${SHARD_COUNT}"' \
      --shard-index ${SHARD_INDEX} \
      --db '"${BV_RESULTS_DIR}/${BENCHMARK}"'-shard-${SHARD_INDEX}.db
  '

echo "Array job submitted. Monitor with: bjobs -J bench"
