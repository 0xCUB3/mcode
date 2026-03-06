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

LAST_INDEX=$((SHARD_COUNT - 1))
LOG_DIR="${BV_MCODE_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

bsub -q "${BV_QUEUE}" \
  -J "bench[0-${LAST_INDEX}]" \
  -n 4 \
  -R "span[hosts=1]" \
  -o "${LOG_DIR}/bench-%I.log" \
  -e "${LOG_DIR}/bench-%I.log" \
  bash -c "
    source ${BV_MCODE_DIR}/venv/bin/activate
    export OPENAI_BASE_URL='${OPENAI_BASE_URL}'
    export OPENAI_API_KEY='${OPENAI_API_KEY}'
    export MCODE_MAX_NEW_TOKENS='${MCODE_MAX_NEW_TOKENS}'

    SHARD_INDEX=\$((LSB_JOBINDEX))

    mcode bench ${BENCHMARK} \
      --model '${MODEL}' \
      --backend '${BACKEND}' \
      --loop-budget ${LOOP_BUDGET} \
      --timeout ${TIMEOUT_S} \
      --strategy ${STRATEGY} \
      --sandbox ${SANDBOX} \
      --shard-count ${SHARD_COUNT} \
      --shard-index \${SHARD_INDEX} \
      --db '${BV_RESULTS_DIR}/${BENCHMARK}-shard-\${SHARD_INDEX}.db'
  "

echo "Array job submitted. Monitor with: bjobs -J bench"
