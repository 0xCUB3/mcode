#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

VLLM_HOST_FILE="${BV_MCODE_DIR}/vllm_host.txt"
VLLM_LOG="${BV_MCODE_DIR}/results/vllm.log"

mkdir -p "$(dirname "${VLLM_LOG}")"

echo "Submitting vLLM server job..."
bsub -q "${BV_QUEUE}" \
  -J "vllm-server" \
  -gpu "num=${VLLM_GPU_COUNT}:mode=shared:j_exclusive=yes" \
  -n 8 \
  -R "span[hosts=1]" \
  -o "${VLLM_LOG}" \
  -e "${VLLM_LOG}" \
  bash -c "
    hostname > ${VLLM_HOST_FILE}
    echo \"vLLM starting on \$(hostname):${VLLM_PORT}\"
    podman run --rm \
      --device nvidia.com/gpu=all \
      --ipc=host \
      -p ${VLLM_PORT}:8000 \
      -v \${HOME}/.cache/huggingface:/root/.cache/huggingface \
      ${VLLM_IMAGE} \
      --model ${MODEL} \
      --port 8000 \
      --trust-remote-code
  "

echo "Job submitted. Waiting for it to start..."
for i in $(seq 1 60); do
  if [[ -f "${VLLM_HOST_FILE}" ]]; then
    VLLM_HOST="$(cat "${VLLM_HOST_FILE}")"
    echo "vLLM server starting on: ${VLLM_HOST}:${VLLM_PORT}"
    echo "Check logs: tail -f ${VLLM_LOG}"
    echo ""
    echo "Once the model is loaded, run: ./run-bench.sh"
    exit 0
  fi
  sleep 5
done

echo "WARNING: vLLM job hasn't started after 5 minutes. Check 'bjobs' for status."
