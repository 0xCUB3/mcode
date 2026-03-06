#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

VLLM_HOST_FILE="${BV_MCODE_DIR}/vllm_host.txt"
VLLM_LOG="${BV_MCODE_DIR}/results/vllm.log"

rm -f "${VLLM_HOST_FILE}"
mkdir -p "$(dirname "${VLLM_LOG}")"
: > "${VLLM_LOG}"

echo "Submitting vLLM server job..."
bsub -q "${BV_QUEUE}" \
  -G "${BV_GROUP}" \
  -J "vllm-server" \
  -gpu "num=${VLLM_GPU_COUNT}" \
  -n 8 \
  -R "span[hosts=1]" \
  -o "${VLLM_LOG}" \
  -e "${VLLM_LOG}" \
  bash -c '
    export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u)
    mkdir -p ${XDG_RUNTIME_DIR}

    hostname > '"${VLLM_HOST_FILE}"'
    echo "vLLM starting on $(hostname):'"${VLLM_PORT}"'"
    nvidia-smi -L

    podman run --rm \
      --hooks-dir=/usr/share/containers/oci/hooks.d \
      --security-opt=label=disable \
      --ipc=host \
      --net=host \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
      '"${VLLM_IMAGE}"' \
      --model '"${MODEL}"' \
      --port '"${VLLM_PORT}"' \
      --trust-remote-code
  '

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
