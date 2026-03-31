#!/usr/bin/env bash
# Blue Vela benchmark configuration
# All values can be overridden via environment variables.

# Cluster
BV_LOGIN=${BV_LOGIN:-skula@login3.bluevela.rmf.ibm.com}
BV_HOME=${BV_HOME:-/u/skula}
BV_MCODE_DIR=${BV_MCODE_DIR:-${BV_HOME}/mcode}
BV_RESULTS_DIR=${BV_RESULTS_DIR:-${BV_MCODE_DIR}/results}
BV_QUEUE=${BV_QUEUE:-normal}
BV_GROUP=${BV_GROUP:-grp_runtime}

# Shared storage (NFS, visible to all nodes)
BV_SHARED_DIR=${BV_SHARED_DIR:-/proj/dmfexp/skula}
BV_PODMAN_ROOT=${BV_PODMAN_ROOT:-${BV_SHARED_DIR}/podman}

# Hugging Face auth and cache
BV_HF_ENV=${BV_HF_ENV:-${BV_HOME}/.config/mcode/hf-env.sh}
HF_HOME=${HF_HOME:-${BV_SHARED_DIR}/hf-cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
if [[ -f "${BV_HF_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${BV_HF_ENV}"
fi

# Model
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
VLLM_PORT=${VLLM_PORT:-8321}
VLLM_IMAGE=${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.17.0}
VLLM_GPU_COUNT=${VLLM_GPU_COUNT:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}

# Benchmark
BACKEND=${BACKEND:-openai}
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
MCODE_MAX_NEW_TOKENS=${MCODE_MAX_NEW_TOKENS:-4096}
LOOP_BUDGET=${LOOP_BUDGET:-15}
SHARD_COUNT=${SHARD_COUNT:-7}
