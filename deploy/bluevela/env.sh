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

# Model
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
VLLM_PORT=${VLLM_PORT:-8321}
VLLM_IMAGE=${VLLM_IMAGE:-docker.io/vllm/vllm-openai:nightly}
VLLM_GPU_COUNT=${VLLM_GPU_COUNT:-2}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

# Benchmark
BENCHMARK=${BENCHMARK:-humaneval}
BACKEND=${BACKEND:-openai}
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
MCODE_MAX_NEW_TOKENS=${MCODE_MAX_NEW_TOKENS:-1024}
LOOP_BUDGET=${LOOP_BUDGET:-15}
TIMEOUT_S=${TIMEOUT_S:-60}
STRATEGY=${STRATEGY:-repair}
SHARD_COUNT=${SHARD_COUNT:-4}
SANDBOX=${SANDBOX:-process}
