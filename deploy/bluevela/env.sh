#!/usr/bin/env bash
# Blue Vela benchmark configuration

# Cluster
BV_LOGIN=skula@login3.bluevela.rmf.ibm.com
BV_HOME=/u/skula
BV_MCODE_DIR=${BV_HOME}/mcode
BV_RESULTS_DIR=${BV_MCODE_DIR}/results
BV_QUEUE=normal
BV_GROUP=grp_runtime

# Model
MODEL=Qwen/Qwen3.5-35B-A3B
VLLM_PORT=8000
VLLM_IMAGE=docker.io/vllm/vllm-openai:nightly
VLLM_GPU_COUNT=2
VLLM_MAX_MODEL_LEN=16384

# Benchmark
BENCHMARK=humaneval
BACKEND=openai
OPENAI_API_KEY=dummy
MCODE_MAX_NEW_TOKENS=1024
LOOP_BUDGET=3
TIMEOUT_S=60
STRATEGY=repair
SHARD_COUNT=4
SANDBOX=process

# Optional
# LIMIT=10
