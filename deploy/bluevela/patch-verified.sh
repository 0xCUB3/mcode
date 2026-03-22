#!/usr/bin/env bash
set -e
LOG_DIR=/u/skula/mcode/results/logs
RESULTS_DIR=/u/skula/mcode/results
VLLM_HOST=$(cat /u/skula/mcode/vllm_host.txt)
PD_RUN=/tmp/podman-mcode-$(id -u)
PD_GRAPH=/proj/dmfexp/skula/podman2/graphroot
rm -rf ${PD_RUN} 2>/dev/null || true
mkdir -p ${PD_GRAPH} ${PD_RUN}/runroot
export XDG_RUNTIME_DIR=${PD_RUN}
SOCK=${PD_RUN}/podman.sock
podman --cgroup-manager=cgroupfs --storage-driver=overlay --storage-opt ignore_chown_errors=true \
    --root=${PD_GRAPH} --runroot=${PD_RUN}/runroot system service --time=0 unix://${SOCK} &
PODMAN_PID=$!
sleep 3
export DOCKER_HOST=unix://${SOCK}
trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT

podman --root=${PD_GRAPH} --runroot=${PD_RUN}/runroot container prune -f 2>/dev/null || true

source /u/skula/mcode/venv/bin/activate
export OPENAI_BASE_URL="http://${VLLM_HOST}:8321/v1"
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=450
export MCODE_KEEP_IMAGES=1
export MCODE_EXPLORE_PROMPT=0
export MCODE_BUDGET_WARNING=1
export MCODE_MID_NUDGE=1
export MCODE_READ_NUDGE=0
export MELLEA_TEXT_TOOLS=1
export MELLEA_BASH_TOOL=1

echo "=== Rerunning 45 missing tasks ==="
date
python -m mcode bench swebench-lite --backend openai --model MiniMaxAI/MiniMax-M2.5 \
    --dataset princeton-nlp/SWE-bench_Verified \
    --loop-budget 15 --timeout 300 \
    --task-ids /proj/dmfexp/skula/missing_verified_ids.txt \
    --db ${RESULTS_DIR}/live-m25-verified-patch.db

echo "=== Patching into main DB ==="
python3 /proj/dmfexp/skula/patch_merge.py
echo "=== Done ==="
date
