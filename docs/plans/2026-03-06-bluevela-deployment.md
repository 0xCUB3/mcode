# Blue Vela Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy mcode benchmarking pipeline on IBM Blue Vela (LSF + H100 GPUs) with vLLM inference and sharded benchmark execution.

**Architecture:** Two-job pattern — a long-running vLLM server in podman on a GPU node, plus LSF array jobs running mcode benchmarks in a uv-managed virtualenv. Results fetched locally via rsync.

**Tech Stack:** LSF (bsub), podman, uv, vLLM, Python 3.11+

---

### Task 1: Delete OpenShift deployment

**Files:**
- Delete: `deploy/k8s/` (entire directory)

**Step 1: Remove the k8s directory**

```bash
rm -rf deploy/k8s/
```

**Step 2: Commit**

```bash
git add -A deploy/k8s/
git commit -m "remove OpenShift deployment (replaced by Blue Vela LSF)"
```

---

### Task 2: Create env.sh config

**Files:**
- Create: `deploy/bluevela/env.sh`

**Step 1: Create the file**

```bash
#!/usr/bin/env bash
# Blue Vela benchmark configuration

# Cluster
BV_LOGIN=skula@login3.bluevela.rmf.ibm.com
BV_HOME=/u/skula
BV_MCODE_DIR=${BV_HOME}/mcode
BV_RESULTS_DIR=${BV_MCODE_DIR}/results
BV_QUEUE=normal

# Model
MODEL=Qwen/Qwen3.5-35B-A3B
VLLM_PORT=8000
VLLM_IMAGE=vllm/vllm-openai:latest
VLLM_GPU_COUNT=1

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
```

**Step 2: Commit**

```bash
git add deploy/bluevela/env.sh
git commit -m "add Blue Vela env config"
```

---

### Task 3: Create setup.sh (one-time cluster setup)

**Files:**
- Create: `deploy/bluevela/setup.sh`

This script runs on the cluster. It installs uv (which manages Python 3.11+), clones the repo, creates a venv, and installs mcode.

**Step 1: Create the file**

```bash
#!/usr/bin/env bash
set -euo pipefail

MCODE_DIR="${BV_MCODE_DIR:-/u/skula/mcode}"
REPO_URL="${MCODE_REPO:-https://github.com/skula/mcode.git}"

echo "=== Installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

echo "=== Setting up mcode ==="
if [[ ! -d "${MCODE_DIR}" ]]; then
  git clone "${REPO_URL}" "${MCODE_DIR}"
fi
cd "${MCODE_DIR}"

echo "=== Creating virtualenv with Python 3.11 ==="
uv venv --python 3.11 venv
source venv/bin/activate

echo "=== Installing mcode ==="
uv pip install -e ".[evalplus,datasets]"

echo "=== Creating results directory ==="
mkdir -p results

echo "=== Done ==="
echo "Activate with: source ${MCODE_DIR}/venv/bin/activate"
```

**Step 2: Make executable and commit**

```bash
chmod +x deploy/bluevela/setup.sh
git add deploy/bluevela/setup.sh
git commit -m "add Blue Vela one-time setup script"
```

---

### Task 4: Create start-vllm.sh

**Files:**
- Create: `deploy/bluevela/start-vllm.sh`

Submits an LSF job that runs vLLM in podman. Writes the hostname to a file so the benchmark job knows where to connect.

**Step 1: Create the file**

```bash
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
```

**Step 2: Make executable and commit**

```bash
chmod +x deploy/bluevela/start-vllm.sh
git add deploy/bluevela/start-vllm.sh
git commit -m "add vLLM server launch script for Blue Vela"
```

---

### Task 5: Create run-bench.sh

**Files:**
- Create: `deploy/bluevela/run-bench.sh`

Submits an LSF array job. Each array task runs one shard of the benchmark.

**Step 1: Create the file**

```bash
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
```

**Step 2: Make executable and commit**

```bash
chmod +x deploy/bluevela/run-bench.sh
git add deploy/bluevela/run-bench.sh
git commit -m "add benchmark array job script for Blue Vela"
```

---

### Task 6: Create stop-vllm.sh

**Files:**
- Create: `deploy/bluevela/stop-vllm.sh`

**Step 1: Create the file**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Killing vLLM server job..."
bkill -J "vllm-server" 2>/dev/null && echo "Done." || echo "No running vLLM job found."
rm -f "${BV_MCODE_DIR}/vllm_host.txt"
```

**Step 2: Make executable and commit**

```bash
chmod +x deploy/bluevela/stop-vllm.sh
git add deploy/bluevela/stop-vllm.sh
git commit -m "add vLLM stop script for Blue Vela"
```

---

### Task 7: Create fetch-results.sh (runs locally)

**Files:**
- Create: `deploy/bluevela/fetch-results.sh`

**Step 1: Create the file**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

LOCAL_RESULTS="$(cd "${SCRIPT_DIR}/../.." && pwd)/results"
mkdir -p "${LOCAL_RESULTS}"

echo "Fetching results from Blue Vela..."
rsync -avz --progress \
  "${BV_LOGIN}:${BV_RESULTS_DIR}/*.db" \
  "${LOCAL_RESULTS}/"

echo "Results saved to: ${LOCAL_RESULTS}/"
ls -la "${LOCAL_RESULTS}"/*.db 2>/dev/null || echo "No .db files found."
```

**Step 2: Make executable and commit**

```bash
chmod +x deploy/bluevela/fetch-results.sh
git add deploy/bluevela/fetch-results.sh
git commit -m "add local result fetch script for Blue Vela"
```

---

### Task 8: Create README.md

**Files:**
- Create: `deploy/bluevela/README.md`

**Step 1: Create with usage instructions**

Cover: prerequisites, one-time setup, running a benchmark, fetching results, common LSF commands.

**Step 2: Commit**

```bash
git add deploy/bluevela/README.md
git commit -m "add Blue Vela deployment README"
```

---

### Task 9: Run setup on the cluster and validate

**Step 1: Push to remote so we can clone on Blue Vela**

```bash
git push
```

**Step 2: SSH in and run setup**

```bash
ssh skula@login3.bluevela.rmf.ibm.com
# copy setup.sh to cluster and run it, or clone repo first manually
```

**Step 3: Validate venv works**

```bash
source /u/skula/mcode/venv/bin/activate
mcode --help
```

**Step 4: Submit a test vLLM job and verify it starts**

**Step 5: Submit a small benchmark run (LIMIT=2) and verify results**
