# mCode command cookbook

This is a “do the thing” reference for `mcode` (local runs + OpenShift/Kubernetes jobs).

## Local setup

Create a virtualenv and install:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Or install as a global tool:

```bash
uv tool install -e .
uv tool update-shell
# restart your shell
```

Check the CLI:

```bash
mcode --help
mcode bench --help
mcode results --help
```

If you haven’t installed `mcode` into your environment, you can still run it via:

```bash
uv run mcode --help
```

## Run benchmarks locally

### HumanEval

```bash
mcode bench humaneval --model granite3.3:8b --samples 1
```

### MBPP

```bash
mcode bench mbpp --model granite3.3:8b --samples 1
```

### Quick smoke test (first N tasks only)

```bash
mcode bench humaneval --model granite3.3:8b --limit 10
```

### SWE-bench Lite (optional)

```bash
uv pip install -e '.[swebench]'
mcode bench swebench-lite --model granite3.3:8b --limit 5
```

Note: SWE-bench Lite is Docker/image-based and heavier than HumanEval/MBPP.

Workable approaches:

1) **OpenShift (x86_64) single-instance Pods** (no Docker-in-Docker): use prebuilt `swebench/sweb.eval.x86_64.*`
images and run one Pod per instance:

```bash
# smoke test (gold patch)
MODE=gold ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590

# model-run (initContainer generates patch via Mellea)
MODE=model ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
```

2) **Local Docker + cluster inference**: run SWE-bench Lite locally (with Docker Desktop) but point Mellea at your
cluster inference service via `oc port-forward`:

```bash
# vLLM (OpenAI-compatible)
oc port-forward svc/vllm 8000:8000
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
mcode bench swebench-lite --backend openai --model ibm-granite/granite-3.0-8b-instruct --namespace "" --limit 5

# (or) Ollama
oc port-forward svc/ollama 11434:11434
export OLLAMA_HOST=http://127.0.0.1:11434
mcode bench swebench-lite --backend ollama --model granite3-dense:8b --namespace "" --limit 5
```

### Key knobs

- `--samples`: attempts per task (stops early on the first passing attempt)
- `--debug-iters`: “fix” attempts after a failure (per sample)
- `--timeout`: seconds per code execution attempt
- `--sandbox`:
  - `docker` (default): safe-ish local sandbox (network disabled)
  - `process`: runs code directly (use only in locked-down containers)
- `--shard-count/--shard-index`: split the task list across multiple runs (parallelism)
- `--db`: SQLite DB path (use a unique DB per shard if running shards concurrently)

### Parallel sharding locally

Run 10 shards in parallel (example):

```bash
mkdir -p results
for i in $(seq 0 9); do
  mcode bench humaneval --model granite3.3:8b \
    --samples 10 \
    --shard-count 10 --shard-index "$i" \
    --db "results/humaneval-shard-$i.db" &
done
wait
```

## Query results (SQLite)

Show runs:

```bash
mcode results --db experiments/results/results.db --benchmark humaneval
```

Filter:

```bash
mcode results --db experiments/results/results.db --benchmark humaneval --model granite3.3:8b --samples 10
```

Compare across sample counts:

```bash
mcode results --db experiments/results/results.db --benchmark humaneval --model granite3.3:8b --compare-samples
```

## OpenShift / Kubernetes jobs (recommended for large runs)

Prereqs (generic):

- You’re already logged in with `oc` and on the right project/namespace.
- Your project has a `BuildConfig` named `mcode` if you want to build via `oc start-build`.

### Build + push the `mcode` image (OpenShift internal registry)

From the repo root:

```bash
oc start-build mcode --from-dir=. --follow
```

### Configure a benchmark run

Edit `deploy/k8s/bench.env` (these values are injected into the Job as env vars).

Important keys:

- `BENCHMARK`: `humaneval` or `mbpp`
- `SAMPLES`, `DEBUG_ITERS`, `TIMEOUT_S`
- `SHARD_COUNT`: total shards (must match Job completions)
- Backend selection:
  - vLLM (recommended on this cluster): `BACKEND=openai` + `OPENAI_BASE_URL=http://vllm:8000/v1`
  - Ollama: `BACKEND=ollama` + `OLLAMA_HOST=http://ollama:11434`

Guardrails (highly recommended):

- `MCODE_MAX_NEW_TOKENS`: caps generation length (prevents rare runaway outputs)
- `MCODE_SANDBOX_MAX_OUTPUT_BYTES`: caps stdout/stderr captured from test runs (prevents OOM in `--sandbox process`)

### Submit the Job

Submit with parallel shards:

```bash
PARALLELISM=4 ./deploy/k8s/run-bench.sh
```

Copy per-shard SQLite DBs back locally after the Job completes:

```bash
PARALLELISM=4 FETCH_RESULTS=1 ./deploy/k8s/run-bench.sh
```

Quick one-off overrides (without editing `bench.env`):

```bash
OVERRIDE_BENCHMARK=mbpp OVERRIDE_LIMIT=50 OVERRIDE_SHARD_COUNT=5 PARALLELISM=4 ./deploy/k8s/run-bench.sh
```

Switch backends quickly (same benchmark/shards):

```bash
# vLLM (OpenAI-compatible)
OVERRIDE_BACKEND=openai OVERRIDE_MODEL=ibm-granite/granite-3.0-8b-instruct OVERRIDE_OPENAI_BASE_URL=http://vllm:8000/v1 OVERRIDE_OPENAI_API_KEY=dummy ./deploy/k8s/run-bench.sh

# Ollama
OVERRIDE_BACKEND=ollama OVERRIDE_MODEL=granite3-dense:8b OVERRIDE_OLLAMA_HOST=http://ollama:11434 ./deploy/k8s/run-bench.sh
```

If a prior run is still active and you want to replace it:

```bash
FORCE_RECREATE=1 ./deploy/k8s/run-bench.sh
```

### Watch / debug

```bash
oc get job mcode-bench -o wide
oc get pods -l job-name=mcode-bench -w
oc logs -l job-name=mcode-bench --tail=200
oc get events --sort-by=.lastTimestamp | tail -n 30
oc describe resourcequota
```

### Fetch results later

If you didn’t use `FETCH_RESULTS=1`:

```bash
./deploy/k8s/fetch-results.sh mcode-bench
```
