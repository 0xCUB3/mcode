# mCode command cookbook

## Install / run

Pick one:

### Option A: virtualenv (repo checkout)

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

### Option B: `uv tool` (puts `mcode` on your PATH)

```bash
uv tool install -e .
uv tool update-shell
# restart your shell
```

Quick sanity check:

```bash
mcode --help
mcode bench --help
mcode results --help

# (If you didn’t activate your venv)
uv run mcode --help
```

## Run benchmarks locally

By default we run untrusted code in a Docker sandbox (`--sandbox docker`, network disabled). Make sure Docker Desktop
is running. If you can’t use Docker, `--sandbox process` runs code directly on your machine (unsafe).

```bash
# HumanEval
mcode bench humaneval --model granite3.3:8b --samples 1

# MBPP
mcode bench mbpp --model granite3.3:8b --samples 1

# Smoke test: first N tasks only
mcode bench humaneval --model granite3.3:8b --limit 10
```

### Useful flags

- `--samples`: attempts per task (stops early on the first passing attempt)
- `--debug-iters`: “fix” attempts after a failure (per sample)
- `--timeout`: seconds per code execution attempt
- `--limit`: run first N tasks only (smoke tests)
- `--sandbox`: `docker` (default) or `process` (unsafe)
- `--shard-count/--shard-index`: split the task list across parallel runs
- `--db`: SQLite DB path (use a unique DB per shard if running shards concurrently)

### Parallel sharding locally

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

### SWE-bench Lite (optional / heavy)

SWE-bench Lite is image-based and way heavier than HumanEval/MBPP.

First:

```bash
uv pip install -e '.[swebench]'
```

#### OpenShift (x86_64): single-instance Pods (no Docker-in-Docker)

Uses prebuilt `swebench/sweb.eval.x86_64.*` images and runs one Pod per instance:

```bash
# smoke test (gold patch)
MODE=gold ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590

# model-run (initContainer generates patch via Mellea)
MODE=model ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
```

What those mean:

- `MODE=gold`: apply the dataset’s “gold” patch for that instance (ground-truth fix). This is mainly a smoke test that
  the eval image runs correctly on your cluster.
- `MODE=model`: `mcode` generates a patch via Mellea, then the eval image applies it and runs tests. Expect low pass
  rates without retrieval / repo context.
- `sympy__sympy-20590`: a SWE-bench Lite `instance_id`. The prefix is the GitHub repo (`sympy/sympy`), and the suffix
  is an issue/PR-style number used by SWE-bench as an ID. Swap this string to run a different instance.

#### OpenShift (x86_64): batch runs + SQLite output (recommended)

`run-swebench-lite.sh` runs many instances by launching **one Pod per instance** and saves:

- `*.eval.log` (eval output), `*.gen.log` (patch generation output, `MODE=model` only), and `*.result.json` per instance
- one SQLite DB you can query with `mcode results`

Examples:

```bash
# Gold-patch smoke test (fast sanity check)
MODE=gold LIMIT=5 PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh

# Model-run via Ollama
MODE=model LIMIT=5 PARALLELISM=2 \
  BACKEND=ollama MODEL=granite3-dense:8b \
  ./deploy/k8s/run-swebench-lite.sh

# Pick specific instances
MODE=gold PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh \
  sympy__sympy-20590 astropy__astropy-12907
```

After it finishes, query the DB it prints:

```bash
uv run mcode results --db experiments/results/swebench-lite-*.db --benchmark swebench-lite
```

#### Local Docker + cluster inference (optional)

Run SWE-bench Lite locally (Docker Desktop), but point Mellea at a cluster service via `oc port-forward`:

```bash
# vLLM (OpenAI-compatible)
oc port-forward svc/vllm 8000:8000
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
mcode bench swebench-lite --backend openai --model ibm-granite/granite-3.0-8b-instruct --limit 5

# (or) Ollama
oc port-forward svc/ollama 11434:11434
export OLLAMA_HOST=http://127.0.0.1:11434
mcode bench swebench-lite --backend ollama --model granite3-dense:8b --limit 5
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

## OpenShift jobs (recommended for big HumanEval/MBPP runs)

Before you start:

- You’re logged in with `oc` and already on the right project/namespace.
- There’s a `BuildConfig` named `mcode` if you want to build with `oc start-build`.

### Build + push the `mcode` image (OpenShift internal registry)

```bash
oc start-build mcode --from-dir=. --follow
```

### Configure a benchmark run

Edit `deploy/k8s/bench.env` (it gets injected into the Job as env vars).

The main knobs:

- `BENCHMARK`: `humaneval` or `mbpp`
- `SAMPLES`, `DEBUG_ITERS`, `TIMEOUT_S`
- `SHARD_COUNT`: total shards (must match Job completions)
- Backend selection:
  - vLLM (OpenAI-compatible): `BACKEND=openai` + `OPENAI_BASE_URL=http://vllm:8000/v1`
  - Ollama: `BACKEND=ollama` + `OLLAMA_HOST=http://ollama:11434`

Guardrails (worth keeping on):

- `MCODE_MAX_NEW_TOKENS`: caps generation length (avoids rare runaway outputs)
- `MCODE_SANDBOX_MAX_OUTPUT_BYTES`: caps captured stdout/stderr (helps avoid OOMs in `--sandbox process`)

### Submit the Job

```bash
PARALLELISM=4 ./deploy/k8s/run-bench.sh
```

Copy per-shard SQLite DBs back locally after the Job completes:

```bash
PARALLELISM=4 FETCH_RESULTS=1 ./deploy/k8s/run-bench.sh
```

Quick one-off overrides (no editing `bench.env`):

```bash
OVERRIDE_BENCHMARK=mbpp \
OVERRIDE_LIMIT=50 \
OVERRIDE_SHARD_COUNT=5 \
PARALLELISM=4 \
./deploy/k8s/run-bench.sh
```

Switch backends quickly (keep everything else the same):

```bash
# vLLM (OpenAI-compatible)
OVERRIDE_BACKEND=openai \
OVERRIDE_MODEL=ibm-granite/granite-3.0-8b-instruct \
OVERRIDE_OPENAI_BASE_URL=http://vllm:8000/v1 \
OVERRIDE_OPENAI_API_KEY=dummy \
./deploy/k8s/run-bench.sh

# Ollama
OVERRIDE_BACKEND=ollama \
OVERRIDE_MODEL=granite3-dense:8b \
OVERRIDE_OLLAMA_HOST=http://ollama:11434 \
./deploy/k8s/run-bench.sh
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
