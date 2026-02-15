# OpenShift / Kubernetes: `mcode` cookbook

This page is only about running `mcode` on OpenShift/Kubernetes (no local Docker instructions).

## Prereqs (quick checklist)

- Logged in with `oc` and on the right namespace: `oc project -q`
- Your model backend is reachable from the namespace:
  - `service/vllm` (OpenAI-compatible; usually best throughput), or
  - `service/ollama` (simple, easier to bottleneck under load)
- You have `uv` locally (SWE-bench helpers use it to pull metadata)

## Start here: run a suite + export + chart

### 1) (Optional) Build the `mcode` image (OpenShift internal registry)

If your project has a `BuildConfig` named `mcode`:

```bash
oc start-build mcode --from-dir=. --follow
```

### 2) Run the suite

```bash
# Default is SMOKE=1 (small LIMITs) so you can validate the pipeline first.
./deploy/k8s/run-suite.sh

# Full suite (takes a while)
SMOKE=0 ./deploy/k8s/run-suite.sh
```

Defaults:
- Backend: `ollama`
- Model: `granite4` (the script resolves this to whatever tag exists, e.g. `granite4:latest`)
- Outputs: `experiments/results/suite-<timestamp>/`

### 3) Export CSV

```bash
uv run mcode export-csv \
  -i experiments/results/suite-<timestamp> \
  --out-dir experiments/results/suite-<timestamp> \
  --prefix suite
```

By default, export omits huge log fields (`stdout`/`stderr`/`error`) so the CSVs are usable. If you want them:

```bash
uv run mcode export-csv \
  -i experiments/results/suite-<timestamp> \
  --out-dir experiments/results/suite-<timestamp> \
  --prefix suite \
  --include-logs
```

### 4) Generate a chart (SVG + PNG)

```bash
python scripts/make_suite_chart.py experiments/results/suite-<timestamp>
```

This writes `suite.summary.svg` and `suite.summary.png` into the suite directory.

PNG rendering requires one of: `rsvg-convert` (recommended), `inkscape`, or ImageMagick (`magick`/`convert`).

## MBPP/HumanEval sweeps with `oc_bench_sweep.py` (recommended defaults)

Use this for parameter sweeps with resume support and shard-level local result copying.

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp \
  --model granite4:latest \
  --loop-budget 1,3,5 --timeout 60,90 \
  --limit 500 --shard-count 20 --parallelism 3 \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id 20260211-mbpp-grid \
  --out-dir results/oc-confirm
```

Resume after disconnect:

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp \
  --loop-budget 1,3,5 --timeout 60,90 \
  --limit 500 --shard-count 20 --parallelism 3 \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id 20260211-mbpp-grid \
  --out-dir results/oc-confirm \
  --resume
```

Note: avoid blindly setting very high memory requests; larger requests reduce schedulable parallelism and can slow total wall-clock completion.

## HumanEval / MBPP: sharded Job (`run-bench.sh`)

Use this when you want one benchmark with specific knobs, or you want to integrate into your own pipeline.

### Configure the run (no YAML edits)

Edit `deploy/k8s/bench.env`.

Main knobs:
- `BENCHMARK`: `humaneval` or `mbpp`
- `LOOP_BUDGET`: mellea retry budget per task (stops early on the first pass)
- `TIMEOUT_S`: seconds per code execution attempt
- `SHARD_COUNT`: number of shards (the indexed Job completions)
- `STRATEGY`: `repair` (default) or `sofai`
- `S2_MODEL`: model ID for the SOFAI S2 solver (required when `STRATEGY=sofai`)
- `S2_BACKEND`: backend for S2 (default: `ollama`)
- `S2_MODE`: `fresh_start`, `continue_chat`, or `best_attempt` (default)

Backend selection:
- vLLM (OpenAI-compatible): `BACKEND=openai` + `OPENAI_BASE_URL=http://vllm:8000/v1`
- Ollama: `BACKEND=ollama` + `OLLAMA_HOST=http://ollama:11434`

### Run it

```bash
# Run with N shards in flight at once
PARALLELISM=4 ./deploy/k8s/run-bench.sh
```

Quick one-off overrides (without editing `bench.env`):

```bash
OVERRIDE_BENCHMARK=mbpp \
OVERRIDE_LIMIT=50 \
OVERRIDE_SHARD_COUNT=5 \
PARALLELISM=4 \
./deploy/k8s/run-bench.sh
```

Fetch shard DBs locally after completion:

```bash
PARALLELISM=4 FETCH_RESULTS=1 ./deploy/k8s/run-bench.sh
```

Or fetch later:

```bash
./deploy/k8s/fetch-results.sh mcode-bench
```

### Outputs + querying

After fetching, you’ll have one folder per pod under `./results-.../`, each containing a shard DB.

Query any shard DB:

```bash
uv run mcode results --db ./results-*/**/*.db --benchmark humaneval
```

If you want a single merged DB for the run (instead of per-shard DBs), use:

```bash
uv run mcode merge-shards --out merged.db --force ./results-*/**/*-shard-*.db
```

### Watch / debug

```bash
oc get job mcode-bench -o wide
oc get pods -l job-name=mcode-bench -w
oc logs -l job-name=mcode-bench --tail=200
oc get events --sort-by=.lastTimestamp | tail -n 30
```

## SWE-bench Lite: one Pod per instance (x86_64)

SWE-bench Lite is image-based and doesn’t fit the one-indexed-Job pattern. The simplest working approach is one Pod per instance, optionally many at once.

Before running, install the extra locally (needed for metadata / eval scripts):

```bash
uv pip install -e '.[swebench]'
```

### Gold vs model

- `MODE=gold`: apply the dataset’s reference patch and run tests (sanity check for the harness + cluster).
- `MODE=model`: generate a patch with `mcode` (via Mellea) and evaluate it (real benchmark mode).

### Single instance (debugging)

```bash
MODE=gold ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
MODE=model BACKEND=ollama MODEL=granite4:latest ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
```

### Batch + SQLite output

`run-swebench-lite.sh` launches multiple instance Pods (up to `PARALLELISM` at a time) and creates one SQLite DB at the end.

```bash
# Gold-patch smoke test (first N instances)
MODE=gold LIMIT=5 PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh

# Model-run via Ollama (first N instances)
MODE=model LIMIT=5 PARALLELISM=2 \
  BACKEND=ollama MODEL=granite4:latest \
  ./deploy/k8s/run-swebench-lite.sh
```

Query the DB it prints:

```bash
uv run mcode results --db experiments/results/swebench-lite-*.db --benchmark swebench-lite
```
