# OpenShift / Kubernetes command cookbook

This page is only about running `mcode` on OpenShift/Kubernetes. It assumes:

- you’re already logged in with `oc` and on the right namespace (`oc project -q`)
- the cluster can reach your model backend (`service/vllm` or `service/ollama`)
- you have `uv` available locally (the SWE-bench helpers use it to pull metadata)

## 1) Build the `mcode` image (OpenShift internal registry)

If your project has a `BuildConfig` named `mcode`:

```bash
oc start-build mcode --from-dir=. --follow
```

## 2) HumanEval / MBPP: sharded Job (recommended)

### Configure the run (no YAML edits)

Edit `deploy/k8s/bench.env`.

Main knobs:
- `BENCHMARK`: `humaneval` or `mbpp`
- `SAMPLES`: attempts per task (stops early on first pass)
- `DEBUG_ITERS`: fix attempts after a failure
- `TIMEOUT_S`: seconds per code execution attempt
- `SHARD_COUNT`: total shards (must match the Job’s completions)

Backend selection:
- vLLM (OpenAI-compatible): `BACKEND=openai` + `OPENAI_BASE_URL=http://vllm:8000/v1`
- Ollama: `BACKEND=ollama` + `OLLAMA_HOST=http://ollama:11434`

### Run it

```bash
# Run with N shards in flight at once
PARALLELISM=4 ./deploy/k8s/run-bench.sh
```

Quick overrides:

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

### Watch / debug

```bash
oc get job mcode-bench -o wide
oc get pods -l job-name=mcode-bench -w
oc logs -l job-name=mcode-bench --tail=200
oc get events --sort-by=.lastTimestamp | tail -n 30
```

### Query results (SQLite)

Each shard writes its own SQLite DB. After fetching, point `mcode results` at any shard DB:

```bash
uv run mcode results --db ./results-*/**/*.db --benchmark humaneval
```

## 3) SWE-bench Lite: one Pod per instance (x86_64)

SWE-bench Lite is image-based and doesn’t fit the one sharded Job pattern. The hack to make it work is one Pod per instance, optionally many at once.

Before running SWE-bench Lite, install the extra locally (needed for metadata / eval scripts):

```bash
uv pip install -e '.[swebench]'
```

### Single instance (debugging)

```bash
MODE=gold ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
MODE=model BACKEND=ollama MODEL=granite3-dense:8b ./deploy/k8s/run-swebench-lite-one.sh sympy__sympy-20590
```

- `MODE=gold`: apply the dataset’s ground-truth patch and run tests. This is mostly a cluster sanity check.
- `MODE=model`: `mcode` generates a patch via Mellea, then the eval image applies it and runs tests.
- `sympy__sympy-20590`: SWE-bench Lite `instance_id` (repo + issue-style ID). Swap this string to run a different one.

### Batch (recommended) + SQLite output

`run-swebench-lite.sh` launches multiple instance Pods (up to `PARALLELISM` at a time) and creates one SQLite DB at the end.

```bash
# Gold-patch smoke test (first N instances)
MODE=gold LIMIT=5 PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh

# Model-run via Ollama (first N instances)
MODE=model LIMIT=5 PARALLELISM=2 \
  BACKEND=ollama MODEL=granite3-dense:8b \
  ./deploy/k8s/run-swebench-lite.sh

# Pick specific instances
MODE=gold PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh \
  sympy__sympy-20590 astropy__astropy-12907
```

Query the DB it prints:

```bash
uv run mcode results --db experiments/results/swebench-lite-*.db --benchmark swebench-lite
```
