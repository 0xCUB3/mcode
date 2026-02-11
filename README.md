# mCode

mCode is a lightweight benchmarking harness for coding tasks through [Mellea](https://mellea.ai), which will eventually become an agentic coding tool tailored for small LLMs.

- Benchmarks: HumanEval, MBPP (SWE-bench Lite optional; local Docker on Apple Silicon can be finicky)
- LLM interface: Mellea (default backend: `ollama`)
- Results: SQLite (default: `experiments/results/results.db`)

## Install

Two good options:

### Option A: project virtualenv (best for development)

```bash
uv venv
uv pip install -e '.[dev]'

source .venv/bin/activate
mcode --help
```

### Option B: global tool

```bash
uv tool install -e .
uv tool update-shell
# restart your shell
mcode --help
```

## Run benchmarks

HumanEval / MBPP:

```bash
mcode bench humaneval --model granite3.3:8b --samples 5
mcode bench mbpp --model granite3.3:8b --samples 5
```

Quick smoke test (first N tasks only):

```bash
mcode bench humaneval --model granite3.3:8b --limit 10
```

### What the key flags mean

- `--samples`: attempts per task (note: stops early on the first passing attempt).
- `--debug-iters`: number of “fix” attempts after a failure per sample.
- `--timeout`: seconds per execution attempt (per sample/debug iteration).
- `--limit`: run the first N tasks.
- `--shard-count/--shard-index`: split tasks across multiple runs for parallelism.
- `--sandbox`:
  - `docker` (default): runs code in a Docker container (network disabled).
  - `process`: runs code directly as a local subprocess (useful inside locked-down containers / k8s Jobs).
    This is not safe isolation; prefer `docker` when you can.
- `--retrieval`: reserved flag; currently non-functional.

### Parallel / Kubernetes runs

Sharding is the simplest “plug-and-play” speedup: run the same command N times with different
`--shard-index` values.

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --shard-count 10 --shard-index 0 --db /results/shard-0.db
```

On Kubernetes, run HumanEval/MBPP inside Jobs with `--sandbox process` (Docker-in-Docker is usually not available).

There’s a minimal container + k8s/OpenShift setup in:

- `Dockerfile`
- `deploy/k8s/bench.env` (knobs)
- `deploy/k8s/mcode-bench-indexed-job.yaml` (indexed sharded Job)
- `deploy/k8s/run-bench.sh` (recommended submit script)

## SWE-bench Lite (optional)

SWE-bench Lite is much heavier than HumanEval/MBPP and has compatibility issues with Apple ARM: it evaluates patches against real repos inside
Docker images.

Install the extra:

```bash
uv pip install -e '.[swebench]'
```

If you installed `mcode` via `uv tool`, install the extra there too:

```bash
uv tool install -e '.[swebench]'
```

Run a small slice:

```bash
mcode bench swebench-lite --model granite3.3:8b --limit 5
```

On OpenShift (x86_64), you can avoid local Docker entirely and run one Pod per SWE-bench instance:

```bash
# gold-patch smoke test (first N instances)
MODE=gold LIMIT=5 PARALLELISM=2 ./deploy/k8s/run-swebench-lite.sh
```

If you see `ImageNotFound` while pulling `swebench/...` images, force local builds:

```bash
mcode bench swebench-lite --namespace "" --model granite3.3:8b --limit 5
```

If image building OOMs, try `--max-workers 1` and increase Docker Desktop memory.

## View results

```bash
mcode results --benchmark humaneval
mcode results --benchmark humaneval --model granite3.3:8b --compare-samples
mcode results --benchmark humaneval --time

# Aggregate shard DBs copied from k8s
mcode results --db-dir ./results --benchmark humaneval --compare-samples --time

# Lightweight HTML report (pass rate vs time-to-solve)
mcode report --db-dir ./results --benchmark humaneval --out ./results/report.html
```

## Export results (CSV) + charts

Export one or more results DBs to CSV (runs + per-task rows):

```bash
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

By default, CSV export omits very large fields (`stdout`/`stderr`/`error`). If you want them:

```bash
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode --include-logs
```

If you ran an OpenShift suite (`experiments/results/suite-...`), you can generate a legible summary chart:

```bash
python scripts/make_suite_chart.py experiments/results/suite-<timestamp>
```

## Command cookbook

For OpenShift/Kubernetes “do the thing” commands, see:

- `docs/COMMANDS.md`

Canonical benchmark + report docs (including OpenShift sweep workflow): `docs/benchmarking.md`.
If you want durable notes per run (and a snapshot of the HTML report), add an entry under `research/`.

For long OpenShift runs, use `--run-id` and `--resume` with `deploy/k8s/oc_bench_sweep.py` so you can reconnect after network drops.

## FAQ

### SWE-bench Lite prints Hugging Face `404 Not Found` messages. Is that bad?

Usually no. The HF client probes for optional files via `HEAD` requests; `404` is expected as long
as the dataset downloads and instances load.

### SWE-bench Lite fails with `base_image_tag cannot be None`

This usually means you’re running an older `mcode`. Reinstall/update and retry.
