# mCode

mCode is a lightweight benchmarking harness for coding tasks, which will eventually become an agentic coding tool tailored for small LLMs. It runs a benchmark, executes the
model’s output, and stores per-task outcomes in SQLite so you can compare models and settings later.

- Benchmarks: HumanEval, MBPP (SWE-bench Lite optional -- currently not fully functional on Apple Silicon)
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

### Option B: global tool (best for “just run the CLI”)

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

- `--samples`: attempts per task (stops early on the first passing attempt).
- `--debug-iters`: number of “fix” attempts after a failure (per sample).
- `--timeout`: seconds per execution attempt (per sample/debug iteration).
- `--limit`: run the first N tasks (useful for quick tests).
- `--shard-count/--shard-index`: split tasks across multiple runs for parallelism.
- `--sandbox`:
  - `docker` (default): runs code in a Docker container (network disabled).
  - `process`: runs code directly on the host via a local subprocess (useful inside k8s pods).
    This is not safe isolation; only use it in a locked-down container if you care about security.
- `--retrieval`: reserved flag (no behavior change yet; stored for later analysis).

### Parallel / Kubernetes runs

Sharding is the simplest “plug-and-play” speedup: run the same command N times with different
`--shard-index` values.

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --shard-count 10 --shard-index 0 --db /results/shard-0.db
```

On Kubernetes/OpenShift, run HumanEval/MBPP inside Jobs with `--sandbox process` (most clusters won’t
support Docker-in-Docker).

## SWE-bench Lite (optional)

SWE-bench Lite is much heavier than HumanEval/MBPP: it evaluates patches against real repos inside
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

If you see `ImageNotFound` while pulling `swebench/...` images, force local builds:

```bash
mcode bench swebench-lite --namespace "" --model granite3.3:8b --limit 5
```

If image building OOMs, try `--max-workers 1` and increase Docker Desktop memory.

## View results

```bash
mcode results --benchmark humaneval
mcode results --benchmark humaneval --model granite3.3:8b --compare-samples
```

## FAQ

### SWE-bench Lite prints Hugging Face `404 Not Found` messages. Is that bad?

Usually no. The HF client probes for optional files via `HEAD` requests; `404` is expected as long
as the dataset downloads and instances load.

### SWE-bench Lite fails with `base_image_tag cannot be None`

This usually means you’re running an older `mcode`. Reinstall/update and retry.
