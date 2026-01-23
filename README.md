# mCode

mCode is a lightweight benchmarking harness for coding tasks through [Mellea](https://mellea.ai), which will eventually become an agentic coding tool tailored for small LLMs.

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
  - `process`: runs code directly on the host via a local subprocess (better for k8s?).
    This is not safe isolation so only use it in a locked-down container if you care about security.
- `--retrieval`: reserved flag; currently non-functional.

### Parallel / Kubernetes runs

Sharding is the simplest “plug-and-play” speedup: run the same command N times with different
`--shard-index` values.

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --shard-count 10 --shard-index 0 --db /results/shard-0.db
```

On Kubernetes, run HumanEval/MBPP inside Jobs with `--sandbox process` (Docker-in-Docker is usually not available).

There’s a minimal Dockerfile + k8s templates in:

- `Dockerfile`
- `deploy/k8s/mcode-bench-indexed-job.yaml`
- `deploy/k8s/results-pvc.yaml`
- `deploy/k8s/ollama.yaml` (optional, in-cluster Ollama Service)

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
