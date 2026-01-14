# mCode

Benchmarking harness for mCode: a research project exploring whether small local LLMs can match
frontier models on agentic coding tasks.

- Benchmarks: HumanEval + MBPP
- LLM layer: Mellea (pluggable backends; default `ollama`)
- Execution: Docker sandbox (network disabled, read-only mount, timeouts)
- Storage: SQLite results DB (run configs + per-task outcomes)

## Quickstart

```bash
# Create a local virtualenv + install (recommended for development)
uv venv
uv pip install -e '.[dev]'

# Run HumanEval
mcode bench humaneval --model qwen2.5-coder-7b --samples 5

# Show results
mcode results --benchmark humaneval

# Compare pass rates across sample counts (same model/config)
mcode results --benchmark humaneval --model qwen2.5-coder-7b --compare-samples
```

## Requirements

- Python 3.11+ (project uses `uv` for packaging/runs)
- Docker (for secure-ish code execution)
- A Mellea backend (for local models: Ollama is easiest)
- For SWE-bench Lite: install the optional dependency `.[swebench]`

## Installing `mcode`

There are two supported ways to run the CLI.

### Option A: install as a global tool (recommended)

This makes `mcode` available on your PATH without activating a virtualenv:

```bash
uv tool install -e .
uv tool update-shell
# restart your shell
mcode --help
```

### Option B: use a project virtualenv

```bash
uv venv
uv pip install -e '.[dev]'

# Either activate:
source .venv/bin/activate
mcode --help

# Or run without activating:
uv run mcode --help
```

If you installed into a virtualenv but didn’t activate it, you can also run the binary directly:
`.venv/bin/mcode` (macOS/Linux) or `.venv\\Scripts\\mcode.exe` (Windows).

## Usage

### Commands

- `mcode bench ...`: run a benchmark and write results to SQLite.
- `mcode results ...`: query pass rates from SQLite.

Tip: add `-v/--verbose` to show Mellea backend logs (useful for debugging backend connectivity).

### `mcode bench`

Run a benchmark:

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --debug-iters 0 --timeout 60
mcode bench mbpp --model qwen2.5-coder-7b --samples 10 --debug-iters 3 --timeout 60

# SWE-bench Lite (requires `uv pip install -e '.[swebench]'` in the same environment)
mcode bench swebench-lite --model qwen2.5-coder-7b --limit 5 --timeout 1800 --max-workers 1
```

Common options (HumanEval + MBPP):

- `--model`: the Mellea model id (e.g. `granite3.3:8b` for Ollama).
- `--backend`: Mellea backend name (default: `ollama`).
- `--samples`: how many independent attempts per task. Stops early on first pass.
- `--debug-iters`: after a failed attempt, how many “fix” attempts to allow (per sample).
- `--timeout`: seconds per sandbox execution attempt.
- `--limit`: number of tasks to run (takes the first `N` tasks in dataset order).
- `--db`: SQLite path (default: `experiments/results/results.db`).
- `--retrieval/--no-retrieval`: placeholder flag for future ablations (currently off by default).

SWE-bench Lite specific options:

- `--split`: dataset split (`dev` or `test`).
- `--arch`: Docker arch for SWE-bench images:
  - `auto` (default): uses `arm64` on Apple Silicon, `x86_64` otherwise.
  - `x86_64`: recommended on Apple Silicon for compatibility (uses emulation).
  - `arm64`: faster on Apple Silicon when it works, but some instances require old conda packages not
    available on `linux-aarch64`.
- `--max-workers`: parallelism for image building (lower this to reduce RAM pressure).
- `--namespace`: use prebuilt SWE-bench images from a container registry namespace (if available).
- `--force-rebuild`: rebuild images even if they exist.
- `--mem-limit`, `--pids-limit`: limits for the evaluation container (not the image build step).

Note: `mcode` must run in the same environment where you installed the `.[swebench]` extra. If you
installed `mcode` via `uv tool install ...`, install the extra there too:

```bash
uv tool install -e '.[swebench]'
```

### `mcode results`

Query results (pass rates):

```bash
mcode results --benchmark humaneval
mcode results --benchmark humaneval --model granite3.3:8b --compare-samples
mcode results --benchmark humaneval --model granite3.3:8b --debug-iters 0 --timeout 60
```

Filters:

- `--benchmark`, `--model`, `--backend`
- `--samples`, `--debug-iters`, `--timeout`
- `--retrieval` (accepts `true/false`)
- `--compare-samples`: group by sample count for easy “does sampling help?” comparisons

By default results are stored in `experiments/results/results.db` (override with `--db`).

## FAQ

### SWE-bench Lite prints a bunch of Hugging Face 404s. Is that bad?

Usually no. The Hugging Face client probes for optional files (like dataset scripts/metadata) via
`HEAD` requests; many datasets don’t have those files, so `404 Not Found` is expected as long as the
actual dataset `GET` requests succeed and instances load.

### SWE-bench Lite fails with `base_image_tag cannot be None`

This typically means you’re running an older `mcode` build. Update/reinstall `mcode` and re-run.

- From a clone: `git pull` then run with `uv run mcode ...`
- If using `uv tool`: reinstall with `uv tool install -e '.[swebench]'` from the repo root

### SWE-bench Lite fails building env images on Apple Silicon (arm64)

If the build log mentions something like `PackagesNotFoundError` for a pinned old package (for
example `setuptools==38.2.4` for Python 3.6), that’s an upstream limitation on `linux-aarch64`.

Fix: run amd64 SWE-bench images via emulation:

```bash
mcode bench swebench-lite --arch x86_64 --max-workers 1 --limit 10 --model granite3.3:8b --samples 1
```

If you still hit errors like exit code `137`, increase Docker Desktop memory and keep
`--max-workers 1`.

### SWE-bench Lite fails building the x86_64 base image with exit code 133 (Miniconda / rosetta error)

If you see something like:

- `rosetta error: failed to open elf at /lib64/ld-linux-x86-64.so.2`
- `Trace/breakpoint trap`
- `returned a non-zero code: 133`

that’s a Docker Desktop amd64-emulation failure while running the Miniconda installer inside the
`sweb.base.py.x86_64:latest` image build. It’s not caused by missing Python packages in `mcode`.

Checks / mitigations:

- Verify Docker can actually run amd64 containers: `docker run --rm --platform linux/amd64 ubuntu:22.04 uname -m`
  should print `x86_64`.
- Restart/upgrade Docker Desktop and retry with `--max-workers 1`.
- If it still fails, the practical workaround is to build/run SWE-bench Lite on a machine that can build amd64
  images natively (Linux x86_64 / Intel Mac), or prebuild and distribute images.
