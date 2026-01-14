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

## Installing `mcode` (so the command works)

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

Run a benchmark:

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --debug-iters 0 --timeout 60
mcode bench mbpp --model qwen2.5-coder-7b --samples 10 --debug-iters 3 --timeout 60

# SWE-bench Lite (requires `uv pip install -e '.[swebench]'`)
mcode bench swebench-lite --model qwen2.5-coder-7b --limit 5 --timeout 1800
```

Query results:

```bash
mcode results --benchmark humaneval
mcode results --benchmark humaneval --model granite3.3:8b --compare-samples
mcode results --benchmark humaneval --model granite3.3:8b --debug-iters 0 --timeout 60
```

By default results are stored in `experiments/results/results.db` (override with `--db`).

## “Just run `mcode`”

Two options:

- In this repo: use `uv run mcode ...` (no activation needed).
- From anywhere: install as a uv tool:
  - `uv tool install -e .`
  - `uv tool update-shell` (then restart your shell)

## Documentation

Architecture notes are currently maintained privately (not checked into the public repo).

## Publishing notes

This repo does not currently include a license file. If you intend it to be open source, add a
`LICENSE` before making it public.
