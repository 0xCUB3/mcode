# mCode

Benchmarking harness for mCode: a research project exploring whether small local LLMs can match
frontier models on agentic coding tasks.

- Benchmarks: HumanEval + MBPP
- LLM layer: Mellea (pluggable backends; default `ollama`)
- Execution: Docker sandbox (network disabled, read-only mount, timeouts)
- Storage: SQLite results DB (run configs + per-task outcomes)

## Quickstart

```bash
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

## Usage

Run a benchmark:

```bash
mcode bench humaneval --model granite3.3:8b --samples 100 --debug-iters 0 --timeout 60
mcode bench mbpp --model qwen2.5-coder-7b --samples 10 --debug-iters 3 --timeout 60
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

- Architecture / roadmap: `mcode-research-v2.md`
- GitHub Wiki (auto-synced): source lives in `docs/wiki/` and is pushed on every `main` update via
  `.github/workflows/wiki-sync.yml`.

## Publishing notes

This repo does not currently include a license file. If you intend it to be open source, add a
`LICENSE` before making it public.
