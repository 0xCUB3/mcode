# mCode

An agentic coding benchmark harness built on [Mellea](https://mellea.ai). Runs SWE-bench Verified, SWE-bench Live, and Aider Polyglot end-to-end on a local model or a remote vLLM server.

## Top results

Best mCode results to date with Qwen3.6-35B-A3B served via vLLM:

|Benchmark|Bare model|Base Mellea|mCode|
|-|-|-|-|
|SWE-bench Verified (500 tasks)|0.3%|37.4%|**63.8%** (319/500)|
|Aider Polyglot (225 tasks)|16%|45.8% (103/225)|**92.0%** (207/225)|

Bare model = LLM with no agentic harness. Base Mellea = upstream Mellea ReACT loop. mCode = full harness with selection, multiturn sampling, workspace-context discovery, generic control-loop nudges, and the verification policy. Per-benchmark data + run notes live under [`research/`](research/).

## Install

```bash
uv sync --extra dev
uv run mcode --help
```

For the full bench dependency set:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

## Pick your path

|Where you want to run the model|Read|
|-|-|
|Locally (Ollama or local vLLM)|[`docs/local.md`](docs/local.md)|
|On the Blue Vela LSF cluster|[`docs/bluevela.md`](docs/bluevela.md)|
|Reference: every command and flag|[`docs/COMMANDS.md`](docs/COMMANDS.md)|

## Commands at a glance

```bash
mcode doctor                 # system + launch diagnostics
mcode launch local-ollama --model granite4
mcode launch wait <id>       # block until healthy
mcode bench smoke --backend openai --model granite4 --shards 4
mcode bench list             # list runs
mcode bench cancel <run-id>
mcode watch                  # live dashboard
mcode launch status
mcode launch stop <id>
mcode results --benchmark swebench-live
```

`mcode --help` lists everything. Add `--json` to bench / wait / status / list for machine-readable output.

## Notes

- [`research/`](research/) — benchmark run notes and result summaries.
- [`docs/COMMANDS.md`](docs/COMMANDS.md) — full reference with every flag.
- `MCODE_DEBUG=1` re-enables raw tracebacks; `NO_COLOR=1` disables ANSI color.
- Launch state lives at `$MCODE_LAUNCH_STATE` (default `~/.config/mcode/launch-state.json`).
