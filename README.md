# mCode

An agentic coding benchmark harness built on [Mellea](https://mellea.ai). Runs SWE-bench Verified, SWE-bench Live, and Aider Polyglot end-to-end on a local model or a remote vLLM server.

## Kept results

These are the checked-in runs that currently define the project. They are not all the same evaluation setting, so use the linked run notes when comparing numbers.

|Benchmark|Model|Setting|Result|
|-|-|-|-:|
|SWE-bench Verified|Qwen3.6-35B-A3B|multiturn, 5 selected attempts, full 500 tasks|**319/500** (63.8%)|
|Aider Polyglot|Qwen3.6-35B-A3B|20+12 turn retry budget, 3 selected attempts|**207/225** (92.0%)|
|Aider Polyglot|Qwen3.6-35B-A3B|single-selection control-loop run|190/225 (84.4%)|
|SWE-bench Verified|MiniMax-M2.5|post-redesign baseline, 1 sample|187/500 (37.4%)|

The 207/225 Aider result is a selected-trajectory result, not a default single-pass score. Per-benchmark data, exact commands, and caveats live under [`research/`](research/).

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

- [`research/`](research/) — benchmark run notes and rendered HTML reports.
- [`docs/COMMANDS.md`](docs/COMMANDS.md) — full reference with every flag.
- `MCODE_DEBUG=1` re-enables raw tracebacks; `NO_COLOR=1` disables ANSI color.
- Launch state lives at `$MCODE_LAUNCH_STATE` (default `~/.config/mcode/launch-state.json`).
