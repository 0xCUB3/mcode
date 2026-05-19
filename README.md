# mCode

mCode is a benchmark harness for local model coding runs with Mellea. It wraps the Mellea ReACT loop, gives it a small set of code tools, records every task in SQLite, and keeps enough run state around that a long benchmark can be resumed, cancelled, inspected, and cleaned up without digging through random temp files.

It runs SWE-bench Verified, SWE-bench Live, SWE-bench Lite, Aider Polyglot, experimental Terminal-Bench 2.0 support, and a small mixed suite. You can point it at Ollama, a local vLLM server, an OpenAI-compatible endpoint, or a vLLM job on IBM's Blue Vela cluster. Terminal-Bench currently has only local Harbor smoke coverage; full Terminal-Bench runs and Blue Vela support are not validated yet.

## Results

These are the best mCode runs I have in the repo right now, using Qwen3.6-35B-A3B served by vLLM.

|Benchmark|Bare model|Base Mellea|mCode|
|-|-|-|-|
|SWE-bench Verified (500 tasks)|0.3%|37.4%|**63.8%** (319/500)|
|Aider Polyglot (225 tasks)|16%|45.8% (103/225)|**92.0%** (207/225)|

Bare model means the model was asked to solve the task without this harness. Base Mellea means the upstream Mellea ReACT loop. mCode adds the harness pieces this repo is about: workspace inspection, tool policy, verification, patch selection, split generation and evaluation, sharding, artifacts, and run bookkeeping. The notes and raw summaries live under [`research/`](research/).

## Installation

I use `uv` for this repo.

```bash
uv sync --extra dev
uv run mcode --help
```

For SWE-bench or dataset-heavy runs, install the extra packages too:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

If you are only running Aider Polyglot, check the language toolchains before a long run:

```bash
uv run mcode deps toolchains --benchmark aider-polyglot
```

## Local quick start

This is the quickest way to get up and running to make sure the checkout, model server, Docker setup, and result DB are all working.

```bash
uv run mcode doctor local-ollama
uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <server-id> --timeout 120
uv run mcode bench smoke --backend openai --model granite4 --shards 4
uv run mcode bench show --latest
```

For a local Qwen or other Ollama model, use the exact name from `ollama list` in both commands. For example, if Ollama shows `qwen3.6:35b-a3b`, pass that string to `launch local-ollama` and to `bench`.

## Blue Vela quick start

The cluster path is longer because there is an SSH config, an rsync step, and an LSF vLLM server job involved. Once the config is written, the normal rhythm is sync, launch server, run bench, inspect, stop server.

```bash
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
uv run mcode launch sync bluevela
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json
uv run mcode launch wait <server-id> --timeout 1800

MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela --shards 4

uv run mcode bench show --latest
uv run mcode launch stop <server-id>
```

## Documentation

Start with the workflow that matches where you plan to run the model.

|Topic|Doc|
|-|-|
|Local Ollama and local vLLM runs|[`docs/local.md`](docs/local.md)|
|Blue Vela LSF runs|[`docs/bluevela.md`](docs/bluevela.md)|
|Command reference with the weird corners included|[`docs/COMMANDS.md`](docs/COMMANDS.md)|
|Terminal-Bench 2.0 via Harbor|[`docs/terminalbench.md`](docs/terminalbench.md)|
|How the code is put together|[`docs/architecture.md`](docs/architecture.md)|

## Common commands

```bash
mcode doctor
mcode launch status
mcode launch wait <id>
mcode bench list
mcode bench list --wide
mcode bench show --latest
mcode bench cancel <run-id>
mcode bench prune --status failed --older-than 7d
mcode watch
mcode results --db-dir research/<run> --time
mcode compare --baseline-dir old.db --candidate-dir new.db --max-lost 0
```

Most benchmark commands accept `--json` if you want one JSON object per line instead of the human progress display. I usually keep the human view on for long local runs because it shows each task moving through repo prep, model turns, tool calls, and official evaluation. Set `MCODE_LIVE_TRACE=0` if that is too noisy.

## State and output locations

Benchmark rows go into SQLite, usually `experiments/results/results.db` unless you pass `--db`. Generated task artifacts sit next to the DB by default as `<db-stem>-artifacts`. Sharded runs create per-shard DBs under `<db-stem>-shards/` and merge them back into the DB you asked for.

Launch and bench state lives in `~/.config/mcode/launch-state.json`, or wherever `MCODE_LAUNCH_STATE` points. That state file is what powers `bench list`, `bench show`, `bench cancel`, `launch status`, and `watch`. If it gets full of failed experiments, use `bench prune`; do not hand edit it unless you have to.

## Usage notes

Run `mcode doctor <target>` before blaming the model. Use `bench smoke` to eliminate infra issues before running the full suite. Keep explicit `--db` paths for anything you may want to compare later. When running on Blue Vela, keep the server id and run id until the DB and artifacts have been fetched back. If a remote bench finishes but the fetch dies, rerun the same command or use `mcode bench artifacts fetch` rather than starting the whole benchmark again.
