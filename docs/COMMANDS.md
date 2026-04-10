# Command Cookbook

Use `mcode` commands for setup, local runs, and Blue Vela runs. Do not default to raw `ssh`, `rsync`, `bsub`, `bjobs`, or the legacy shell scripts unless you are debugging the launcher or the user explicitly asks for the old flow.

## Supported environments

- macOS and Linux are supported natively
- Windows should run `uv run mcode ...` inside `WSL2`
- native PowerShell and CMD are not supported launcher environments
- Blue Vela can still be driven from Windows, but the launcher should run inside `WSL2`

## Install and sync

Development environment:

```bash
uv sync --extra dev
```

Benchmark extras:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

Remote Blue Vela workspace bootstrapping happens through `uv run mcode launch sync` and the launcher itself. You should not need manual `rsync`.

## Local benchmark runs

SWE-bench Lite:

```bash
uv run mcode bench swebench-lite --model granite3.3:8b --limit 5
```

SWE-bench Live:

```bash
uv run mcode bench swebench-live --model granite3.3:8b --limit 5
```

## Blue Vela runs

Check cluster access and storage:

```bash
uv run mcode launch doctor --target bluevela --json
```

Check sync state:

```bash
uv run mcode launch sync --target bluevela --check --json
```

Apply sync:

```bash
uv run mcode launch sync --target bluevela --apply --json
```

Launch a Blue Vela run:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-live \
  --parallelism 4 \
  --yes
```

Launch and follow logs immediately:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-live \
  --parallelism 4 \
  --yes \
  --follow
```

The launcher defaults the split by benchmark:
- `swebench-live` -> `verified`
- `swebench-lite` -> `test`

For `swebench-lite`, `launch` also accepts `--dataset`. This matters if you are replaying a Verified slice through the lite harness:

```bash
uv run mcode launch \
  --target bluevela \
  --model google/gemma-4-31B-it \
  --benchmark swebench-lite \
  --dataset princeton-nlp/SWE-bench_Verified \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --parallelism 2 \
  --yes
```

Override the split only when you have a specific reason:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3-1.7B \
  --benchmark swebench-lite \
  --split test \
  --limit 1 \
  --loop-budget 1 \
  --timeout 60 \
  --parallelism 1 \
  --yes
```

## Blue Vela run management

List known runs, servers, and synced workspaces:

```bash
uv run mcode launch status --json
```

Attach to a known run:

```bash
uv run mcode launch attach run-12345678
```

`launch attach` streams logs by default. Use `--json` when you only want metadata.

Fetch run artifacts:

```bash
uv run mcode launch fetch run-12345678 --destination results
```

Stop a run or server:

```bash
uv run mcode launch stop run-12345678
uv run mcode launch stop server-12345678
```

## Local model serving

Local Ollama health:

```bash
uv run mcode launch doctor --target local-ollama --json
```

Local vLLM health:

```bash
uv run mcode launch doctor --target local-vllm --json
```

Local vLLM dry run:

```bash
uv run mcode launch \
  --target local-vllm \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-lite \
  --limit 1 \
  --loop-budget 1 \
  --timeout 60 \
  --parallelism 1 \
  --json
```

Local runs also honor real shard parallelism against one shared endpoint per launch:

```bash
uv run mcode launch \
  --target openai-compatible \
  --openai-base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-lite \
  --parallelism 2 \
  --limit 1 \
  --yes
```

## Legacy Blue Vela scripts

The scripts under `deploy/bluevela/` remain available for debugging, but the default documented workflow is:

- `uv run mcode launch doctor`
- `uv run mcode launch sync`
- `uv run mcode launch`
- `uv run mcode launch status`
- `uv run mcode launch fetch`
- `uv run mcode launch stop`
