# Command Cookbook

Use `mcode` commands for setup, local runs, and Blue Vela runs. Do not default to raw `ssh`, `rsync`, `bsub`, `bjobs`, or the legacy shell scripts unless you are debugging the launcher or the user explicitly asks for the old flow.

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

The launcher defaults the split by benchmark:
- `swebench-live` -> `verified`
- `swebench-lite` -> `test`

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

## Legacy Blue Vela scripts

The scripts under `deploy/bluevela/` remain available for debugging, but the default documented workflow is:

- `uv run mcode launch doctor`
- `uv run mcode launch sync`
- `uv run mcode launch`
- `uv run mcode launch status`
- `uv run mcode launch fetch`
- `uv run mcode launch stop`
