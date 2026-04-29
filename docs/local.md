# Local workflow

Run mCode against a model on your own machine — either Ollama or a local vLLM server. Steps are in the order you run them. For Blue Vela LSF, see [`bluevela.md`](bluevela.md). For the full flag reference, see [`COMMANDS.md`](COMMANDS.md).

## 1. Install + sanity check

```bash
uv sync --extra dev
uv run mcode --version
uv run mcode doctor
```

`mcode doctor` (no target) checks the four things you need: results dir writable, container runtime (podman or docker) on PATH, Mellea importable, ruff present. Red `✗` means follow the `next:` line.

For the full bench dependency set:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

## 2. Bring up a model

Pick one. For most local work, Ollama is easiest because it manages the model server for you.

### Option A: Ollama

```bash
# in another shell, if it isn't already running
ollama serve

uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <id printed above> --timeout 120
```

### Option B: local vLLM

```bash
uv run mcode launch local-vllm --model Qwen/Qwen2.5-0.5B
uv run mcode launch wait <id> --timeout 600
```

`launch wait` blocks until the server is healthy / failed / timeout. Exit codes: 0 healthy, 1 failed-or-stopped, 2 timeout, 3 no-such-id.

## 3. Run a benchmark

The bench will auto-resolve `--backend openai` to the healthy server matching `--model`:

```bash
# 16-task smoke slice (good first run)
uv run mcode bench smoke --backend openai --model granite4 --shards 4

# SWE-bench Lite, first 16 tasks
uv run mcode bench swebench-lite --backend openai --model granite4 --limit 16 --shards 4

# Aider Polyglot
uv run mcode bench aider-polyglot --backend openai --model granite4
```

Useful flags:

- `--shards N` — fan out N worker processes; per-shard DBs merge into `--db` automatically
- `--limit N` — only the first N tasks (use this for a fast smoke before a real run)
- `--loop-budget N` — agent retry budget per task (default 15)
- `--sampling {none,multiturn}` — Mellea sampling strategy
- `--selection-attempts N` — independent full-budget trajectories; pick one patch per task
- `--json` — emit one JSON object per state change with strictly monotonic `seq`

Bench runs resume automatically when you rerun the same command with the same `--db`. Completed tasks are skipped, retryable infra failures are retried, and sharded runs reuse their shard DBs under `<db-stem>-shards/`.

```bash
uv run mcode bench smoke --backend openai --model granite4 --shards 4 --json | jq -c '.'
```

## 4. Watch / list / cancel

While a sharded bench runs, the dashboard updates a live Rich table per shard. From a separate shell:

```bash
uv run mcode bench list                 # historical runs
uv run mcode bench list --json          # machine-readable
uv run mcode bench cancel <run-id>      # SIGTERM each shard pid, SIGKILL after 10s
uv run mcode launch status              # servers + runs
uv run mcode watch                      # combined live dashboard, refreshes every 2s
```

`bench cancel` only works for sharded runs. A single non-sharded run executes in-process and isn't cancellable from another shell — use Ctrl+C in the running terminal.

## 5. Stop and inspect results

```bash
uv run mcode launch stop <server-id>
# or
uv run mcode launch stop --all          # only your recorded servers; never bkill 0

uv run mcode results --benchmark swebench-live
uv run mcode results --benchmark swebench-live --time

uv run mcode report --db-dir ./experiments/results --benchmark swebench-live --out report.html
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

`results --time` adds `sec/solve`, `solves/hour`, and p95 latency.

## Common environment variables

|Variable|Purpose|
|-|-|
|`MCODE_CONTEXT_WINDOW`|LLM context window override|
|`MCODE_MAX_NEW_TOKENS`|LLM max output tokens|
|`MCODE_REACT_TIMEOUT`|ReACT loop timeout in seconds|
|`MCODE_KEEP_IMAGES`|Skip post-task image cleanup|
|`MCODE_SKIP_IMAGE_PULL`|Use existing local images instead of pulling|
|`OPENAI_BASE_URL` / `OPENAI_API_KEY`|Override the auto-resolved endpoint|

Full env-var list and every flag: [`COMMANDS.md`](COMMANDS.md).

## Troubleshooting

- `mcode doctor` reports a `✗` — follow the `next:` line on that row.
- `bench smoke` hangs at "pulling images" — first run pulls eval containers (~5 min). Set `MCODE_SKIP_IMAGE_PULL=1` if they're already cached locally.
- "no healthy server for `<model>`" — `mcode launch status` will tell you what's running. Either launch it, or set `OPENAI_BASE_URL` to point at any OpenAI-compatible endpoint.
- Need a clean retry — `mcode launch stop --all` then `mcode launch local-ollama --model ...` again.
- `MCODE_DEBUG=1` disables the formatted error layout and dumps a raw traceback.
