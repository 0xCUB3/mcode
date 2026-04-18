# mCode

mCode is a SWE-bench-focused benchmark harness for agentic coding runs through [Mellea](https://mellea.ai).

- Benchmarks: `swebench-lite`, `swebench-live`
- LLM interface: Mellea
- Results: SQLite in `experiments/results/results.db` by default

## Install

For development:

```bash
uv sync --extra dev
uv run mcode --help
```

To sync the project with the pinned upstream Mellea release:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

If you want to override that temporarily with a local checkout, set
`MCODE_MELLEA_PATH=/path/to/mellea` before running the command.

## Run SWE-bench Lite

```bash
uv run mcode bench swebench-lite --model granite3.3:8b --limit 5
```

Useful flags:

- `--loop-budget`: retry budget for the agent
- `--timeout`: eval timeout per task
- `--limit`: run the first N tasks
- `--shard-count/--shard-index`: shard a run across multiple workers
- `--strategy`: `repair`, `sofai`, or `raw`
- `--n-samples`: number of patch samples to generate per task

If you need to force local image builds instead of pulling prebuilt images:

```bash
uv run mcode bench swebench-lite --namespace "" --model granite3.3:8b --limit 5
```

## Run SWE-bench Live

```bash
uv run mcode bench swebench-live --model granite3.3:8b --limit 5
```

`swebench-live` uses prebuilt evaluation images, so it does not need the lite image-build settings.

## Results

Per-run summaries:

```bash
uv run mcode results --benchmark swebench-live
uv run mcode results --benchmark swebench-live --time
```

HTML report:

```bash
uv run mcode report --db-dir ./results --benchmark swebench-live --out ./results/report.html
```

Merge shard DBs:

```bash
uv run mcode merge-shards --out ./results/merged.db ./results/swebench-live-shard-0.db ./results/swebench-live-shard-1.db
```

CSV export:

```bash
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

## Launch vLLM (Blue Vela or local)

`uv run mcode launch` spins up a vLLM server and records its endpoint so you can run benchmarks against it.

First time on a cluster:

```bash
uv run mcode launch doctor bluevela --init --login <user>@<login-host>
uv run mcode launch doctor bluevela
```

Push the local repo to the cluster (rsync, respects `.gitignore`):

```bash
uv run mcode launch sync bluevela              # subsequent syncs
uv run mcode launch sync bluevela --bootstrap  # first sync into a non-empty remote dir
uv run mcode launch sync bluevela --dry-run    # preview
```

Launch a server:

```bash
uv run mcode launch bluevela    --model Qwen/Qwen3.5-35B-A3B
uv run mcode launch local-vllm  --model Qwen/Qwen2.5-0.5B
uv run mcode launch local-ollama --model granite4
```

Manage running servers:

```bash
uv run mcode launch status
uv run mcode launch refresh
uv run mcode launch stop <server-id> | --all
```

Blue Vela profiles ship with correct vLLM flags per model: Qwen3.5 (27B / 35B-A3B), Gemma-4-31B-it, Granite 4.0, MiniMax-M2.5. Add more in `src/mcode/launch/profiles.py`.

Once a server is healthy, point the bench at it:

```bash
uv run mcode bench swebench-live --backend openai --model Qwen/Qwen3.5-35B-A3B --limit 10
```

`--backend openai` auto-resolves the endpoint from `uv run mcode launch status` when a healthy server matches `--model`. `OPENAI_BASE_URL` / `OPENAI_API_KEY` still override if set.

The legacy shell scripts under `deploy/bluevela/` still work as a fallback. See `deploy/bluevela/README.md`.

## Notes

- `research/` is the canonical home for benchmark run notes and rendered reports.
- `docs/e2e-verification.md` captures the live cluster test of the launcher.
- `docs/bluevela-probe-findings.md` documents the Phase 0.5 Blue Vela cluster probe.
