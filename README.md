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

To sync against the pinned fork revision in `pyproject.toml`:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

If you want to override that temporarily with a local checkout, set
`MCODE_MELLEA_PATH=/path/to/mellea-fork` before running the command.

## Run SWE-bench Lite

```bash
mcode bench swebench-lite --model granite3.3:8b --limit 5
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
mcode bench swebench-lite --namespace "" --model granite3.3:8b --limit 5
```

## Run SWE-bench Live

```bash
mcode bench swebench-live --model granite3.3:8b --limit 5
```

`swebench-live` uses prebuilt evaluation images, so it does not need the lite image-build settings.

## Results

Per-run summaries:

```bash
mcode results --benchmark swebench-live
mcode results --benchmark swebench-live --time
```

HTML report:

```bash
mcode report --db-dir ./results --benchmark swebench-live --out ./results/report.html
```

Merge shard DBs:

```bash
mcode merge-shards --out ./results/merged.db ./results/swebench-live-shard-0.db ./results/swebench-live-shard-1.db
```

CSV export:

```bash
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

## Blue Vela

The maintained Blue Vela path is under `deploy/bluevela/`:

- `setup.sh`
- `start-vllm.sh`
- `run-swebench-live.sh`
- `stop-vllm.sh`
- `fetch-results.sh`

See `deploy/bluevela/README.md` for the remote workflow.

## Notes

- `docs/COMMANDS.md` has the higher-level command cookbook.
- `docs/benchmarking.md` and `docs/swebench-optimization-log.md` hold the project benchmarking notes.
- `research/` is the place for durable run notes and comparisons.
