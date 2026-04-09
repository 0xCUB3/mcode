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

The preferred launcher path is now:

```bash
uv run mcode launch
```

The new launcher supports:

- `bluevela`
- `local-vllm`
- `local-ollama`
- `openai-compatible`

Useful commands:

```bash
uv run mcode launch --help
uv run mcode launch doctor --target bluevela
uv run mcode launch sync --target bluevela --check
uv run mcode launch status --json
```

See [`docs/COMMANDS.md`](/Users/skula/Documents/mcode/docs/COMMANDS.md) for the full command cookbook, including Blue Vela launch, attach, fetch, stop, and local provider examples.

The launcher reads optional per-user defaults from:

```bash
~/.config/mcode/launch.toml
```

It uses the current process environment for `$USER`, so the default Blue Vela profile resolves to paths like `/u/$USER/mcode-launch` and `/proj/dmfexp/$USER`. Override those in `launch.toml` if your cluster setup differs.

The legacy Blue Vela scripts are still available during the transition under `deploy/bluevela/`:

- `setup.sh`
- `start-vllm.sh`
- `run-swebench-live.sh`
- `stop-vllm.sh`
- `fetch-results.sh`

See [`deploy/bluevela/README.md`](/Users/skula/Documents/mcode/deploy/bluevela/README.md) for the remote workflow and the legacy script path.

## Notes

- `docs/COMMANDS.md` has the higher-level command cookbook.
- `research/` is the canonical home for durable benchmark run notes, optimization history, and rendered reports. Start with `research/2026-03-31-swebench-optimization-log/README.md` for the running SWE-bench optimization history.
