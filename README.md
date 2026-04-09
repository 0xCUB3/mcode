# mCode

mCode is a SWE-bench-focused benchmark harness for agentic coding runs through [Mellea](https://mellea.ai).

## Install

```bash
uv sync --extra dev
uv run mcode --help
```

Benchmark extras:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

Use `MCODE_MELLEA_PATH=/path/to/mellea-fork` if you want to temporarily override the pinned `mellea` source with a local checkout.

## Core Commands

SWE-bench Lite:

```bash
uv run mcode bench swebench-lite --model granite3.3:8b --limit 5
```

SWE-bench Live:

```bash
uv run mcode bench swebench-live --model granite3.3:8b --limit 5
```

Results:

```bash
uv run mcode results --benchmark swebench-live
uv run mcode results --benchmark swebench-live --time
uv run mcode report --db-dir ./results --benchmark swebench-live --out ./results/report.html
uv run mcode merge-shards --out ./results/merged.db ./results/swebench-live-shard-0.db ./results/swebench-live-shard-1.db
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

## Blue Vela

```bash
uv run mcode launch
```

Quick checks:

```bash
uv run mcode launch --help
uv run mcode launch doctor --target bluevela
uv run mcode launch sync --target bluevela --check
uv run mcode launch status --json
```

Optional per-user defaults live in:

```bash
~/.config/mcode/launch.toml
```

The default Blue Vela profile resolves from `$USER`, for example `/u/$USER/mcode-launch` and `/proj/dmfexp/$USER`.

See [`docs/COMMANDS.md`](docs/COMMANDS.md) for the full command cookbook, including local `vllm`, local `ollama`, and generic `openai-compatible` examples, and [`deploy/bluevela/README.md`](deploy/bluevela/README.md) for the Blue Vela quickstart.

## Notes

- `docs/COMMANDS.md` is the source of truth for operational commands.
- `research/` holds durable benchmark notes, optimization history, and reports.
