# Commands

## Install

```bash
uv run mcode deps sync --extra swebench --extra datasets
uv run mcode deps sync --extra swebench --extra datasets --extra observability
```

## Bench

```bash
uv run mcode bench swebench-lite --model granite3.3:8b --limit 5
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench swebench-lite --backend openai --model Qwen/Qwen3.6-35B-A3B --on bluevela --limit 5
uv run mcode bench swebench-lite --model granite3.3:8b --limit 16 --shards 4
uv run mcode bench swebench-lite --model granite3.3:8b --sampling rejection --n-samples 3 --limit 5
uv run mcode bench swebench-live --model granite3.3:8b --limit 5
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench smoke --model Qwen/Qwen3.6-35B-A3B --backend openai --shards 4
```

Key flags:

- `--shards N`
- `--sampling {none,rejection,repair,sofai}`
- `--sampling-budget N`
- `--n-samples N`

## Results

```bash
uv run mcode results --benchmark swebench-live
uv run mcode results --benchmark swebench-live --time
uv run mcode compare --baseline-dir ./results/baseline --candidate-dir ./results/candidate
uv run mcode report --db-dir ./results --benchmark swebench-live --out ./results/report.html
uv run mcode merge-shards --out ./results/merged.db ./results/shard-0.db ./results/shard-1.db
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

## Launch

```bash
uv run mcode launch doctor bluevela --init --login <user>@<login-host>
uv run mcode launch sync bluevela
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
uv run mcode launch status
uv run mcode launch refresh
uv run mcode launch stop --all
```
