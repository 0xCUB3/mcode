# 2026-02-08 MBPP OpenShift focused rerun (granite4, limit=500)

Goal: validate the best speed/accuracy band from the earlier 18-config sweep on a larger slice.

HTML snapshot: [`mbpp-sweep.html`](https://htmlpreview.github.io/?https://raw.githubusercontent.com/0xCUB3/mcode/main/research/2026-02-08-mbpp-oc-focused-500-granite4/mbpp-sweep.html) ([source](mbpp-sweep.html))

Data source:

- `results/oc-sweep/20260208-154252/`

## Command

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --samples 2,3 --debug-iters 0,1 --timeout 60 \
  --limit 500 --shard-count 20 --parallelism 4 \
  --env MCODE_MAX_NEW_TOKENS=1024
```

Report build:

```bash
.venv/bin/mcode report --db-dir ./results/oc-sweep/20260208-154252 --benchmark mbpp --out ./results/oc-sweep/20260208-154252/mbpp-sweep.html
```

## Key results

| config | pass_rate | sec/solve | avg_s | p95_s | timed_out | passed/total |
|---|---:|---:|---:|---:|---:|---:|
| s=2 d=0 t=60s | 52.8% | 6.73 | 3.56 | 4.68 | 0.0% | 264/500 |
| s=2 d=1 t=60s | 60.0% | 8.87 | 5.32 | 9.28 | 0.0% | 300/500 |
| s=3 d=0 t=60s | 59.8% | 7.35 | 4.39 | 6.42 | 0.0% | 299/500 |
| s=3 d=1 t=60s | 68.0% | 9.70 | 6.60 | 14.04 | 0.0% | 340/500 |

## Findings

- Objective: timeout behavior is not the bottleneck (`0/2000` timed out across this run).
- Objective: increasing debug from `d=0` to `d=1` improves pass rate for both sample counts, but adds meaningful latency.
- Objective: `s=3 d=0` is slightly less accurate than `s=2 d=1` (59.8% vs 60.0%) while being faster (7.35 vs 8.87 sec/solve).
- Subjective: for balanced operation, `s=2 d=1 t=60` is currently the best midpoint in this focused set.
- Subjective: for max accuracy, `s=3 d=1 t=60` is best but should be opt-in because latency is highest in this set.
- Decision: set default to `s=2 d=1 t=60`, with `s=3 d=1 t=60` as an explicit accuracy mode.
