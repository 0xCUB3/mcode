# 2026-02-08 MBPP OpenShift sweep (granite4)

Goal: tune for pass rate vs time-to-solve using `granite4:latest` on the OpenShift cluster (Ollama backend).

HTML snapshot: [`mbpp-sweep.html`](https://htmlpreview.github.io/?https://raw.githubusercontent.com/0xCUB3/mcode/main/research/2026-02-08-mbpp-oc-sweep-granite4/mbpp-sweep.html) ([source](mbpp-sweep.html))

Data sources:

- `results/oc-sweep/20260208-081832/` (most configs)
- `results/oc-sweep/20260208-105834/` (2 follow-up configs)

## Command

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --samples 1,2,3 --debug-iters 0,1,2 --timeout 60,120 \
  --limit 100 --shard-count 20 --parallelism 2 \
  --env MCODE_MAX_NEW_TOKENS=1024
```

Report build:

```bash
.venv/bin/mcode report --db-dir ./results/oc-sweep --benchmark mbpp --out ./results/oc-sweep/mbpp-sweep.html
```

This research entry snapshots that report to `./mbpp-sweep.html` for durability.

## Best tradeoffs (Pareto)

Configs where no other config is both faster and more accurate.

| config | pass_rate | sec/solve | avg_s | p95_s | passed/total |
|---|---:|---:|---:|---:|---:|
| s=1 d=0 t=60s | 43.0% | 1.34 | 0.58 | 1.28 | 43/100 |
| s=2 d=0 t=60s | 60.0% | 1.49 | 0.90 | 1.95 | 60/100 |
| s=2 d=0 t=120s | 61.0% | 1.74 | 1.06 | 2.29 | 61/100 |
| s=3 d=0 t=60s | 65.0% | 2.70 | 1.76 | 3.90 | 65/100 |
| s=3 d=1 t=60s | 74.0% | 3.23 | 2.39 | 7.46 | 74/100 |

## All configs (sorted by sec/solve)

| config | pass_rate | sec/solve | avg_s | p95_s | passed/total |
|---|---:|---:|---:|---:|---:|
| s=1 d=0 t=60s | 43.0% | 1.34 | 0.58 | 1.28 | 43/100 |
| s=2 d=0 t=60s | 60.0% | 1.49 | 0.90 | 1.95 | 60/100 |
| s=1 d=0 t=120s | 45.0% | 1.69 | 0.76 | 2.09 | 45/100 |
| s=2 d=0 t=120s | 61.0% | 1.74 | 1.06 | 2.29 | 61/100 |
| s=3 d=0 t=60s | 65.0% | 2.70 | 1.76 | 3.90 | 65/100 |
| s=2 d=1 t=120s | 65.0% | 3.00 | 1.95 | 4.82 | 65/100 |
| s=1 d=2 t=60s | 59.0% | 3.05 | 1.80 | 5.03 | 59/100 |
| s=2 d=1 t=60s | 63.0% | 3.15 | 1.98 | 5.45 | 63/100 |
| s=3 d=1 t=60s | 74.0% | 3.23 | 2.39 | 7.46 | 74/100 |
| s=1 d=2 t=120s | 54.0% | 3.57 | 1.93 | 5.01 | 54/100 |
| s=1 d=1 t=60s | 57.0% | 3.72 | 2.12 | 2.17 | 57/100 |
| s=2 d=2 t=60s | 58.0% | 5.26 | 3.05 | 8.58 | 58/100 |
| s=3 d=0 t=120s | 63.0% | 8.63 | 5.44 | 3.42 | 63/100 |
| s=2 d=2 t=120s | 72.0% | 9.94 | 7.16 | 9.11 | 72/100 |
| s=1 d=1 t=120s | 51.0% | 10.80 | 5.51 | 2.90 | 51/100 |
| s=3 d=2 t=60s | 74.0% | 11.63 | 8.60 | 18.31 | 74/100 |
| s=3 d=1 t=120s | 72.0% | 12.37 | 8.91 | 9.06 | 72/100 |
| s=3 d=2 t=120s | 71.0% | 18.61 | 13.21 | 51.73 | 71/100 |

## Findings

- Objective: `timeout=120` is usually a poor tradeoff here; it adds major latency for small pass-rate gains.
- Objective: `debug-iters=2` is generally not worth it on MBPP with this model; slowdown is large without consistent pass-rate improvement.
- Objective: best speed/accuracy band in this grid is `s=2 d=0 t=60`, `s=3 d=0 t=60`, `s=3 d=1 t=60`.
- Subjective: operator default should be `s=2 d=0 t=60` for a strong speed/accuracy balance.
- Subjective: `s=3 d=1 t=60` should be an opt-in accuracy mode due to latency cost.
- Subjective: confidence is medium because this was `limit=100`; ranking should be confirmed on a larger slice.
