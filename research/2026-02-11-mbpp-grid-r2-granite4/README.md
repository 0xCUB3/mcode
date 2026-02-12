# 2026-02-11 MBPP OpenShift grid rerun (granite4, limit=500)

Goal: run a full 18-config MBPP grid on OpenShift and quantify pass-rate vs speed with timeout visibility.

HTML snapshot: [`mbpp-sweep.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-02-11-mbpp-grid-r2-granite4/mbpp-sweep.html) ([source](mbpp-sweep.html))

Data source:

- `results/oc-confirm/20260211-mbpp-grid-r2/`

## Commands

Run/resume on OpenShift:

```bash
nohup .venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --samples 1,2,3 --debug-iters 0,1,2 --timeout 60,90 \
  --limit 500 --shard-count 20 --parallelism 3 --no-build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id 20260211-mbpp-grid-r2 --resume \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/oc-confirm \
  > results/oc-confirm/run-20260211-mbpp-grid-r2.log 2>&1 &
```

Report + summary:

```bash
.venv/bin/mcode results --db-dir ./results/oc-confirm/20260211-mbpp-grid-r2 --benchmark mbpp --compare-samples --time > ./results/oc-confirm/20260211-mbpp-grid-r2/results-summary.txt
.venv/bin/mcode report --db-dir ./results/oc-confirm/20260211-mbpp-grid-r2 --benchmark mbpp --out ./results/oc-confirm/20260211-mbpp-grid-r2/report.html
cp ./results/oc-confirm/20260211-mbpp-grid-r2/report.html ./research/2026-02-11-mbpp-grid-r2-granite4/mbpp-sweep.html
cp ./results/oc-confirm/20260211-mbpp-grid-r2/results-summary.txt ./research/2026-02-11-mbpp-grid-r2-granite4/results-summary.txt
```

## Key results

| config | pass_rate | sec/solve | avg_s | p95_s | timed_out | passed/total |
|---|---:|---:|---:|---:|---:|---:|
| s=2 d=0 t=60s | 54.4% | 5.59 | 3.04 | 5.17 | 0.2% | 272/500 |
| s=3 d=0 t=60s | 59.8% | 8.92 | 5.33 | 6.11 | 0.0% | 299/500 |
| s=3 d=1 t=60s | 67.6% | 11.64 | 7.87 | 14.00 | 0.0% | 338/500 |
| s=3 d=2 t=60s | 67.8% | 14.98 | 10.16 | 22.31 | 0.0% | 335/494 |
| s=3 d=0 t=90s | 60.8% | 6.67 | 4.06 | 6.16 | 0.0% | 293/482 |
| s=2 d=2 t=90s | 62.8% | 19.53 | 12.26 | 19.32 | 0.6% | 314/500 |

Full grouped output is in `results-summary.txt`.

## Findings

- Speed-first winner is `s=2 d=0 t=60` (5.59 sec/solve, 54.4% pass).
- Best complete high-accuracy point is `s=3 d=1 t=60` (67.6% pass at 11.64 sec/solve).
- Highest measured pass rate is `s=3 d=2 t=60` (67.8%), but this config has incomplete coverage (`494` tasks), so treat it as provisional.
- Timeout volume is low overall (`17/8976`, 0.19%), so misses are mostly non-timeout failures.
- Increasing `samples` raises pass rate in this run (`s=1` mean 46.0% → `s=3` mean 65.1%) and improves the best available speed/accuracy frontier.
- `timeout=90` does not show a consistent gain over `timeout=60`; keep `60` as default unless a benchmark-specific run proves otherwise.
