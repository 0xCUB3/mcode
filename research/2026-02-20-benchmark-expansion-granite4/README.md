# 2026-02-20 Benchmark expansion sweeps (granite4)

Goal: compare granite4 pass rates across five new benchmarks (humaneval+, mbpp+, livecodebench, bigcodebench-complete, bigcodebench-instruct) to establish baseline numbers alongside existing humaneval/mbpp results.

## Commands

Build the image first (only needed once, then use `--no-build` for subsequent runs):

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks humaneval+,mbpp+ --model granite4:latest \
  --loop-budget 1,3,5 --timeout 60,120 \
  --shard-count 20 --parallelism 3 --build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id benchmark-expansion \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/2026-02-20-benchmark-expansion-granite4
```

LiveCodeBench sweep (with LCB_CUTOFF=2024-06-01):

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks livecodebench --model granite4:latest \
  --loop-budget 1,3,5 --timeout 60,120 \
  --shard-count 20 --parallelism 3 --no-build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id benchmark-expansion \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --env LCB_CUTOFF=2024-06-01 \
  --out-dir results/2026-02-20-benchmark-expansion-granite4
```

BigCodeBench sweep (complete + instruct):

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks bigcodebench-complete,bigcodebench-instruct --model granite4:latest \
  --loop-budget 1,3,5 --timeout 60,120 \
  --shard-count 20 --parallelism 3 --no-build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id benchmark-expansion \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/2026-02-20-benchmark-expansion-granite4
```

Report commands:

```bash
.venv/bin/mcode results \
  --db-dir results/2026-02-20-benchmark-expansion-granite4/benchmark-expansion \
  --compare-configs --time \
  > research/2026-02-20-benchmark-expansion-granite4/results-summary.txt

.venv/bin/mcode report \
  --db-dir results/2026-02-20-benchmark-expansion-granite4/benchmark-expansion \
  --out research/2026-02-20-benchmark-expansion-granite4/sweep-report.html
```

## Key results

Compiled from:
- `results/2026-02-20-benchmark-expansion-granite4/benchmark-expansion/*/*.db`
- `research/2026-02-20-benchmark-expansion-granite4/results-summary.txt`
- `research/2026-02-20-benchmark-expansion-granite4/sweep-report.html`

| benchmark | config | pass_rate | sec/solve | passed/total |
|---|---|---:|---:|---:|
| humaneval+ | b1-t60 | 64.6% | 1.81 | 106/164 |
| humaneval+ | b1-t120 | 67.7% | 1.97 | 111/164 |
| humaneval+ | b3-t60 | 78.7% | 4.34 | 129/164 |
| humaneval+ | b3-t120 | 81.1% | 3.39 | 133/164 |
| humaneval+ | b5-t60 | 84.8% | 6.48 | 139/164 |
| humaneval+ | b5-t120 | 84.1% | 6.23 | 138/164 |
| mbpp+ | b1-t60 | 90.2% | 0.89 | 341/378 |
| mbpp+ | b1-t120 | 86.8% | 0.92 | 328/378 |
| mbpp+ | b3-t60 | 97.1% | 1.02 | 367/378 |
| mbpp+ | b3-t120 | 96.6% | 0.95 | 365/378 |
| mbpp+ | b5-t60 | 98.1% | 1.07 | 371/378 |
| mbpp+ | b5-t120 | 98.4% | 1.11 | 372/378 |
| livecodebench | b1-t60 | 0.4% | 524.05 | 2/511 |
| livecodebench | b1-t120 | 0.6% | 384.57 | 3/511 |
| livecodebench | b3-t60 | 2.0% | 260.19 | 10/511 |
| livecodebench | b3-t120 | 2.0% | 391.03 | 10/511 |
| livecodebench | b5-t60 | 2.7% | 527.16 | 14/511 |
| livecodebench | b5-t120 | 3.0% | 520.70 | 15/503 |
| bigcodebench-complete | b1-t60 | 11.8% | 45.30 | 132/1122 |
| bigcodebench-complete | b1-t120 | 12.6% | 39.84 | 144/1140 |
| bigcodebench-complete | b3-t60 | 18.0% (partial) | 91.51 | 141/784 (14 shards) |
| bigcodebench-complete | b3-t120 | skipped | - | - |
| bigcodebench-complete | b5-t60 | skipped | - | - |
| bigcodebench-complete | b5-t120 | skipped | - | - |
| bigcodebench-instruct | b1-t60 | skipped | - | - |
| bigcodebench-instruct | b1-t120 | skipped | - | - |
| bigcodebench-instruct | b3-t60 | skipped | - | - |
| bigcodebench-instruct | b3-t120 | skipped | - | - |
| bigcodebench-instruct | b5-t60 | skipped | - | - |
| bigcodebench-instruct | b5-t120 | skipped | - | - |

## Findings

- 2026-02-23 18:51 UTC: `bigcodebench-instruct` shard execution showed repeated freeze/retry behavior (especially around shard 9/12/13 paths) with intermittent OOM/forced-retry churn. To keep the sweep moving, we stopped `bigcodebench-instruct b5-t60` and all `t120` instruct runs (`b1-t120`, `b3-t120`, `b5-t120`) and treated them as skipped for this expansion pass.
