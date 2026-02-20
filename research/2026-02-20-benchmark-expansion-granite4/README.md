# 2026-02-20 Benchmark expansion sweeps (granite4)

Goal: compare granite4 pass rates across five new benchmarks (humaneval+, mbpp+, livecodebench, bigcodebench-complete, bigcodebench-instruct) to establish baseline numbers alongside existing humaneval/mbpp results.

## Commands

EvalPlus sweep (humaneval+ and mbpp+):

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks humaneval+,mbpp+ --model granite4:latest \
  --loop-budget 1,3,5 --timeout 60,120 \
  --shard-count 20 --parallelism 3 --no-build \
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

Report + merge commands:

```bash
.venv/bin/mcode merge-shards --db-dir results/2026-02-20-benchmark-expansion-granite4/benchmark-expansion --out results/2026-02-20-benchmark-expansion-granite4/merged.db
.venv/bin/mcode results --db results/2026-02-20-benchmark-expansion-granite4/merged.db --compare-configs --time > research/2026-02-20-benchmark-expansion-granite4/results-summary.txt
.venv/bin/mcode report --db results/2026-02-20-benchmark-expansion-granite4/merged.db --out research/2026-02-20-benchmark-expansion-granite4/sweep-report.html
```

## Key results

(pending sweep execution)

| benchmark | config | pass_rate | sec/solve | passed/total |
|---|---|---:|---:|---:|
| humaneval+ | b1-t60 | | | |
| humaneval+ | b1-t120 | | | |
| humaneval+ | b3-t60 | | | |
| humaneval+ | b3-t120 | | | |
| humaneval+ | b5-t60 | | | |
| humaneval+ | b5-t120 | | | |
| mbpp+ | b1-t60 | | | |
| mbpp+ | b1-t120 | | | |
| mbpp+ | b3-t60 | | | |
| mbpp+ | b3-t120 | | | |
| mbpp+ | b5-t60 | | | |
| mbpp+ | b5-t120 | | | |
| livecodebench | b1-t60 | | | |
| livecodebench | b1-t120 | | | |
| livecodebench | b3-t60 | | | |
| livecodebench | b3-t120 | | | |
| livecodebench | b5-t60 | | | |
| livecodebench | b5-t120 | | | |
| bigcodebench-complete | b1-t60 | | | |
| bigcodebench-complete | b1-t120 | | | |
| bigcodebench-complete | b3-t60 | | | |
| bigcodebench-complete | b3-t120 | | | |
| bigcodebench-complete | b5-t60 | | | |
| bigcodebench-complete | b5-t120 | | | |
| bigcodebench-instruct | b1-t60 | | | |
| bigcodebench-instruct | b1-t120 | | | |
| bigcodebench-instruct | b3-t60 | | | |
| bigcodebench-instruct | b3-t120 | | | |
| bigcodebench-instruct | b5-t60 | | | |
| bigcodebench-instruct | b5-t120 | | | |

## Findings

(pending sweep execution)
