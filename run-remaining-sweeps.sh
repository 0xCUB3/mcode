#!/bin/bash
set -euo pipefail

SWEEP=".venv/bin/python deploy/k8s/oc_bench_sweep.py"
COMMON="--model granite4:latest --loop-budget 1,3,5 --timeout 60,120 --shard-count 20 --parallelism 3 --no-build --stalled-seconds 1800 --mcode-memory-request 1Gi --mcode-memory-limit 12Gi --run-id benchmark-expansion --env MCODE_MAX_NEW_TOKENS=1024 --out-dir results/2026-02-20-benchmark-expansion-granite4"

echo "=== Waiting for EvalPlus sweep to finish ==="
while pgrep -f "oc_bench_sweep.*humaneval" > /dev/null 2>&1; do
  sleep 30
done

echo "=== EvalPlus done. Starting LiveCodeBench ==="
$SWEEP --benchmarks livecodebench $COMMON --env LCB_CUTOFF=2024-06-01

echo "=== LiveCodeBench done. Starting BigCodeBench ==="
$SWEEP --benchmarks bigcodebench-complete,bigcodebench-instruct $COMMON

echo "=== All sweeps complete ==="
