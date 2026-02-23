# 2026-02-20 Benchmark expansion sweeps (granite4)

Goal: compare granite4 pass rates across five new benchmarks (humaneval+, mbpp+, livecodebench, bigcodebench-complete, bigcodebench-instruct) to establish baseline numbers alongside existing humaneval/mbpp results.

HTML snapshot: [`sweep-report.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-02-20-benchmark-expansion-granite4/sweep-report.html) ([source](sweep-report.html))

Summary table source: [`results-summary.txt`](results-summary.txt)

Data source:

- `results/2026-02-20-benchmark-expansion-granite4/benchmark-expansion/`

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

**Loop budget scaling.** Retries help a lot on easy-to-medium benchmarks. mbpp+ jumps from 90% (b1) to 97% (b3) to 98% (b5), and humaneval+ from 65% to 79% to 85%. The gains from b3 to b5 are small relative to b1 to b3, suggesting diminishing returns past 3 attempts. On harder benchmarks (livecodebench, bigcodebench-complete) the absolute gains from retries are smaller (0.4% to 3.0% for livecodebench), indicating the model genuinely can't solve those problems rather than just making fixable mistakes.

**Timeout has little effect.** For humaneval+ and mbpp+, 60s and 120s produce nearly identical results because granite4 generates solutions in 1-4 seconds. Even on bigcodebench-complete (40-90s per solve), the two timeout levels are within noise. Timeout only matters for livecodebench where problems that do get solved sometimes need the full window.

**Benchmark difficulty spectrum.** The five benchmarks span a wide range:
- mbpp+ (90-98%): near-saturated, mostly useful as a sanity check
- humaneval+ (65-85%): good dynamic range, responds well to retries
- bigcodebench-complete (12-18%): substantially harder, requires library knowledge and longer solutions
- livecodebench (0.4-3.0%): competition-level problems, effectively out of reach for granite4 at 8B parameters

**Speed characteristics.** mbpp+ solves average under 1.2s even at b5. humaneval+ scales linearly with budget (1.8s at b1, 6.5s at b5). bigcodebench-complete is 10-20x slower (40-90s/solve) due to longer prompts and sandbox execution. livecodebench sec/solve numbers (260-527s) are inflated because unsolved problems burn the full timeout.

**Infrastructure issues.** bigcodebench-complete b3-t60 only completed 14/20 shards before OOMKills stalled progress. bigcodebench-complete b3-t120 through b5-t120 were skipped due to the same memory pressure. All bigcodebench-instruct configs were skipped entirely because shards hung on sandbox execution (not LLM inference or OOM).

**Root cause of bigcodebench hangs (post-mortem).** Three bugs in `ProcessSandbox` (the in-cluster fallback used when Docker is unavailable) combined to cause cascading resource exhaustion:

1. `proc.kill()` only killed the direct Python child, not grandchild processes. BigCodeBench tasks frequently use libraries that spawn threads or subprocesses (309 tasks use matplotlib, 152 use sklearn/joblib, 31 use subprocess directly). Without `start_new_session=True` and `os.killpg()`, orphaned child processes survived after each task timeout and accumulated over hundreds of tasks until the pod OOMed or exhausted PIDs.

2. No `MPLBACKEND=Agg` in the sandbox env. matplotlib (used by 309/1140 tasks) defaults to an interactive backend when no display variable is set. In a headless pod this either hangs waiting for a display or raises an error, but either way it slows down execution and can leave zombie processes. 19 tasks explicitly call `plt.show()`.

3. No network isolation. 123 tasks require libraries that make network calls (requests, urllib, bs4, socket). In the process sandbox, these are real TCP calls that hang on DNS/connection timeouts (30-120s each). The Docker sandbox blocks this with `network_disabled=True`, but the process sandbox had no equivalent. Instruct-mode code is more likely to attempt real network calls because the model gets less structured guidance.

All three issues are fixed in `process_sandbox.py`: process-group killing via `start_new_session=True` + `os.killpg()`, `MPLBACKEND=Agg` + `OPENBLAS_NUM_THREADS=1` + `MKL_NUM_THREADS=1` in env, and the Docker sandbox also got `MPLBACKEND=Agg` for consistency. Network isolation in the process sandbox would require user-namespace networking or iptables, which is out of scope for now.

## Conclusion

For granite4 at 8B parameters, humaneval+ and mbpp+ confirm competitive single-attempt pass rates (65% and 90%) that scale well with retries. The model hits a wall on competition-level problems (livecodebench) and library-heavy tasks (bigcodebench-complete). Budget=3 captures most of the retry benefit; budget=5 adds only 2-5pp. Timeout=60s is sufficient for all benchmarks except livecodebench.

Recommended default config for future granite4 runs: b3-t60. For cross-model comparisons, b1-t60 (cheapest) and b3-t60 (best tradeoff) are the two most useful data points.

Next steps: re-run bigcodebench-complete (full 20 shards) and bigcodebench-instruct now that the sandbox bugs are fixed. If pass rates remain low, it's the model, not infrastructure.
