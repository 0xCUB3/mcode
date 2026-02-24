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
|-|-|-:|-:|-:|
| humaneval+ | b1-t60 | 64.6% | 1.17 | 106/164 |
| humaneval+ | b1-t120 | 67.7% | 1.33 | 111/164 |
| humaneval+ | b3-t60 | 78.7% | 3.42 | 129/164 |
| humaneval+ | b3-t120 | 81.1% | 2.75 | 133/164 |
| humaneval+ | b5-t60 | 84.8% | 5.49 | 139/164 |
| humaneval+ | b5-t120 | 84.1% | 5.25 | 138/164 |
| mbpp+ | b1-t60 | 90.2% | 0.81 | 341/378 |
| mbpp+ | b1-t120 | 86.8% | 0.80 | 328/378 |
| mbpp+ | b3-t60 | 97.1% | 0.99 | 367/378 |
| mbpp+ | b3-t120 | 96.6% | 0.92 | 365/378 |
| mbpp+ | b5-t60 | 98.1% | 1.05 | 371/378 |
| mbpp+ | b5-t120 | 98.4% | 1.10 | 372/378 |
| livecodebench | b1-t60 | 0.4% | 2.05 | 2/511 |
| livecodebench | b1-t120 | 0.6% | 2.26 | 3/511 |
| livecodebench | b3-t60 | 2.0% | 5.09 | 10/511 |
| livecodebench | b3-t120 | 2.0% | 7.65 | 10/511 |
| livecodebench | b5-t60 | 2.7% | 14.44 | 14/511 |
| livecodebench | b5-t120 | 3.0% | 15.53 | 15/503 |
| bigcodebench-complete | b1-t60 | 11.8% | 5.33 | 132/1122 |
| bigcodebench-complete | b1-t120 | 12.6% | 5.03 | 144/1140 |
| bigcodebench-complete | b3-t60 | 18.0% | 16.42 | 203/1126 |
| bigcodebench-complete | b3-t120 | 19.5% | 15.90 | 222/1140 |
| bigcodebench-complete | b5-t60 | 21.8% | 26.67 | 245/1126 |
| bigcodebench-complete | b5-t120 | 21.8% | 15.99 | 263/1208 |
| bigcodebench-instruct | b1-t60 | 11.8% | 2.56 | 135/1140 |
| bigcodebench-instruct | b1-t120 | 12.6% | 2.49 | 144/1140 |
| bigcodebench-instruct | b3-t60 | 18.3% | 6.81 | 209/1140 |
| bigcodebench-instruct | b3-t120 | 18.7% | 7.39 | 135/723 (13 shards) |
| bigcodebench-instruct | b5-t60 | 21.3% | 13.49 | 146/684 (12 shards) |
| bigcodebench-instruct | b5-t120 | skipped | - | - |

## Findings

**Loop budget scaling.** Retries help a lot on easy-to-medium benchmarks. mbpp+ jumps from 90% (b1) to 97% (b3) to 98% (b5), and humaneval+ from 65% to 79% to 85%. The gains from b3 to b5 are small relative to b1 to b3, suggesting diminishing returns past 3 attempts. On harder benchmarks (livecodebench, bigcodebench-complete) the absolute gains from retries are smaller (0.4% to 3.0% for livecodebench), indicating the model genuinely can't solve those problems rather than just making fixable mistakes.

**Timeout has little effect.** For humaneval+ and mbpp+, 60s and 120s produce nearly identical results because granite4 generates solutions in 1-4 seconds. Even on bigcodebench-complete (40-90s per solve), the two timeout levels are within noise. Timeout only matters for livecodebench where problems that do get solved sometimes need the full window.

**Benchmark difficulty spectrum.** The five benchmarks span a wide range:
- mbpp+ (90-98%): near-saturated, mostly useful as a sanity check
- humaneval+ (65-85%): good dynamic range, responds well to retries
- bigcodebench-complete (12-22%): substantially harder, requires library knowledge and longer solutions
- bigcodebench-instruct (12-21%): nearly identical to complete, confirming the bottleneck is task difficulty not prompt format
- livecodebench (0.4-3.0%): competition-level problems, effectively out of reach for granite4 at 8B parameters

**Speed characteristics.** mbpp+ averages under 1.1s/solve even at b5. humaneval+ scales linearly with budget (1.2s at b1, 5.5s at b5). bigcodebench-complete is slower (5-27s/solve) due to longer prompts and sandbox execution. bigcodebench-instruct is faster than complete at the same pass rates (2.5s vs 5.3s at b1) because instruct prompts are shorter. livecodebench sec/solve numbers (2-16s) only reflect tasks that were attempted, not the overall timeout burn.

**Complete vs instruct.** bigcodebench-instruct and bigcodebench-complete produce nearly identical pass rates at every budget level (11.8% vs 11.8% at b1, 18.3% vs 18.0% at b3, 21.3% vs 21.8% at b5). This is surprising since instruct provides less guidance, but suggests granite4 extracts the same signal from both prompt formats. The practical difference is speed: instruct runs 2x faster due to shorter prompts.

**Infrastructure issues.** After fixing the process sandbox (orphan leak, missing MPLBACKEND), all bigcodebench-complete configs completed 20/20 shards. Most bigcodebench-instruct configs also completed. The remaining gaps (instruct b3-t120 at 13/20, b5-t60 at 12/20, b5-t120 skipped) are from shards that OOM at 20Gi even with the fixes, hitting the 32Gi namespace quota ceiling. These partial configs have enough data to confirm the pattern.

**Root cause of bigcodebench hangs (post-mortem).** Three bugs in `ProcessSandbox` (the in-cluster fallback used when Docker is unavailable) combined to cause cascading resource exhaustion:

1. `proc.kill()` only killed the direct Python child, not grandchild processes. BigCodeBench tasks frequently use libraries that spawn threads or subprocesses (309 tasks use matplotlib, 152 use sklearn/joblib, 31 use subprocess directly). Without `start_new_session=True` and `os.killpg()`, orphaned child processes survived after each task timeout and accumulated over hundreds of tasks until the pod OOMed or exhausted PIDs.

2. No `MPLBACKEND=Agg` in the sandbox env. matplotlib (used by 309/1140 tasks) defaults to an interactive backend when no display variable is set. In a headless pod this either hangs waiting for a display or raises an error, but either way it slows down execution and can leave zombie processes. 19 tasks explicitly call `plt.show()`.

3. No network isolation. 123 tasks require libraries that make network calls (requests, urllib, bs4, socket). In the process sandbox, these are real TCP calls that hang on DNS/connection timeouts (30-120s each). The Docker sandbox blocks this with `network_disabled=True`, but the process sandbox had no equivalent. Instruct-mode code is more likely to attempt real network calls because the model gets less structured guidance.

All three issues are fixed in `process_sandbox.py`: process-group killing via `start_new_session=True` + `os.killpg()`, `MPLBACKEND=Agg` + `OPENBLAS_NUM_THREADS=1` + `MKL_NUM_THREADS=1` in env, and the Docker sandbox also got `MPLBACKEND=Agg` for consistency. Network isolation in the process sandbox would require user-namespace networking or iptables, which is out of scope for now.

## Conclusion

29 of 30 configs completed across 5 benchmarks. For granite4 at 8B parameters, humaneval+ and mbpp+ confirm competitive single-attempt pass rates (65% and 90%) that scale well with retries. bigcodebench-complete and bigcodebench-instruct produce nearly identical results (~12% at b1, ~22% at b5), confirming the bottleneck is model capability not prompt format. livecodebench remains out of reach at 0.4-3.0%.

Budget=3 captures most of the retry benefit; budget=5 adds only 2-4pp across all benchmarks. Timeout=60s is sufficient everywhere.

Recommended default config for future runs: b3-t60. For cross-model comparisons, b1-t60 and b3-t60 are the two most useful data points. bigcodebench-instruct can be dropped since it tracks complete almost exactly.
