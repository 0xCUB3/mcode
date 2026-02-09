# 2026-02-09 OpenShift confirm runs (granite4)

Goal: verify how stable the MBPP speed/accuracy tradeoff is across repeated runs, then spot-check transfer on HumanEval.

HTML snapshots:

- MBPP repeats: [`mbpp-sweep.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-02-09-oc-confirm-granite4/mbpp-sweep.html) ([source](mbpp-sweep.html))
- HumanEval spot: [`humaneval-sweep.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-02-09-oc-confirm-granite4/humaneval-sweep.html) ([source](humaneval-sweep.html))

Data sources:

- `results/oc-confirm/mbpp-repeats/20260209-070721/`
- `results/oc-confirm/mbpp-repeats/20260209-084546/`
- `results/oc-confirm/mbpp-repeats/20260209-095707/`
- `results/oc-confirm/humaneval-spot/20260209-111917/`

## Commands

MBPP confirm repeats (3x):

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --samples 2,3 --debug-iters 0,1 --timeout 60 \
  --limit 500 --shard-count 20 --parallelism 4 --no-build \
  --mcode-cpu-request 60m --mcode-memory-request 256Mi \
  --hold-cpu-request 20m --hold-memory-request 64Mi \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/oc-confirm/mbpp-repeats
```

HumanEval spot check:

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks humaneval --model granite4:latest \
  --samples 2,3 --debug-iters 1 --timeout 60 \
  --shard-count 20 --parallelism 4 --no-build \
  --mcode-cpu-request 60m --mcode-memory-request 256Mi \
  --hold-cpu-request 20m --hold-memory-request 64Mi \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/oc-confirm/humaneval-spot
```

Report build:

```bash
.venv/bin/mcode report --db-dir ./results/oc-confirm/mbpp-repeats --benchmark mbpp --out ./research/2026-02-09-oc-confirm-granite4/mbpp-sweep.html
.venv/bin/mcode report --db-dir ./results/oc-confirm/humaneval-spot --benchmark humaneval --out ./research/2026-02-09-oc-confirm-granite4/humaneval-sweep.html
```

## Key results

### MBPP (3 repeats, 1500 tasks per config)

| config | pass_rate | sec/solve | avg_s | p95_s | timed_out | passed/total |
|---|---:|---:|---:|---:|---:|---:|
| s=2 d=0 t=60s | 52.8% | 9.01 | 4.76 | 6.18 | 0.07% | 792/1500 |
| s=2 d=1 t=60s | 59.9% | 12.75 | 7.64 | 12.72 | 0.20% | 899/1500 |
| s=3 d=0 t=60s | 58.9% | 13.09 | 7.72 | 8.74 | 0.20% | 884/1500 |
| s=3 d=1 t=60s | 64.5% | 22.99 | 14.83 | 21.66 | 0.07% | 968/1500 |

### HumanEval (spot check, 164 tasks per config)

| config | pass_rate | sec/solve | avg_s | p95_s | timed_out | passed/total |
|---|---:|---:|---:|---:|---:|---:|
| s=2 d=1 t=60s | 78.7% | 6.35 | 5.00 | 13.74 | 0.61% | 129/164 |
| s=3 d=1 t=60s | 81.7% | 5.89 | 4.81 | 14.63 | 0.00% | 134/164 |

## Findings

- Timeouts are low and are not the primary source of misses in these runs.
- On MBPP, `s=2 d=1` is materially faster than `s=3 d=1` for a smaller pass-rate drop, so it remains the better default tradeoff.
- On MBPP, `s=3 d=1` still defines the max-accuracy mode, but with a clear speed penalty.
- On HumanEval spot-check, `s=3 d=1` improves pass rate and is also faster than `s=2 d=1` in this sample.
- Cross-benchmark behavior is not identical, so benchmark-specific defaults are justified.
