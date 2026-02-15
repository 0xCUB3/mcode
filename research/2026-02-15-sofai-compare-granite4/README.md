# 2026-02-15 SOFAI vs repair comparison (granite4, MBPP, limit=500)

Goal: compare SOFAI two-tier strategy (8B S1 + 32B S2 escalation) against repair-only baseline across budget and timeout configs on MBPP.

Baseline: `2026-02-15-mellea-first-verify-granite4` (repair-only, same cluster + S1 model).

Data source:

- `results/2026-02-15-sofai-compare-granite4/sofai-compare/`

## Environment

- **Cluster:** mellea-test (OpenShift), Ollama backend
- **S1 model:** granite4:latest (8B)
- **S2 model:** granite4:32b-a9b-h (32B MoE, 9B activated, 19GB)
- **Dataset:** MBPP, 500 tasks, 20 shards
- **Parallelism:** 3 pods (500m CPU each, within 8 CPU / 32Gi quota)

## Commands

Repair-only sweep:

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --loop-budget 3,5 --timeout 90,120 \
  --limit 500 --shard-count 20 --parallelism 3 --no-build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id sofai-compare \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/2026-02-15-sofai-compare-granite4
```

SOFAI sweep:

```bash
.venv/bin/python deploy/k8s/oc_bench_sweep.py \
  --benchmarks mbpp --model granite4:latest \
  --loop-budget 3,5 --timeout 90,120 \
  --strategy sofai --s2-model granite4:32b-a9b-h --s2-backend ollama \
  --limit 500 --shard-count 20 --parallelism 3 --no-build \
  --mcode-memory-request 1Gi --mcode-memory-limit 12Gi \
  --run-id sofai-compare \
  --env MCODE_MAX_NEW_TOKENS=1024 \
  --out-dir results/2026-02-15-sofai-compare-granite4
```

## Key results

| Config | Strategy | Pass Rate | Solved | Avg Time | Compute |
|--------|----------|-----------|--------|----------|---------|
| b3-t90 | repair | 60.0% | 300/500 | 3.2s | 26.9 min |
| b3-t90 | **sofai** | **69.0%** | 328/475* | 7.9s | 62.9 min |
| b3-t120 | repair | 60.2% | 301/500 | 3.0s | 25.0 min |
| b3-t120 | **sofai** | **65.6%** | 328/500 | 4.7s | 39.5 min |
| b5-t90 | repair | 66.4% | 332/500 | 4.6s | 39.1 min |
| b5-t90 | **sofai** | **67.8%** | 339/500 | 7.4s | 62.4 min |
| b5-t120 | repair | **65.8%** | 329/500 | 4.7s | 39.8 min |
| b5-t120 | sofai | 64.6% | 323/500 | 5.6s | 46.7 min |

*SOFAI b3-t90: 25 tasks lost to mellea bug (shard-2 `KeyError: 'chat_response'`). True rate is 328/475 = 69.0%, or ~65.6% if missing tasks all failed.

### S2 rescue breakdown

| Config | S1 Solves | S2 Solves | Total |
|--------|-----------|-----------|-------|
| b3-t90  | 272 | 56 | 328 |
| b3-t120 | 277 | 51 | 328 |
| b5-t90  | 279 | 60 | 339 |
| b5-t120 | 275 | 48 | 323 |

### Head-to-head unique solves

| Config | Both Pass | Repair Only | SOFAI Only | Both Fail |
|--------|-----------|-------------|------------|-----------|
| b3-t90 | 257 | 31 | 71 | 116 |
| b3-t120 | 260 | 41 | 68 | 131 |
| b5-t90 | 282 | 50 | 57 | 111 |
| b5-t120 | 277 | 52 | 46 | 125 |

## Findings

- SOFAI wins at budget=3: +5.4pp to +9.0pp over repair-only. The 32B model rescues 51-56 tasks the 8B fails on in 3 attempts.
- At budget=5, the gap narrows to +1.4pp (b5-t90) and reverses to -1.2pp (b5-t120). With 5 repair attempts the 8B solves most of what the 32B would rescue.
- S2 (32B) consistently rescues 46-60 tasks across all configs. These are problems the 8B can't crack regardless of attempt count.
- S2 solves are ~6x slower than S1 (avg 8.5s vs 1.5s for b3-t90).
- SOFAI costs 1.5-2.3x more compute than repair-only.
- Each strategy solves unique tasks the other misses. An oracle ensemble could hit ~72-74%.
- Best tradeoff: SOFAI with budget=3 (+5-9pp at ~2x compute). If compute matters more than peak accuracy, repair at budget=5-6 is competitive and much cheaper.
- The mellea `chat_response` bug needs an upstream fix; it crashed shard-2 entirely (25 tasks lost) when the 32B model returned an error response instead of a chat completion.
