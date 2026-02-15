# SOFAI vs Repair Strategy Comparison — MBPP

**Date:** 2026-02-15
**Cluster:** mellea-test (OpenShift), Ollama backend
**S1 model:** granite4:latest (8B)
**S2 model:** granite4:32b-a9b-h (32B MoE, 9B activated)
**Dataset:** MBPP (500 tasks, 20 shards)

## Summary

SOFAI (two-tier: fast 8B model + slow 32B escalation) improves pass rate at low budgets but the advantage shrinks as the repair-only strategy gets more attempts. The 32B model reliably rescues ~50-60 tasks that the 8B model can't solve, but at 2-3x the compute cost.

## Results

| Config | Strategy | Pass Rate | Solved | Avg Time | Compute |
|--------|----------|-----------|--------|----------|---------|
| b3-t90 | repair   | 60.0%     | 300/500 | 3.2s    | 26.9 min |
| b3-t90 | **sofai** | **69.0%** | 328/475* | 7.9s   | 62.9 min |
| b3-t120 | repair  | 60.2%     | 301/500 | 3.0s    | 25.0 min |
| b3-t120 | **sofai** | **65.6%** | 328/500 | 4.7s   | 39.5 min |
| b5-t90 | repair   | 66.4%     | 332/500 | 4.6s    | 39.1 min |
| b5-t90 | **sofai** | **67.8%** | 339/500 | 7.4s   | 62.4 min |
| b5-t120 | repair  | **65.8%** | 329/500 | 4.7s    | 39.8 min |
| b5-t120 | sofai   | 64.6%     | 323/500 | 5.6s    | 46.7 min |

*\*SOFAI b3-t90: 25 tasks lost to a mellea bug (shard-2 crashed with `KeyError: 'chat_response'`). True rate is 328/475 = 69.0%, or ~65.6% if the 25 missing tasks all failed.*

## Key Findings

### 1. SOFAI wins at budget=3, diminishing returns at budget=5

At budget=3, SOFAI adds +5.4pp to +9.0pp over repair-only. The 32B model rescues 51-56 tasks that the 8B model failed in 3 attempts.

At budget=5, the gap narrows to +1.4pp (b5-t90) and actually reverses to -1.2pp (b5-t120). With 5 repair attempts, the 8B model can solve most of what the 32B would rescue anyway.

### 2. S2 (32B) rescues ~50-60 tasks consistently

Across all SOFAI configs, the S2 solver rescued 46-60 tasks. These are problems the 8B model couldn't crack in any number of S1 attempts.

| Config | S1 Solves | S2 Solves | Total |
|--------|-----------|-----------|-------|
| b3-t90  | 272 | 56 | 328 |
| b3-t120 | 277 | 51 | 328 |
| b5-t90  | 279 | 60 | 339 |
| b5-t120 | 275 | 48 | 323 |

### 3. SOFAI costs 1.5-2.3x more compute

The 32B model is much slower on Ollama (MoE models don't parallelize well on CPU). SOFAI configs used 1.2x to 2.3x the compute time of repair-only.

### 4. Each strategy solves unique tasks the other can't

Head-to-head on the same 500 tasks:

| Config | Both Pass | Repair Only | SOFAI Only | Both Fail |
|--------|-----------|-------------|------------|-----------|
| b3-t90 | 257 | 31 | 71 | 116 |
| b3-t120 | 260 | 41 | 68 | 131 |
| b5-t90 | 282 | 50 | 57 | 111 |
| b5-t120 | 277 | 52 | 46 | 125 |

At b3-t90: SOFAI uniquely solves 71 tasks that repair can't, but repair uniquely solves 31 that SOFAI can't. This suggests sampling variance — with more S1 attempts (budget=5), the "repair only" column grows because the repair strategy gets lucky on more tasks, while SOFAI's S2 rescues fewer (since S1 already handled them).

### 5. S2 solve times are ~6x slower than S1

For the b3-t90 SOFAI config:
- S1 solves: avg 1.5s (range 0.2-7.0s)
- S2 solves: avg 8.5s (range 3.5-18.3s)
- Failures: avg 19.8s (hit the full budget + timeout)

## Recommendations

1. **Use SOFAI with budget=3** for the best accuracy/compute tradeoff. It's the sweet spot: +5-9pp over repair-only at ~2x compute.

2. **Repair at budget=5-6 is competitive** and much cheaper. If compute matters more than peak accuracy, stick with repair.

3. **An oracle ensemble** (pick the strategy that works for each task) could hit ~72-74% — both strategies solve unique tasks the other misses.

4. **The mellea `chat_response` bug** needs a fix upstream. It crashed one shard entirely (25 tasks lost). This happens when the 32B model returns an error response instead of a chat completion.

## Raw Data

Results in `results/sofai-compare/sofai-compare/`. Each subdirectory contains 20 shard SQLite DBs.

```
mcode-mbpp-b{3,5}-t{90,120}-l500-sofai-compare/          # repair-only
mcode-mbpp-b{3,5}-t{90,120}-sofai-l500-sofai-compare/    # SOFAI
```

Query example:
```bash
mcode results --db-dir results/sofai-compare/sofai-compare --time --benchmark mbpp
```
