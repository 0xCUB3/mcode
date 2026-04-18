# verify_rerank_v1 on the Blue Vela smoke-16 slice

## Goal / scope

This was a controller-side A/B of `verify_rerank_v1`, a harness experiment that forced `n_samples >= 3` and reranked candidate diffs by post-hoc verification signal instead of exact-diff majority vote.

The target was a real pass-rate gain on the existing 16-task Blue Vela smoke slice for both `Qwen/Qwen3.5-35B-A3B` and `google/gemma-4-31B-it`. The experiment did not deliver that.

## Environment + commands

Shared smoke settings:

- slice: `src/mcode/bench/fixtures/smoke-16.txt`
- budget: `15`
- timeout: `300`
- backend: Blue Vela via `uv run mcode bench smoke --on bluevela`

Control commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run mcode bench smoke --model Qwen/Qwen3.5-35B-A3B --on bluevela --db experiments/results/ab-qwen-control-verify-rerank-v1.db
UV_CACHE_DIR=/tmp/uv-cache uv run mcode bench smoke --model google/gemma-4-31B-it --on bluevela --db experiments/results/ab-gemma-control-verify-rerank-v1.db
```

Treatment commands:

```bash
MCODE_HARNESS_EXPERIMENTS=verify_rerank_v1 UV_CACHE_DIR=/tmp/uv-cache uv run mcode bench smoke --model Qwen/Qwen3.5-35B-A3B --on bluevela --db experiments/results/ab-qwen-verify-rerank-v1.db
MCODE_HARNESS_EXPERIMENTS=verify_rerank_v1 UV_CACHE_DIR=/tmp/uv-cache uv run mcode bench smoke --model google/gemma-4-31B-it --on bluevela --db experiments/results/ab-gemma-verify-rerank-v1.db
```

Source DBs:

- [`ab-qwen-control-verify-rerank-v1.db`](../../experiments/results/ab-qwen-control-verify-rerank-v1.db)
- [`ab-qwen-verify-rerank-v1.db`](../../experiments/results/ab-qwen-verify-rerank-v1.db)
- [`ab-gemma-control-verify-rerank-v1.db`](../../experiments/results/ab-gemma-control-verify-rerank-v1.db)
- [`ab-gemma-verify-rerank-v1.db`](../../experiments/results/ab-gemma-verify-rerank-v1.db)

## Key results

| Run | Passed | Total | Status |
|-|-|-|-|
| Qwen control | 4 | 16 | complete |
| Qwen `verify_rerank_v1` | 4 | 16 | complete |
| Gemma control | 6 | 16 | complete |
| Gemma `verify_rerank_v1` | 4 | 9 | incomplete, stalled at 9 tasks |

Qwen failure mix:

| Metric | Control | Treatment |
|-|-|-|
| `budget_exhausted` | 2 | 5 |
| `unverified_diff_discarded` | 8 | 1 |
| `wrong_patch_after_verification` | 2 | 6 |
| `submitted` | 4 | 4 |
| `zero_edit` | 2 | 5 |
| `zero_verification` | 2 | 7 |
| `verification_succeeded` | 6 | 10 |

Gemma partial comparison:

| Metric | Control | Treatment |
|-|-|-|
| Completed tasks | 16 | 9 |
| Passed | 6 | 4 |
| `budget_exhausted` | 2 | 3 |
| `unverified_diff_discarded` | 5 | 0 |
| `wrong_patch_after_verification` | 3 | 2 |
| `submitted` | 6 | 4 |
| `zero_edit` | 1 | 3 |
| `zero_verification` | 3 | 3 |
| `verification_succeeded` | 10 | 6 |

On the 9 task IDs that actually finished under Gemma treatment, Gemma control went `3/9` and treatment went `4/9`.

## Findings

- Qwen was flat on pass rate, `4/16` to `4/16`. The experiment reduced `unverified_diff_discarded`, but mostly converted those cases into `wrong_patch_after_verification` and extra `budget_exhausted` runs.
- Gemma control landed at `6/16`, which is better than the April 7 valid Gemma control run at `5/16`, but that improvement was in the control leg, not the experiment.
- Gemma treatment looked mildly positive on the first 9 overlapping tasks, `4/9` versus control's `3/9`, but the run stalled before finishing. That is not enough evidence to keep the experiment.
- There is no sign of a harness grand slam here. The experiment added complexity without a clean pass-rate win, so the `verify_rerank_v1` code was reverted.
- The one useful side effect from this session was operational: live local DB syncing for Blue Vela remote runs worked and was kept, because it makes long remote smokes observable while they are still running.
