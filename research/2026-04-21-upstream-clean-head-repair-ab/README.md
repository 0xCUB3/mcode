# Upstream clean-head repair and multiturn sampling on Qwen3.5 smoke

## Goal / scope

This entry tracks the clean upstream-head Blue Vela sweep on `Qwen/Qwen3.5-35B-A3B`. The first question was whether any upstream-compatible Mellea knobs could beat the clean control without depending on fork-only `mellea.agent.*` code. Repair sampling was the first positive signal. After that, we added clean upstream-compatible wiring for upstream `MultiTurnStrategy`, confirmed it with multiple full-smoke reruns, and then ran a small `(loop_budget, sampling_budget)` matrix for multiturn.

Rendered HTML report:

- `research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html

## Environment + commands

Shared setup:

- model: `Qwen/Qwen3.5-35B-A3B`
- backend: `openai`
- benchmark: `swebench-lite` over the smoke-16 task list
- target: Blue Vela
- clean worktree: `/tmp/mcode-main-confirm` at pushed `main`

Bootstrap:

```bash
git worktree add -d /tmp/mcode-main-confirm origin/main
cd /tmp/mcode-main-confirm
uv sync --extra dev
uv run mcode launch sync bluevela
ssh skula@login3.bluevela.rmf.ibm.com 'bash -lc "cd /u/skula/mcode-launch && uv sync --extra swebench --extra datasets --extra dev"'
```

That remote `uv sync` line matters. `--all-extras` on upstream `mellea[telemetry,hooks]==0.4.2` pulled in a `cpex` and protobuf combination that crashed on import. Restricting the remote env to the extras needed for the benchmark kept the clean-head runs valid.

Representative commands used in this entry:

```bash
uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair-full-20260421.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-multiturn-full-20260421.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --loop-budget 15 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 3 \
  --on bluevela \
  --db experiments/results/ab-qwen35-matrix-15-3-20260422.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --loop-budget 20 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-matrix-20-2-20260422.db
```

## Key results

Earlier clean upstream-compatible sweeps:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| Clean control | 4/16 | 4 | 4 | 4 | 6 | 6 | 0 | 0 |
| Repair sampling, run 1 | 6/16 | 4 | 7 | 6 | 4 | 4 | 2 | 0 |
| Repair sampling, run 2 | 5/16 | 4 | 5 | 5 | 4 | 5 | 2 | 0 |
| Arg repair + repair sampling, run 1 | 6/16 | 4 | 5 | 6 | 4 | 4 | 2 | 0 |
| Arg repair + repair sampling, run 2 | 5/16 | 4 | 6 | 5 | 5 | 5 | 1 | 0 |

Confirmed multiturn runs:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| Multiturn, run 1 | 8/16 | 3 | 4 | 8 | 3 | 3 | 2 | 0 |
| Multiturn, run 2 | invalid | 0 | 16 | 0 | 0 | 0 | 0 | 16 |
| Multiturn, run 3 | 5/16 | 3 | 4 | 5 | 4 | 4 | 3 | 0 |
| Multiturn, run 4 | 7/16 | 4 | 5 | 7 | 4 | 4 | 1 | 0 |
| Arg repair + multiturn | 5/16 | 7 | 6 | 5 | 7 | 2 | 2 | 0 |

Multiturn `(loop_budget, sampling_budget)` matrix:

| Setting | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| `15-2` | 6/16 | 3 | 2 | 6 | 3 | 5 | 2 | 0 |
| `15-3` | 7/16 | 4 | 4 | 7 | 4 | 2 | 3 | 0 |
| `20-2` | 7/16 | 3 | 4 | 7 | 4 | 3 | 2 | 0 |
| `20-3` | invalid | 11 | 12 | 1 | 0 | 3 | 0 | 12 |

## Findings

Repair sampling still looks like the best purely low-risk baseline improvement over clean control. It consistently moved the valid full run from `4/16` into the `5-6/16` range. That is useful, but it is no longer the lead.

The real signal is upstream `multiturn`. We now have three valid full multiturn runs at `8/16`, `7/16`, and `5/16`, plus one invalid all-infra-failure rerun. That is enough to say the `8/16` result was not just a one-off accident. The setting is high variance, but it clearly shifts the distribution upward relative to the clean control and the repair-only runs.

The small malformed-argument retry does not earn its keep on pass rate. With repair sampling it stayed flat, and with multiturn it was worse than plain multiturn. I would keep that out of `main`.

The matrix says two more things. First, higher multiturn repair budget is not helping. `15-3` tied the better clean multiturn rerun at `7/16`, but `20-3` collapsed under infra failures and is not a serious candidate. Second, higher outer loop budget can help a little, but it does not obviously dominate. `20-2` tied `15-3` at `7/16`, while `15-2` dropped to `6/16` in this fresh batch.

So the current best honest summary is this:

- best valid single run: `8/16` with `multiturn` and budget `15-2`
- best stable-looking band: `7-8/16` is plausible, `5/16` still happens
- strongest next direction toward `10/16`: keep `sampling=multiturn`, favor budget `2`, and test adjacent changes that are orthogonal to sampling, not more repair budget

## Files

Included in this entry:

- `research/2026-04-21-upstream-clean-head-repair-ab/README.md`
- `research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html`
- `research/2026-04-21-upstream-clean-head-repair-ab/control-first4/ab-qwen35-clean-control-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/control-full/ab-qwen35-clean-control-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/n2-first4/ab-qwen35-clean-n2-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-first4/ab-qwen35-clean-repair-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full/ab-qwen35-clean-repair-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full-rerun/ab-qwen35-clean-repair-full-rerun-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/rejection-first4/ab-qwen35-clean-rejection-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/required-arg-first4/ab-qwen35-clean-required-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-first4/ab-qwen35-clean-repair-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full/ab-qwen35-clean-repair-arg-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full-rerun/ab-qwen35-clean-repair-arg-full-rerun-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-first4/ab-qwen35-clean-multiturn-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full/ab-qwen35-clean-multiturn-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full-rerun2/ab-qwen35-clean-multiturn-full-rerun2-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full-rerun3/ab-qwen35-clean-multiturn-full-rerun3-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-arg-first4/ab-qwen35-clean-multiturn-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-arg-full/ab-qwen35-clean-multiturn-arg-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-matrix-15-2/ab-qwen35-matrix-15-2-20260422.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-matrix-15-3/ab-qwen35-matrix-15-3-20260422.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-matrix-20-2/ab-qwen35-matrix-20-2-20260422.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-matrix-20-3/ab-qwen35-matrix-20-3-20260422.db`

Invalid infra-heavy runs kept out of the report tables:

- `experiments/results/ab-qwen35-clean-multiturn-full-rerun-20260421.db`
