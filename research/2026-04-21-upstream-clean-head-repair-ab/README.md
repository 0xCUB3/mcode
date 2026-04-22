# Upstream clean-head repair and multiturn sampling on Qwen3.5 smoke

## Goal / scope

This entry extends the clean upstream-head Blue Vela sweep on `Qwen/Qwen3.5-35B-A3B`. The original question was whether any upstream-compatible Mellea knobs could beat the clean control without depending on fork-only `mellea.agent.*` code. The first positive signal was repair sampling. After that, we added a small amount of clean upstream-compatible wiring for upstream `MultiTurnStrategy` and checked whether it was a real step up or just variance.

Rendered HTML report:

- `research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html

## Environment + commands

Shared setup:

- model: `Qwen/Qwen3.5-35B-A3B`
- backend: `openai`
- benchmark: `swebench-lite` over the smoke-16 task list
- loop budget: `15`
- timeout: `300`
- target: Blue Vela
- clean worktree: `/tmp/mcode-upstream-head` at pushed `HEAD`

Bootstrap:

```bash
git worktree add /tmp/mcode-upstream-head HEAD
cd /tmp/mcode-upstream-head
uv sync --extra dev
uv run mcode launch sync bluevela
ssh skula@login3.bluevela.rmf.ibm.com 'bash -lc "cd /u/skula/mcode-launch && uv sync --extra swebench --extra datasets --extra dev"'
```

The remote `uv sync` line matters. `--all-extras` on upstream `mellea[telemetry,hooks]==0.4.2` pulled in a `cpex` and protobuf combination that crashed on import. Restricting the remote env to the extras needed for the benchmark kept the clean-head runs valid.

Sampling sweeps on the first four smoke tasks:

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-control-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --n-samples 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-n2-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 3 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair3-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 4 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair4-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling rejection \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-rejection-first4-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-required-arg-first4-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair-arg-first4-20260421.db

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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-multiturn-first4-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-multiturn-arg-first4-20260421.db
```

Promoted full-slice runs:

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
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-control-full-20260421.db

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
  --sampling repair \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-repair-full-rerun-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --db experiments/results/ab-qwen35-clean-repair-arg-full-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --db experiments/results/ab-qwen35-clean-repair-arg-full-rerun-20260421.db

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
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling multiturn \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-clean-multiturn-full-rerun-20260421.db

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
  --db experiments/results/ab-qwen35-clean-multiturn-full-rerun2-20260421.db

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
  --db experiments/results/ab-qwen35-clean-multiturn-full-rerun3-20260421.db

MCODE_HARNESS_EXPERIMENTS=required_arg_repair_v1 \
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
  --db experiments/results/ab-qwen35-clean-multiturn-arg-full-20260421.db
```

## Key results

First four smoke tasks:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| Clean control | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| `--n-samples 2` | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| `--sampling repair --sampling-budget 2` | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| `--sampling repair --sampling-budget 3` | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| `--sampling repair --sampling-budget 4` | 1/4 | 1 | 2 | 1 | 1 | 2 | 0 | 0 |
| `--sampling rejection --sampling-budget 2` | 1/4 | 1 | 2 | 1 | 1 | 1 | 1 | 0 |
| `required_arg_repair_v1` | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| `required_arg_repair_v1` + repair sampling | 2/4 | 1 | 1 | 2 | 1 | 0 | 1 | 0 |
| `--sampling multiturn --sampling-budget 2` | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| `required_arg_repair_v1` + multiturn | 2/4 | 1 | 0 | 2 | 1 | 1 | 0 | 0 |

Full smoke:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| Clean control | 4/16 | 4 | 4 | 4 | 6 | 6 | 0 | 0 |
| Repair sampling, run 1 | 6/16 | 4 | 7 | 6 | 4 | 4 | 2 | 0 |
| Repair sampling, run 2 | 5/16 | 4 | 5 | 5 | 4 | 5 | 2 | 0 |
| Arg repair + repair sampling, run 1 | 6/16 | 4 | 5 | 6 | 4 | 4 | 2 | 0 |
| Arg repair + repair sampling, run 2 | 5/16 | 4 | 6 | 5 | 5 | 5 | 1 | 0 |
| Multiturn, run 1 | 8/16 | 3 | 4 | 8 | 3 | 3 | 2 | 0 |
| Multiturn, run 2 | invalid | 16 | 16 | 0 | 0 | 0 | 0 | 16 |
| Multiturn, run 3 | 5/16 | 3 | 4 | 5 | 4 | 4 | 3 | 0 |
| Multiturn, run 4 | 7/16 | 4 | 5 | 7 | 4 | 4 | 1 | 0 |
| Arg repair + multiturn | 5/16 | 7 | 6 | 5 | 7 | 2 | 2 | 0 |

## Findings

Repair sampling remains the strongest fully upstream-compatible baseline improvement over the clean control. It turned the valid full run from `4/16` to `6/16`, and the rerun held `5/16`. Increasing the repair budget above `2` did not help on the first four tasks, which is a good sign that the easy repair win saturates quickly.

The important new result is multiturn. One valid full run reached `8/16`, which is the best single upstream-compatible smoke result we have seen. A later valid rerun hit `7/16`. That is enough to treat multiturn as real signal, not just a one-off spike. There was also one invalid all-`infra_failure` rerun and one weaker valid rerun at `5/16`, so variance is still high, but the center of gravity looks stronger than repair.

The malformed-argument retry still does not earn its keep on pass rate. With repair sampling it stayed flat, and with multiturn it got worse, `7/16` zero-edit and only `5/16` passed. I would keep that code out of `main` for now.

The practical read is simple. If we want the highest-value upstream-compatible next move, it is upstream `MultiTurnStrategy`, not higher turn budget, not more repair budget, and not malformed-arg retry.

That still does not prove the benchmark will sit stably at `8/16`, let alone `10/16`. But it is the best current direction.

## Files

Included in this entry:

- `research/2026-04-21-upstream-clean-head-repair-ab/README.md`
- `research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html`
- `research/2026-04-21-upstream-clean-head-repair-ab/control-first4/ab-qwen35-clean-control-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/n2-first4/ab-qwen35-clean-n2-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-first4/ab-qwen35-clean-repair-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/rejection-first4/ab-qwen35-clean-rejection-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/required-arg-first4/ab-qwen35-clean-required-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-first4/ab-qwen35-clean-repair-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-first4/ab-qwen35-clean-multiturn-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-arg-first4/ab-qwen35-clean-multiturn-arg-first4-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/control-full/ab-qwen35-clean-control-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full/ab-qwen35-clean-repair-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full-rerun/ab-qwen35-clean-repair-full-rerun-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full/ab-qwen35-clean-repair-arg-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full-rerun/ab-qwen35-clean-repair-arg-full-rerun-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full/ab-qwen35-clean-multiturn-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full-rerun2/ab-qwen35-clean-multiturn-full-rerun2-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-full-rerun3/ab-qwen35-clean-multiturn-full-rerun3-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/multiturn-arg-full/ab-qwen35-clean-multiturn-arg-full-20260421.db`

The invalid all-`infra_failure` multiturn rerun is intentionally left out of the report bundle:

- `experiments/results/ab-qwen35-clean-multiturn-full-rerun-20260421.db`
