# Upstream clean-head repair sampling on Qwen3.5 smoke

## Goal / scope

This entry is the first follow-up after the Mellea-first harness experiments made it clear that `mcode` itself should keep tracking upstream Mellea. I ran a fresh Blue Vela sweep from a clean worktree at pushed `HEAD`, then tested only upstream-compatible controls.

The two questions were straightforward. First, does any existing upstream-compatible Mellea knob improve Qwen3.5 smoke behavior on the clean baseline. Second, does a very small mcode-only malformed-argument retry help on top of that baseline without depending on fork-only `mellea.agent.*` APIs.

Rendered HTML report:

- `research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-21-upstream-clean-head-repair-ab/upstream-clean-head-repair-ab-report.html

## Environment + commands

Shared setup:

- model: `Qwen/Qwen3.5-35B-A3B`
- backend: `openai`
- target: Blue Vela
- benchmark: `swebench-lite` over the smoke-16 task list
- loop budget: `15`
- timeout: `300`
- clean worktree: `/tmp/mcode-upstream-head` at pushed `HEAD`

Commands used:

```bash
git worktree add /tmp/mcode-upstream-head HEAD
cd /tmp/mcode-upstream-head
uv sync --extra dev
uv run mcode launch sync bluevela
ssh skula@login3.bluevela.rmf.ibm.com 'bash -lc "cd /u/skula/mcode-launch && uv sync --extra swebench --extra datasets --extra dev"'
```

The remote `uv sync` line matters. `--all-extras` pulled in upstream `mellea[telemetry,hooks]==0.4.2`, which dragged in a `cpex` and protobuf combination that crashed on import. Restricting the remote env to the extras needed for the benchmark kept the clean-head runs valid.

Smoke commands:

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
```

## Key results

First four smoke tasks:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch |
|-|-|-|-|-|-|-|-|
| Clean control | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 |
| `--n-samples 2` | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 |
| `--sampling repair --sampling-budget 2` | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 |
| `--sampling rejection --sampling-budget 2` | 1/4 | 1 | 2 | 1 | 1 | 1 | 1 |
| `required_arg_repair_v1` | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 |
| `required_arg_repair_v1` + repair sampling | 2/4 | 1 | 1 | 2 | 1 | 0 | 1 |

Full smoke:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch |
|-|-|-|-|-|-|-|-|
| Clean control | 4/16 | 4 | 4 | 4 | 6 | 6 | 0 |
| Repair sampling, run 1 | 6/16 | 4 | 7 | 6 | 4 | 4 | 2 |
| Repair sampling, run 2 | 5/16 | 4 | 5 | 5 | 4 | 5 | 2 |
| Arg repair + repair sampling, run 1 | 6/16 | 4 | 5 | 6 | 4 | 4 | 2 |
| Arg repair + repair sampling, run 2 | 5/16 | 4 | 6 | 5 | 5 | 5 | 1 |

## Findings

The clean-head baseline matters. On upstream Mellea with only the needed benchmark extras installed remotely, the control run landed at `4/16`. That is the baseline to beat for upstream-compatible work. It is lower than the earlier `7/16` result on pushed `main`, which means environment shape and dependency shape still matter a lot here.

The best upstream-compatible signal is still repair sampling. `--sampling repair --sampling-budget 2` improved the first full run from `4/16` to `6/16`, and the rerun held `5/16`. That is the first control-loop knob in this branch that looks better than noise.

`--n-samples 2` and rejection sampling were both worse on the first-four slice. There is no reason to spend more Blue Vela budget on them right now.

The small malformed-argument retry was not a clear pass-rate win by itself. On the first-four slice it dropped from `2/4` control to `1/4`. Combined with repair sampling it did not raise pass count beyond repair alone, but it did tighten the failure mix a little by reducing `unverified_diff_discarded` and `zero_verification` in some runs.

That makes the next decision pretty simple. If we want an upstream-compatible change to keep testing, repair sampling is the real lead. The required-arg retry is still plausible as a quality improvement, but not benchmark-proven yet.

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
- `research/2026-04-21-upstream-clean-head-repair-ab/control-full/ab-qwen35-clean-control-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full/ab-qwen35-clean-repair-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-full-rerun/ab-qwen35-clean-repair-full-rerun-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full/ab-qwen35-clean-repair-arg-full-20260421.db`
- `research/2026-04-21-upstream-clean-head-repair-ab/repair-arg-full-rerun/ab-qwen35-clean-repair-arg-full-rerun-20260421.db`
