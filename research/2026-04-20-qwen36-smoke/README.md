# Qwen3.6-35B-A3B smoke on Blue Vela

## Goal / scope

This entry now covers both the original Qwen3.6 smoke baseline and the first follow-up multiturn sweep. The baseline result was puzzling because Qwen3.6 did not beat the older Qwen3.5 best. The follow-up question was whether the same upstream-compatible multiturn settings that helped Qwen3.5 would unlock the newer model.

Rendered HTML report:

- `research/2026-04-20-qwen36-smoke/qwen36-smoke-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-20-qwen36-smoke/qwen36-smoke-report.html

## Setup

Shared setup:

- model: `Qwen/Qwen3.6-35B-A3B`
- serving: vLLM `v0.19.0`
- parser flags: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
- tensor parallel: `2`
- max model len: `262144`
- benchmark: `uv run mcode bench swebench-lite` over the smoke-16 task list
- backend: `openai`
- target: Blue Vela

Launch support:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
```

For the follow-up runs, the healthy endpoint came from a fresh Blue Vela launch and resolved to:

- `http://p5-r08-n4.bluevela.rmf.ibm.com:8321/v1`

## Commands

Original baseline:

```bash
OPENAI_BASE_URL=http://p1-r15-n1.bluevela.rmf.ibm.com:8321/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --db experiments/results/smoke-bluevela-qwen36-20260420.db
```

Multiturn follow-up runs from the clean main worktree:

```bash
uv run mcode launch sync bluevela
ssh skula@login3.bluevela.rmf.ibm.com 'bash -lc "cd /u/skula/mcode-launch && uv sync --extra swebench --extra datasets --extra dev"'

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B \
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
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen36-multiturn-15-2-20260422.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B \
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
  --db experiments/results/ab-qwen36-multiturn-15-3-20260422.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B \
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
  --db experiments/results/ab-qwen36-multiturn-20-2-20260422.db
```

## Results

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch |
|-|-|-|-|-|-|-|-|
| Baseline smoke | 4/16 | 6 | 4 | 4 | 7 | 4 | 1 |
| Multiturn `15-2` | 4/16 | 7 | 6 | 4 | 7 | 3 | 2 |
| Multiturn `15-3` | 5/16 | 6 | 5 | 5 | 7 | 2 | 2 |
| Multiturn `20-2` | 8/16 | 4 | 2 | 8 | 4 | 2 | 2 |

Submitted tasks for the best run, `20-2`:

- `astropy__astropy-12907`
- `astropy__astropy-13236`
- `astropy__astropy-13453`
- `astropy__astropy-13579`
- `astropy__astropy-14309`
- `scikit-learn__scikit-learn-13328`
- `sphinx-doc__sphinx-8120`
- `sympy__sympy-13877`

## Findings

The weird part turned out to be real. Qwen3.6 was not obviously better than Qwen3.5 on the plain smoke path. The original baseline stayed at `4/16`, and the first multiturn setting, `15-2`, also stayed at `4/16`. So the model upgrade by itself did not solve the control-loop problem.

What changed the picture was the same sort of orthogonal sweep that helped us on Qwen3.5. A slightly larger multiturn repair budget, `15-3`, moved Qwen3.6 to `5/16`. Then a larger outer loop budget with the smaller multiturn repair budget, `20-2`, jumped all the way to `8/16`.

That means Qwen3.6 does seem capable of beating the old Qwen3.5 baseline, but only when we give it a loop shape that lets it recover from failed verification attempts without pushing repair depth too hard. The sweet spot here looks different from the Qwen3.5 one. On Qwen3.5 the best stable-looking band was around `15-3` and `20-2`. On Qwen3.6 the clear leader so far is `20-2`.

The most important change in the failure mix is not just the pass count. `20-2` cut `zero_edit` from `7` to `4` and `zero_verification` from `6` to `2` relative to the `15-2` multiturn run. That says the bigger win is better conversion from searching into real edit-and-verify loops, not just luck on final patches.

This is only one valid `20-2` run, so I would not call `8/16` the stable expectation yet. But it is enough to justify iterating on Qwen3.6 instead of writing it off.

## Files

Included in this entry:

- `research/2026-04-20-qwen36-smoke/README.md`
- `research/2026-04-20-qwen36-smoke/qwen36-smoke-report.html`
- `research/2026-04-20-qwen36-smoke/baseline/smoke-bluevela-qwen36-20260420.db`
- `research/2026-04-20-qwen36-smoke/multiturn-15-2/ab-qwen36-multiturn-15-2-20260422.db`
- `research/2026-04-20-qwen36-smoke/multiturn-15-3/ab-qwen36-multiturn-15-3-20260422.db`
- `research/2026-04-20-qwen36-smoke/multiturn-20-2/ab-qwen36-multiturn-20-2-20260422.db`
