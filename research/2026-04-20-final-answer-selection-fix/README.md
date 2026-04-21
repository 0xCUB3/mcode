# Final answer tool selection fix on Qwen3.5 smoke

## Goal / scope

This entry covers the fix that made the Mellea ReAct driver read the actual `final_answer` tool response instead of blindly taking the first tool response in the turn. The code change shipped in `47c1e290d907bd550ce4457bcd49052b3d3a729f`.

The benchmark question was simple. Keep the correctness fix, then check whether a few generic control-loop ideas help or hurt Qwen3.5 on the Blue Vela smoke slice, without adding model-specific hacks.

Rendered HTML report:

- `research/2026-04-20-final-answer-selection-fix/final-answer-selection-fix-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-20-final-answer-selection-fix/final-answer-selection-fix-report.html

## Environment + commands

Shared setup:

- model: `Qwen/Qwen3.5-35B-A3B`
- backend: `openai`
- target: Blue Vela
- benchmark: `uv run mcode bench smoke`
- loop budget: `15`
- timeout: `300`

Commands used:

```bash
uv run mcode launch sync bluevela

uv run mcode bench smoke \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --on bluevela \
  --db experiments/results/ab-qwen35-control-full-20260420.db

uv run mcode bench smoke \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --harness-experiments mid_edit_nudge_v1 \
  --on bluevela \
  --db experiments/results/ab-qwen35-mid-edit-full-20260420.db

uv run mcode bench smoke \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --harness-experiments mid_edit_nudge_v1,post_edit_verify_nudge_v1 \
  --on bluevela \
  --db experiments/results/ab-qwen35-mid-edit-verify-full-20260420.db
```

We also ran 4-task diagnostic checks on the first four smoke tasks:

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
  --db experiments/results/ab-qwen35-control-first4-20260420.db

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
  --harness-experiments mid_edit_nudge_v1 \
  --on bluevela \
  --db experiments/results/ab-qwen35-mid-edit-first4-20260420.db

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
  --harness-experiments mid_edit_nudge_v1,finalizer_guard_v1 \
  --on bluevela \
  --db experiments/results/ab-qwen35-mid-edit-finalizer-first4-20260420.db
```

## Results

Full smoke results:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|
| Control | 7/16 | 5 | 5 | 7 | 5 | 2 | 2 | 0 |
| `mid_edit_nudge_v1` | 4/16 | 1 | 5 | 4 | 3 | 8 | 1 | 0 |
| `mid_edit_nudge_v1,post_edit_verify_nudge_v1` | 2/16 | 5 | 5 | 2 | 1 | 6 | 3 | 4 |

First four smoke tasks only:

| Run | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch |
|-|-|-|-|-|-|-|-|
| Control | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 |
| `mid_edit_nudge_v1` | 2/4 | 1 | 1 | 2 | 1 | 1 | 0 |
| `mid_edit_nudge_v1,finalizer_guard_v1` | 1/4 | 1 | 1 | 1 | 1 | 1 | 1 |

Valid full-control submitted tasks:

- `astropy__astropy-12907`
- `astropy__astropy-13453`
- `astropy__astropy-13579`
- `astropy__astropy-14309`
- `scikit-learn__scikit-learn-13328`
- `sphinx-doc__sphinx-8120`
- `sympy__sympy-13877`

One later clean rerun on the same code path, `experiments/results/qwen35-control-final-clean-20260420.db`, went `0/16` with `16` infra failures from Blue Vela podman blob writes. That run is excluded from the comparison tables because it does not say anything useful about harness behavior.

## Findings

The fix itself is still worth keeping. It corrects a real logic bug in the default Mellea driver, and the regression test now covers the case where `final_answer` is not the first tool returned in a turn.

The control run after the fix reached `7/16`, which is better than the earlier `6/16` smoke result we had been treating as the best Qwen3.5 baseline. I do not want to over-claim causality from one smoke run, especially since a later rerun was dominated by cluster-side podman failures, but the kept code path is at least not hurting the benchmark.

The generic mid-edit nudge looked good on the first four tasks and bad on the full sixteen. It reduced zero-edit wandering, but most of that gain turned into extra `unverified_diff_discarded` outcomes later in the slice. In other words, it pushed the model into making more diffs without getting enough of them across verification.

Adding more late-loop pressure made things worse. The finalizer guard did not improve the early slice, and the verify nudge variant was clearly negative on the full smoke. We discarded those ideas.

The practical takeaway is that the kept change is the small correctness fix in the ReAct driver. The next round should stay Mellea-first and focus on runtime-level progress detection or tool-surface shaping, not more prompt nudges in mcode.

## Files

Included in this entry:

- `research/2026-04-20-final-answer-selection-fix/README.md`
- `research/2026-04-20-final-answer-selection-fix/final-answer-selection-fix-report.html`
- `research/2026-04-20-final-answer-selection-fix/control-first4/ab-qwen35-control-first4-20260420.db`
- `research/2026-04-20-final-answer-selection-fix/mid-edit-first4/ab-qwen35-mid-edit-first4-20260420.db`
- `research/2026-04-20-final-answer-selection-fix/mid-edit-finalizer-first4/ab-qwen35-mid-edit-finalizer-first4-20260420.db`
- `research/2026-04-20-final-answer-selection-fix/control-full/ab-qwen35-control-full-20260420.db`
- `research/2026-04-20-final-answer-selection-fix/mid-edit-full/ab-qwen35-mid-edit-full-20260420.db`
- `research/2026-04-20-final-answer-selection-fix/mid-edit-verify-full/ab-qwen35-mid-edit-verify-full-20260420.db`

Related but excluded from the report because they were invalid infra-heavy reruns:

- `experiments/results/ab-qwen35-verify-only-full-20260420.db`
- `experiments/results/qwen35-control-final-clean-20260420.db`
