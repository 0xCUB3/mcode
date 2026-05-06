# SWE-bench Verified eval-repair failure slice

Qwen3.6-35B-A3B on Blue Vela, SWE-bench Verified 32-task prior-failure slice, same baseline harness settings as the 319/500 run plus `--eval-repair-attempts 1`.

## Result

- Score: 5/32 (15.6%) on tasks that all failed in the accepted 319/500 run.
- Repair recovered 4 tasks where candidate 0 failed official eval and candidate 1 passed:
  - `scikit-learn__scikit-learn-10844`
  - `scikit-learn__scikit-learn-12973`
  - `pylint-dev__pylint-4661`
  - `pylint-dev__pylint-4970`
- One task passed without repair:
  - `sphinx-doc__sphinx-10449`

## Error modes

- `unverified_diff_discarded`: 15/32. The agent often found and edited plausible code, but could not produce counted verification before submit.
- `wrong_patch_after_verification`: 12/32. Visible checks or agent-chosen checks passed, but official SWE-bench issue tests still failed.
- `submitted`: 5/32.

By repository:

- `matplotlib`: 0/4
- `pylint`: 2/4
- `pytest`: 0/4
- `scikit-learn`: 2/4
- `sphinx`: 1/8
- `sympy`: 0/8

## Findings

The repair loop has real signal. Four selected patches were only found after feeding deterministic official-eval failure summaries back into a second pass. The recovered patches were small and targeted.

The main weakness is not selection. It is pre-submit issue reproduction. Many failures passed local verification but failed hidden issue tests, especially pytest, sphinx, sympy, and sklearn ranking/set-output tasks. The next iteration should make the agent derive a minimal reproducer from the issue text before finalizing, without using SWE-bench test patches or oracle data.

Eval repair should remain opt-in for now. It improves this prior-failure slice, but it uses official-eval feedback and is better treated as a diagnostic or repair mode until a leaderboard-compatible reproducer path gets tested.
