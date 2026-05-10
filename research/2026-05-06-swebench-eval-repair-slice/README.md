# SWE-bench Verified eval-repair failure slice

Qwen3.6-35B-A3B on Blue Vela, SWE-bench Verified 32-task prior-failure slice.
The baseline settings match the 319/500 run, with `--eval-repair-attempts 1`
added.

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

- `unverified_diff_discarded`: 15/32. The agent often found plausible code and edited it, but did not produce counted verification before submit.
- `wrong_patch_after_verification`: 12/32. Visible checks or agent-chosen checks passed, but official SWE-bench issue tests still failed.
- `submitted`: 5/32.

By repository:

- `matplotlib`: 0/4
- `pylint`: 2/4
- `pytest`: 0/4
- `scikit-learn`: 2/4
- `sphinx`: 1/8
- `sympy`: 0/8

## Notes

The repair loop has signal. Four selected patches appeared only after feeding
deterministic official-eval failure summaries back into a second pass. The
recovered patches were small and targeted.

The weak point is pre-submit issue reproduction, not selection. Many failures
passed local verification but failed hidden issue tests, especially in pytest,
sphinx, sympy, and sklearn ranking/set-output tasks. The next useful step is to
make the agent derive a minimal reproducer from the issue text before finalizing,
without using SWE-bench test patches or oracle data.

Eval repair should stay opt-in. It helped this prior-failure slice, but it uses
official-eval feedback. Treat it as a diagnostic or repair mode unless we have a
leaderboard-compatible reproducer path.

## Follow-up iterations

A direct prompt change asking the agent to derive a minimal issue reproducer
before editing did not work. An early slice probe showed tool-schema errors,
budget exhaustion, and 0/7 completed tasks passing before cancellation.

Including full SWE-bench test status in official reports also did not help. The
full slice matched the 5/32 score, swapped one recovered pylint task, introduced
one infrastructure/context failure, and produced no net gain.

A second repair attempt was better. Running the same 32-task slice with
`--eval-repair-attempts 2` reached 7/32. New recoveries beyond the one-repair
slice were `pytest-dev__pytest-5787`, `scikit-learn__scikit-learn-25747`, and
`sympy__sympy-13974`. The run lost `pylint-dev__pylint-4661` and
`sphinx-doc__sphinx-10449` relative to the first one-repair run. Tasks fixed on
candidate 2 were `pylint-dev__pylint-4604` and `pytest-dev__pytest-5787`.
