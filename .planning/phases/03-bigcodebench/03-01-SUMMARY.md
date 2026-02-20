---
phase: 03-bigcodebench
plan: "01"
subsystem: bench
tags: [bigcodebench, unittest-harness, cli, dockerfile, mocked-tests]
dependency_graph:
  requires: []
  provides: [bigcodebench-complete-loader, bigcodebench-instruct-loader, bigcodebench-unittest-harness, bigcodebench-cli-commands]
  affects: [src/mcode/bench/tasks.py, src/mcode/bench/runner.py, src/mcode/cli.py, Dockerfile]
tech_stack:
  added: [bigcode/bigcodebench HuggingFace dataset]
  patterns: [lazy-import-try-except, unittest-harness, sys-modules-mock-injection]
key_files:
  created:
    - src/mcode/bench/bigcodebench.py
    - tests/test_bigcodebench.py
  modified:
    - src/mcode/bench/tasks.py
    - src/mcode/bench/runner.py
    - src/mcode/cli.py
    - Dockerfile
    - tests/test_cli_help.py
decisions:
  - "unittest.main(argv=['']) used in harness to avoid picking up sys.argv when run via exec()"
  - "End-to-end harness tests use unittest.TestLoader directly rather than exec(__main__) to avoid module discovery issues in test context"
  - "Harness structure correctness verified separately from test execution correctness"
metrics:
  duration_s: 236
  completed: 2026-02-20
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 5
---

# Phase 3 Plan 1: BigCodeBench Loader and Benchmark Summary

BigCodeBench complete/instruct variants with unittest execution harness, load_benchmark dispatch, Dockerfile scientific Python deps, and mocked unit tests.

## What Was Built

- `src/mcode/bench/bigcodebench.py`: `load_bigcodebench(cache_dir, *, variant)` loader using lazy `datasets.load_dataset` import with try/except RuntimeError pattern. Supports `variant="complete"` (docstring completion) and `variant="instruct"` (natural language). Raises `ValueError` for unknown variants.
- `src/mcode/bench/tasks.py`: Added dispatch for `"bigcodebench-complete"` and `"bigcodebench-instruct"` before the final `raise ValueError`.
- `src/mcode/bench/runner.py`: Added `_combine_for_eval` branch for `task.benchmark.startswith("bigcodebench")` that builds a unittest execution script (code + TestCase class + `unittest.main(argv=[''])`). Added `_dataset_metadata` branch returning BigCodeBench metadata dict.
- `src/mcode/cli.py`: Added `bench bigcodebench-complete` and `bench bigcodebench-instruct` commands with standard `_bench_common` parameter set (no `lcb_cutoff`).
- `Dockerfile`: Added `pip install numpy pandas matplotlib seaborn scipy scikit-learn requests` for BCB sandbox task execution.
- `tests/test_bigcodebench.py`: 9 tests covering loader (both variants, invalid variant, missing datasets, limit via load_benchmark), `_combine_for_eval` structure (both variants), and harness correctness (passing/failing cases via `unittest.TestLoader`).
- `tests/test_cli_help.py`: Added help tests for both new CLI commands.

## Task Outcomes

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | BigCodeBench loader, load_benchmark dispatch, Dockerfile deps | 7149da1 | Done |
| 2 | _combine_for_eval harness, _dataset_metadata, CLI commands | 26ed7a6 | Done |
| 3 | Unit tests (mocked, no network) | 1b4e826 | Done |

## Test Results

- 42 total tests passing (31 pre-existing + 9 new BCB + 2 new CLI help)
- All tests pass with no network calls (datasets mocked via types.ModuleType injection)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] unittest.main() picked up pytest sys.argv**

- **Found during:** Task 3 (test_combine_for_eval_bigcodebench_harness_fails)
- **Issue:** `unittest.main()` uses `sys.argv` by default. When invoked inside pytest, `sys.argv` contains pytest flags (`-xvs`) causing argparse errors. Passing `__name__ == '__main__'` via exec globals didn't help either because `unittest.main(argv=[''])` still exits with code 5 (NO_TESTS_RAN) since TestCase subclasses defined inside `exec()` aren't visible to `unittest.main()`'s module-based test discovery.
- **Fix:** Changed harness to use `unittest.main(argv=[''])` for sandbox execution (avoids sys.argv). Changed end-to-end tests to use `unittest.TestLoader().loadTestsFromTestCase()` directly on the class from the exec namespace, which correctly discovers and runs tests.
- **Files modified:** `src/mcode/bench/runner.py`, `tests/test_bigcodebench.py`
- **Commit:** 1b4e826

## Self-Check: PASSED

- [x] `src/mcode/bench/bigcodebench.py` exists
- [x] `tests/test_bigcodebench.py` exists (112 lines, above 80-line minimum)
- [x] Commits 7149da1, 26ed7a6, 1b4e826 all exist in git log
- [x] All 42 tests pass
- [x] `mcode bench bigcodebench-complete --help` and `mcode bench bigcodebench-instruct --help` work
- [x] Dockerfile contains numpy/pandas/matplotlib/seaborn/scipy/scikit-learn/requests
