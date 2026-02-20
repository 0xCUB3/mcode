---
phase: 04-swe-bench-live
plan: 01
subsystem: bench
tags: [swebench, swe-bench-verified, cli, benchmarking, huggingface]

# Dependency graph
requires:
  - phase: 03-bigcodebench
    provides: BenchmarkRunner, BenchConfig, CLI bench subcommand pattern
provides:
  - swebench-live CLI subcommand dispatching through SWEbenchSandbox
  - load_swebench_lite with dataset_name and benchmark params for reuse
  - swebench-live runner dispatch in BenchmarkRunner.run_benchmark
affects: [future swebench phases, results/reporting phases]

# Tech tracking
tech-stack:
  added: []
  patterns: [parameterize dataset_name in loader to reuse pipeline for related benchmarks]

key-files:
  created: []
  modified:
    - src/mcode/bench/swebench_lite.py
    - src/mcode/bench/runner.py
    - src/mcode/cli.py
    - tests/test_cli_help.py

key-decisions:
  - "Reuse SWEbenchSandbox for swebench-live: only dataset_name changes, no new pipeline"
  - "Add dataset_name and benchmark params to load_swebench_lite with backward-compatible defaults"
  - "Extract _run_swebench_live as sibling method to _run_swebench_lite rather than shared helper (40 lines, clear separation)"

patterns-established:
  - "Parameterize loader functions to handle dataset variants rather than duplicating loaders"

requirements-completed: [LOAD-06, RUN-04, CLI-03]

# Metrics
duration: 2min
completed: 2026-02-20
---

# Phase 04 Plan 01: SWE-bench Live Summary

**swebench-live wired through SWEbenchSandbox using SWE-bench_Verified dataset, with load_swebench_lite parameterized and a full CLI subcommand matching swebench-lite's interface**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-20T00:30:23Z
- **Completed:** 2026-02-20T00:32:15Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `load_swebench_lite()` now accepts `dataset_name` and `benchmark` parameters with backward-compatible defaults, enabling reuse for swebench-live
- `BenchmarkRunner.run_benchmark("swebench-live")` dispatches to `_run_swebench_live()`, which loads from `SWE-bench/SWE-bench_Verified` and runs through the same SWEbenchSandbox pipeline
- `mcode bench swebench-live` CLI subcommand added with identical options to swebench-lite (--split, --arch, --namespace, --max-workers, --force-rebuild, --mem-limit, --pids-limit, plus all common options)
- CLI help test added; all 43 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire swebench-live loader and runner dispatch** - `9e192de` (feat)
2. **Task 2: Add CLI swebench-live command and tests** - `5fe0d53` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/mcode/bench/swebench_lite.py` - Added `dataset_name` and `benchmark` params to `load_swebench_lite()`
- `src/mcode/bench/runner.py` - Added `swebench-live` branch in `run_benchmark()` and new `_run_swebench_live()` method
- `src/mcode/cli.py` - Added `bench_swebench_live` command mirroring `bench_swebench_lite` signature
- `tests/test_cli_help.py` - Added `test_cli_bench_swebench_live_help()`

## Decisions Made
- Reuse SWEbenchSandbox for swebench-live: the only difference is the dataset name, so no new sandbox or runner infrastructure was needed
- Add `dataset_name` and `benchmark` params to `load_swebench_lite` with defaults that preserve all existing behavior
- Implemented `_run_swebench_live` as a sibling method to `_run_swebench_lite` rather than extracting a shared helper. The 40-line duplication is small and each method is clearer in isolation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- swebench-live is fully wired; running `mcode bench swebench-live --model <model>` will load from SWE-bench_Verified and run through the existing Docker/SWEbenchSandbox pipeline
- No blockers

## Self-Check: PASSED

- src/mcode/bench/swebench_lite.py: FOUND
- src/mcode/bench/runner.py: FOUND
- src/mcode/cli.py: FOUND
- tests/test_cli_help.py: FOUND
- .planning/phases/04-swe-bench-live/04-01-SUMMARY.md: FOUND
- commit 9e192de: FOUND
- commit 5fe0d53: FOUND

---
*Phase: 04-swe-bench-live*
*Completed: 2026-02-20*
