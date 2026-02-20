# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** Accurate, reproducible benchmark results across multiple code generation benchmarks with contamination-aware filtering.
**Current focus:** Phase 4 - SWE-bench Live

## Current Position

Phase: 4 of 5 (SWE-bench Live)
Plan: 1 of 1 in current phase
Status: Phase 4 complete
Last activity: 2026-02-20 — Plan 04-01 complete (swebench-live loader, runner dispatch, CLI, tests)

Progress: [████████░░] 80%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 3 min
- Total execution time: 0.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-evalplus | 1 | 3 min | 3 min |
| 02-livecodebench | 1 | 3 min | 3 min |
| 03-bigcodebench | 1 | 4 min | 4 min |
| 04-swe-bench-live | 1 | 2 min | 2 min |

**Recent Trend:**
- Last 5 plans: 3 min (01-01), 3 min (02-01), 4 min (03-01), 2 min (04-01)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- All phases: Lazy imports for evalplus/datasets following swebench_lite.py pattern
- Phase 2: stdin/stdout harness for LCB (full-program tasks, not function-call style)
- Phase 3: BigCodeBench libs added to Dockerfile (numpy/pandas/etc. needed in sandbox)
- Phase 4: Reuse SWEbenchSandbox for swebench-live — only dataset name changes
- [Phase 04-swe-bench-live]: Parameterize load_swebench_lite with dataset_name/benchmark params (backward-compatible defaults)
- [Phase 04-swe-bench-live]: _run_swebench_live as sibling method (not shared helper) — 40-line duplication acceptable for clarity
- [Phase 01-evalplus]: Lazy import pattern for evalplus matches swebench_lite.py (try/except RuntimeError)
- [Phase 01-evalplus]: Mock uninstalled packages in tests via types.ModuleType injection into sys.modules + importlib.reload
- [Phase 02-livecodebench]: Use repr() for code embedding in LCB harness to avoid triple-quote injection
- [Phase 02-livecodebench]: load_benchmark gets **kwargs so lcb_cutoff flows through without breaking existing callers
- [Phase 03-bigcodebench]: unittest.main(argv=['']) used in BCB harness to avoid sys.argv pytest contamination
- [Phase 03-bigcodebench]: End-to-end harness tests use unittest.TestLoader directly (exec __main__ doesn't surface TestCase subclasses to unittest.main module discovery)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 04-swe-bench-live-01-PLAN.md
Resume file: None
