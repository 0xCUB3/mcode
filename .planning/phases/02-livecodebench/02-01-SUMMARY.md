---
phase: 02-livecodebench
plan: "01"
subsystem: bench
tags: [livecodebench, datasets, stdin-stdout, harness, cli, tests]
dependency_graph:
  requires: []
  provides:
    - livecodebench loader with cutoff filtering
    - stdin/stdout eval harness in _combine_for_eval
    - BenchConfig.lcb_cutoff field
    - datasets optional extra in pyproject.toml
    - bench livecodebench CLI subcommand with --lcb-cutoff
  affects:
    - src/mcode/bench/tasks.py (load_benchmark dispatch, **kwargs)
    - src/mcode/bench/runner.py (BenchConfig, _combine_for_eval, _dataset_metadata, run_benchmark)
    - src/mcode/cli.py (_bench_common lcb_cutoff param, new command)
tech_stack:
  added:
    - datasets>=2.14.0 (optional extra)
  patterns:
    - Lazy import with try/except RuntimeError (matching swebench_lite.py pattern)
    - repr() embedding to avoid triple-quote injection in harness strings
    - types.ModuleType injection for mocking uninstalled packages in tests
key_files:
  created:
    - src/mcode/bench/livecodebench.py
    - tests/test_livecodebench.py
  modified:
    - src/mcode/bench/tasks.py
    - src/mcode/bench/runner.py
    - src/mcode/cli.py
    - pyproject.toml
    - tests/test_cli_help.py
decisions:
  - repr() for code/test embedding in harness (avoids triple-quote injection, no extra escaping needed)
  - load_benchmark gets **kwargs so cutoff flows through without changing signature for other benchmarks
  - _bench_common gets lcb_cutoff param with default None (all existing callers unaffected)
metrics:
  duration: "3 min"
  completed: "2026-02-20"
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 5
---

# Phase 2 Plan 1: LiveCodeBench Loader and Harness Summary

**One-liner:** LiveCodeBench loader from HuggingFace datasets with date-based cutoff filtering, repr()-safe stdin/stdout exec harness, and `mcode bench livecodebench --lcb-cutoff` CLI command.

## What Was Built

### Task 1: LiveCodeBench loader, pyproject extra, load_benchmark dispatch

- Created `src/mcode/bench/livecodebench.py` with `load_livecodebench(cache_dir, *, cutoff, limit)` following the evalplus.py lazy-import pattern.
- Loader calls `load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", split="test")`.
- Cutoff filtering uses string comparison on `release_date` (YYYY-MM-DD format).
- Each row becomes a `Task` with `entry_point=None` and `test_code` set to the `input_output` JSON string.
- Added `livecodebench` branch to `load_benchmark` in `tasks.py`, with `**kwargs` added to the signature so `cutoff` flows through without changing the interface for existing callers.
- Added `datasets = ["datasets>=2.14.0"]` optional extra to `pyproject.toml`.

### Task 2: stdin/stdout harness, BenchConfig.lcb_cutoff, metadata, CLI command

- Added `lcb_cutoff: str | None = None` field to `BenchConfig`.
- Updated `run_benchmark` to pass `cutoff=self.config.lcb_cutoff` when calling `load_benchmark`.
- Added `livecodebench` branch to `_combine_for_eval` that builds a harness using `repr()` to safely embed code and test JSON (avoids triple-quote injection issues). The harness uses `exec(compile(...))` with redirected `sys.stdin`/`sys.stdout` via `io.StringIO`.
- Added `livecodebench` branch to `_dataset_metadata` returning source/version info.
- Added `_bench_common` `lcb_cutoff` param (default None, existing callers unaffected).
- Added `@bench_app.command("livecodebench")` with `--lcb-cutoff` option.

### Task 3: Unit tests (mocked, no network)

- Created `tests/test_livecodebench.py` with 10 tests covering:
  - Task production from fixture (benchmark, task_id, prompt, entry_point, test_code, metadata)
  - Cutoff filtering (before/after/none)
  - Limit application
  - Starter code inclusion in prompt
  - RuntimeError on missing datasets package
  - `_combine_for_eval` structure check
  - End-to-end harness exec (passing case)
  - End-to-end harness exec (failing case raises SystemExit)
- Added `test_cli_bench_livecodebench_help` to `tests/test_cli_help.py`.
- All 31 tests pass (16 new + 15 existing).

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use `repr()` for code embedding in harness | Avoids triple-quote injection, no manual escaping |
| `**kwargs` on `load_benchmark` | Cutoff passes through without breaking existing callers |
| `_bench_common` gets `lcb_cutoff` param | Cleaner than constructing BenchConfig separately in the command |
| No `limit` param to `load_livecodebench` | Consistent with evalplus; `_limit()` in `load_benchmark` handles it |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- `uv run ruff check src tests`: All checks passed
- `uv run pytest`: 31 passed
- `BenchConfig(model_id='x', lcb_cutoff='2024-06-01').lcb_cutoff` returns `"2024-06-01"`
- `from mcode.bench.tasks import load_benchmark` imports without error
- `uv run mcode bench livecodebench --help` shows `--lcb-cutoff` option

## Self-Check: PASSED
