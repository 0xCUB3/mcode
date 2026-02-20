---
phase: 05-oc-sweeps-research
plan: 01
subsystem: deploy/research
tags: [oc-sweep, livecodebench, evalplus, bigcodebench, research]
dependency_graph:
  requires: []
  provides: [lcb-cutoff-passthrough, benchmark-expansion-research-entry]
  affects: [deploy/k8s/oc_bench_sweep.py, research/README.md]
tech_stack:
  added: []
  patterns: [bash-template-conditional, research-entry-format]
key_files:
  created:
    - research/2026-02-20-benchmark-expansion-granite4/README.md
  modified:
    - deploy/k8s/oc_bench_sweep.py
    - research/README.md
decisions:
  - LCB_CUTOFF passthrough follows LIMIT/S2_MODEL conditional pattern (bash template, not Python-level)
  - Research entry uses merge-shards + mcode results/report workflow consistent with existing entries
metrics:
  duration: 2 min
  completed: 2026-02-20
---

# Phase 5 Plan 1: OC Sweeps + Research Entry Summary

LCB_CUTOFF env var wired through OC sweep bash template to `--lcb-cutoff` CLI arg; research entry created with exact copy-pasteable sweep commands for humaneval+, mbpp+, livecodebench, bigcodebench-complete, bigcodebench-instruct.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Wire LCB_CUTOFF passthrough in OC sweep bash template | e92095e | deploy/k8s/oc_bench_sweep.py |
| 2 | Create research entry with sweep commands for all new benchmarks | c3f7b2b | research/2026-02-20-benchmark-expansion-granite4/README.md, research/README.md |

## What Was Built

### Task 1: LCB_CUTOFF passthrough

Added a conditional block in the `_render_job` bash template that reads `LCB_CUTOFF` from the pod's env (via ConfigMap) and passes it as `--lcb-cutoff` to `mcode bench`:

```bash
lcb_cutoff_args=""
if [ -n "${LCB_CUTOFF:-}" ]; then
  lcb_cutoff_args="--lcb-cutoff ${LCB_CUTOFF}"
fi
```

The variable is appended to the mcode command line after `${limit_args}`. Existing sweeps without `LCB_CUTOFF` in their env are unaffected (empty string expands to nothing).

### Task 2: Research entry

Created `research/2026-02-20-benchmark-expansion-granite4/README.md` (94 lines) with:
- Goal section describing the benchmark expansion scope
- Three separate sweep commands: EvalPlus (humaneval+, mbpp+), LiveCodeBench (with LCB_CUTOFF=2024-06-01), BigCodeBench (complete + instruct)
- merge-shards and report commands for post-sweep analysis
- Placeholder result table with all 30 benchmark/config rows
- Placeholder findings section

Updated `research/README.md` index with the new entry.

## Verification

- `uv run ruff check deploy/k8s/oc_bench_sweep.py`: PASS
- `uv run ruff format --check deploy/k8s/oc_bench_sweep.py`: PASS (auto-formatted during task)
- `grep LCB_CUTOFF deploy/k8s/oc_bench_sweep.py`: PASS
- `grep lcb-cutoff deploy/k8s/oc_bench_sweep.py`: PASS
- Research README exists with all 5 benchmark names: PASS
- `research/README.md` index updated: PASS
- `uv run pytest tests/ -x -q`: 43 passed

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED
