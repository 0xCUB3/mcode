# Roadmap: mcode Benchmark Expansion

## Overview

Add four new benchmarks (EvalPlus, LiveCodeBench, BigCodeBench, SWE-bench Live) to the existing harness, then run OC sweeps and record results. Each phase delivers one complete, runnable benchmark. Phases execute sequentially but are independent enough that each can be verified before starting the next.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: EvalPlus** - Add humaneval+ and mbpp+ loader, runner integration, tests, pyproject extras (completed 2026-02-20)
- [x] **Phase 2: LiveCodeBench** - Add LCB loader with stdin/stdout harness, cutoff filter CLI arg, tests (completed 2026-02-20)
- [x] **Phase 3: BigCodeBench** - Add BCB loader with unittest harness, Dockerfile deps, tests (completed 2026-02-20)
- [ ] **Phase 4: SWE-bench Live** - Wire swebench-live into existing SWE-bench loader and CLI
- [ ] **Phase 5: OC Sweeps + Research** - Run EvalPlus/LCB/BCB sweeps, collect results, write research entry

## Phase Details

### Phase 1: EvalPlus
**Goal**: EvalPlus benchmarks (humaneval+ and mbpp+) run end-to-end through the existing harness
**Depends on**: Nothing (builds on existing harness)
**Requirements**: LOAD-01, LOAD-02, RUN-01, INF-01, INF-04, TEST-01
**Success Criteria** (what must be TRUE):
  1. `mcode bench humaneval+` and `mcode bench mbpp+` run without errors and produce results in the DB
  2. EvalPlus tasks load with the correct fields (prompt, entry_point/test_list, test) matching existing format expectations
  3. evalplus is an optional extra in pyproject.toml and imported lazily (base install unaffected)
  4. Unit tests for the EvalPlus loader pass with no network calls
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — EvalPlus loader, runner integration, CLI commands, pyproject extra, and unit tests

### Phase 2: LiveCodeBench
**Goal**: LiveCodeBench runs through a stdin/stdout harness with contamination cutoff filtering
**Depends on**: Phase 1
**Requirements**: LOAD-03, RUN-02, CLI-01, CLI-02, INF-02, TEST-02
**Success Criteria** (what must be TRUE):
  1. `mcode bench livecodebench --lcb-cutoff 2024-06-01` loads only tasks released before the cutoff and runs them
  2. LCB tasks execute via stdin/stdout harness (not function-call style)
  3. BenchConfig accepts lcb_cutoff field and it flows through to task filtering
  4. datasets is an optional extra in pyproject.toml and imported lazily
  5. Unit tests for the LiveCodeBench loader pass with no network calls
**Plans**: 1 plan

Plans:
- [ ] 02-01-PLAN.md — LCB loader, stdin/stdout harness, BenchConfig.lcb_cutoff, CLI subcommand, and unit tests

### Phase 3: BigCodeBench
**Goal**: BigCodeBench (complete and instruct variants) runs inside the Docker sandbox with required library deps
**Depends on**: Phase 2
**Requirements**: LOAD-04, LOAD-05, RUN-03, INF-03, TEST-03
**Success Criteria** (what must be TRUE):
  1. `mcode bench bigcodebench-complete` and `mcode bench bigcodebench-instruct` run without errors and produce results
  2. BCB tasks execute via a unittest execution script (not function-call or stdin/stdout style)
  3. BigCodeBench common libs (numpy, pandas, etc.) are present in the sandbox Dockerfile so tasks don't fail on import
  4. Unit tests for the BigCodeBench loader pass with no network calls
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md — BigCodeBench loader (complete + instruct), unittest harness, Dockerfile deps, CLI commands, and unit tests

### Phase 4: SWE-bench Live
**Goal**: SWE-bench Live is a runnable benchmark using the existing SWE-bench infrastructure
**Depends on**: Phase 1
**Requirements**: LOAD-06, RUN-04, CLI-03
**Success Criteria** (what must be TRUE):
  1. `mcode bench swebench-live` dispatches through the existing SWE-bench pipeline without new infrastructure
  2. swebench-live appears as a valid choice in the CLI bench subcommand alongside all other new benchmarks
  3. No new sandbox or runner code added — existing SWEbenchSandbox handles it
**Plans**: 1 plan

Plans:
- [ ] 04-01-PLAN.md — Wire swebench-live loader, runner dispatch, CLI command, and tests

### Phase 5: OC Sweeps + Research
**Goal**: EvalPlus, LiveCodeBench, and BigCodeBench sweep results are collected and documented in the research folder
**Depends on**: Phase 3, Phase 4
**Requirements**: OC-01, OC-02, OC-03, OC-04, CLI-04, RES-01
**Success Criteria** (what must be TRUE):
  1. OC sweeps for humaneval+, mbpp+, livecodebench (with LCB_CUTOFF), bigcodebench-complete, and bigcodebench-instruct complete and shards merge cleanly
  2. LCB_CUTOFF env var passes through the OC sweep command so contamination date is reproducible
  3. Research entry README exists following the existing format with sweep commands, parameters, and result tables
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. EvalPlus | 1/1 | Complete    | 2026-02-20 |
| 2. LiveCodeBench | 1/1 | Complete   | 2026-02-20 |
| 3. BigCodeBench | 1/1 | Complete   | 2026-02-20 |
| 4. SWE-bench Live | 0/1 | Not started | - |
| 5. OC Sweeps + Research | 0/TBD | Not started | - |
