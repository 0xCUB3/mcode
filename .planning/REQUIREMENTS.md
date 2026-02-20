# Requirements: mcode Benchmark Expansion

**Defined:** 2026-02-20
**Core Value:** Accurate, reproducible benchmark results across multiple code generation benchmarks with contamination-aware filtering.

## v1 Requirements

### Benchmark Loaders

- [x] **LOAD-01**: EvalPlus loader produces humaneval+ tasks with prompt, entry_point, test
- [x] **LOAD-02**: EvalPlus loader produces mbpp+ tasks with prompt, test_list, test_setup_code
- [x] **LOAD-03**: LiveCodeBench loader produces tasks from HuggingFace with question_content + starter_code
- [ ] **LOAD-04**: BigCodeBench loader produces bigcodebench-complete tasks (complete_prompt)
- [ ] **LOAD-05**: BigCodeBench loader produces bigcodebench-instruct tasks (instruct_prompt)
- [ ] **LOAD-06**: SWE-bench Live loader reuses existing SWEbenchSandbox infrastructure

### Runner Integration

- [x] **RUN-01**: _combine_for_eval() handles humaneval+ and mbpp+ same as vanilla counterparts
- [x] **RUN-02**: _combine_for_eval() builds stdin/stdout test harness for livecodebench
- [ ] **RUN-03**: _combine_for_eval() builds unittest execution script for bigcodebench
- [ ] **RUN-04**: Runner dispatches swebench-live to existing SWE-bench pipeline

### CLI & Config

- [x] **CLI-01**: --lcb-cutoff CLI arg filters LiveCodeBench by release_date
- [x] **CLI-02**: BenchConfig gains lcb_cutoff field
- [ ] **CLI-03**: All new benchmarks wired into CLI bench subcommand
- [ ] **CLI-04**: OC sweep supports LCB_CUTOFF env var passthrough

### Infrastructure

- [x] **INF-01**: evalplus optional extra in pyproject.toml
- [x] **INF-02**: datasets optional extra in pyproject.toml
- [ ] **INF-03**: BigCodeBench common libs added to Dockerfile
- [x] **INF-04**: Lazy imports for evalplus and datasets (swebench_lite.py pattern)

### Testing

- [x] **TEST-01**: Unit tests for EvalPlus loader (mocked)
- [x] **TEST-02**: Unit tests for LiveCodeBench loader (mocked)
- [ ] **TEST-03**: Unit tests for BigCodeBench loader (mocked)

### OC Sweeps

- [ ] **OC-01**: OC sweep for humaneval+ and mbpp+ with appropriate shard/parallelism params
- [ ] **OC-02**: OC sweep for livecodebench with LCB_CUTOFF=2024-06-01
- [ ] **OC-03**: OC sweep for bigcodebench-complete and bigcodebench-instruct
- [ ] **OC-04**: Results collected and merged per existing research/ patterns

### Research

- [ ] **RES-01**: Research entry with README following existing format

## v2 Requirements

### Additional Benchmarks

- **BENCH-01**: MultiPL-E for multi-language code generation
- **BENCH-02**: APPS for competitive programming tasks

## Out of Scope

| Feature | Reason |
|---------|--------|
| Modifying existing benchmark loaders | Working as-is, no changes needed |
| New LLM backends or strategies | Not part of this milestone |
| Results DB schema changes | Current schema handles new benchmarks |
| SWE-bench Live OC sweeps | Local verification only for now |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| LOAD-01 | Phase 1 | Complete |
| LOAD-02 | Phase 1 | Complete |
| LOAD-03 | Phase 2 | Complete |
| LOAD-04 | Phase 3 | Pending |
| LOAD-05 | Phase 3 | Pending |
| LOAD-06 | Phase 4 | Pending |
| RUN-01 | Phase 1 | Complete |
| RUN-02 | Phase 2 | Complete |
| RUN-03 | Phase 3 | Pending |
| RUN-04 | Phase 4 | Pending |
| CLI-01 | Phase 2 | Complete |
| CLI-02 | Phase 2 | Complete |
| CLI-03 | Phase 4 | Pending |
| CLI-04 | Phase 5 | Pending |
| INF-01 | Phase 1 | Complete |
| INF-02 | Phase 2 | Complete |
| INF-03 | Phase 3 | Pending |
| INF-04 | Phase 1 | Complete |
| TEST-01 | Phase 1 | Complete |
| TEST-02 | Phase 2 | Complete |
| TEST-03 | Phase 3 | Pending |
| OC-01 | Phase 5 | Pending |
| OC-02 | Phase 5 | Pending |
| OC-03 | Phase 5 | Pending |
| OC-04 | Phase 5 | Pending |
| RES-01 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0

---
*Requirements defined: 2026-02-20*
*Last updated: 2026-02-20 after roadmap creation*
