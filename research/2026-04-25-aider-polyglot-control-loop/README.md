# Aider Polyglot upstream control-loop run

This run tested the upstream Mellea cutover and the generic control-loop changes
on Aider Polyglot with Qwen3.6-35B-A3B. The changes kept here are harness
changes, not Polyglot prompt patches: remote Blue Vela Aider runs, the Polyglot
toolchain on Blue Vela, mixed valid/malformed tool-call handling, mandatory
verification before `final_answer`, reminders to verify after edits, and
suppression of repeated identical failed `run_tests` calls until the code
changes.

## Setup

- Model: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM endpoint on Blue Vela, `http://p1-r12-n4.bluevela.rmf.ibm.com:8321/v1`
- Benchmark root on Blue Vela: `/u/skula/mcode-launch/benchmarks/polyglot-benchmark`
- Toolchain root on Blue Vela: `/proj/dmfexp/skula/mcode-shared/toolchains/aider-polyglot`
- Baseline to beat: little-coder Qwen3.6 result, `177/225 = 78.67%`
- Previous mcode baseline: 170/225 = 75.6%

## Commands

The first 12-task Blue Vela slice used the representative failure set from the
previous full run:

```bash
MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 \
uv run mcode bench aider-polyglot \
  --on bluevela \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --loop-budget 20 \
  --task-ids cpp/binary-search-tree,cpp/meetup,go/poker,java/connect,java/react,java/rest-api,javascript/ledger,javascript/react,python/connect,python/sgf-parsing,rust/doubly-linked-list,rust/poker \
  --shards 4 \
  --db research/2026-04-25-aider-polyglot-control-loop/progress-slice-12-toolchain.db
```

The promotion set combined 55 previous `unverified_diff_discarded` or
`budget_exhausted` failures with 18 sentinel tasks that had passed before. The
first 73-task run used `--shards 4`; one shard was killed after partial progress,
and the missing 13 tasks were rerun with `--shards 4`. I rebuilt the DB with:

```bash
uv run mcode merge-shards --force \
  --out research/2026-04-25-aider-polyglot-control-loop/promotion-73-recovered.db \
  research/2026-04-25-aider-polyglot-control-loop/promotion-partial-shards/results-shard-0.db \
  research/2026-04-25-aider-polyglot-control-loop/promotion-partial-shards/results-shard-1.db \
  research/2026-04-25-aider-polyglot-control-loop/promotion-partial-shards/results-shard-2.db \
  research/2026-04-25-aider-polyglot-control-loop/promotion-partial-shards/results-shard-3.db \
  research/2026-04-25-aider-polyglot-control-loop/promotion-missing-13.db
```

For the full run, I tried to push parallelism without raising `n_samples`. A
single 8-shard promotion run failed, and a single 4-shard promotion run lost one
shard. The full run therefore used seven language chunks at the same time, with
`--shards 2` inside each chunk. That gave up to 14 active Aider workers while
keeping each remote job small enough to recover. The exact chunk commands and
the Rust recovery commands are in `full-parallel-shards2-lock/commands.sh`.

## Results

Representative 12-task slice:

| Slice | Passed | Total | Notes |
|-|-:|-:|-|
| Initial Blue Vela run before toolchain fix | 0 | 12 | Failed on missing Go, CMake, npm, and Rust toolchains |
| Toolchain-fixed run | 2 | 12 | Baseline for these same 12 tasks was 0/12 |

Promotion slice:

| Metric | Value |
|-|-:|
| Rows | 73 |
| Passed | 41 |
| Previous failure tasks in slice | 55 |
| Previous failures solved | 26 |
| Sentinel regressions | 3 |

Full run:

| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 29 | 34 | 85.3% |
| Go | 30 | 39 | 76.9% |
| Rust | 24 | 30 | 80.0% |
| JavaScript | 46 | 49 | 93.9% |
| C++ | 21 | 26 | 80.8% |
| Java | 40 | 47 | 85.1% |
| Total | 190 | 225 | 84.4% |

Terminal reasons for the final merged DB:

| Reason | Count | Passed |
|-|-:|-:|
| submitted | 190 | 190 |
| unverified_diff_discarded | 32 | 0 |
| budget_exhausted | 3 | 0 |

Comparison with the previous best mcode full run:

| Metric | Previous best | This run | Delta |
|-|-:|-:|-:|
| Passed | 170 | 190 | +20 |
| Pass rate | 75.6% | 84.4% | +8.9 points |
| New solves vs previous best | 0 | 27 | +27 |
| Regressions vs previous best | 0 | 7 | +7 |
| Gap to little-coder 177/225 | -7 | +13 | +20 |

## Files

- Final merged DB: `full-parallel-shards2-lock/results.db`
- Exact full-run commands: `full-parallel-shards2-lock/commands.sh`
- Full-run chunk DBs and logs: `full-parallel-shards2-lock/*.db`, `full-parallel-shards2-lock/*.log`
- Promotion DB: `promotion-73-recovered.db`
- Initial slice DBs: `progress-slice-12.db`, `progress-slice-12-toolchain.db`

## Notes from the run

The Blue Vela path mattered. The first slice looked like a model failure until
the logs showed missing language toolchains. After wiring the Polyglot toolchain
into remote Aider runs, the same slice moved from 0/12 to 2/12, which matched
the later promotion signal.

The control-loop changes were the main result. The promotion slice solved 26 of
the previous unverified or budget failures, then the full run landed at
190/225. That is 13 tasks above the little-coder reference score and 20 tasks
above the previous best mcode run.

Parallelism had a real ceiling. Eight shards failed, and the 73-task four-shard
promotion run lost a shard with exit `-7`. Running seven language chunks with
two internal shards each was the highest useful setting I saw. The shared
benchmark checkout also needed a lock; without it, concurrent remote jobs fought
over `.git/shallow.lock` before any benchmark work started.
