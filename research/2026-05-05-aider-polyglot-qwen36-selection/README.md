# Aider Polyglot Qwen3.6 selection run

This run tested the Aider Polyglot harness with Qwen3.6-35B-A3B after adding
deterministic failure feedback and reactive-stream retry guidance. The headline
number is `207/225 = 92.0%` on the full benchmark.

This should not be compared directly to the earlier `190/225` single-selection
run. This run used higher turn budgets and `--selection-attempts 3`, so the
honest label is selected-trajectory result, not default single-pass score.

## Setup

- Date: 2026-05-05
- Model: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM on Blue Vela
- Benchmark: Aider Polyglot, 225 tasks
- First-attempt budget: 20 turns
- Retry budget: 12 turns
- Selection attempts: 3
- Shards: 8
- Latest code commit for the run: `4082f3d7a9cda96fa773e760360a1d6508d44c19`

## Commands

Regression smoke before the full run:

```bash
MCODE_LIVE_TRACE=1 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=420 \
uv run mcode bench aider-polyglot \
  --benchmark-root benchmarks/polyglot-benchmark \
  --task-ids python/affine-cipher,python/connect,python/phone-number,python/sgf-parsing,python/zebra-puzzle,go/connect,go/poker,go/pov,go/zebra-puzzle,rust/decimal,rust/dot-dsl,rust/forth,rust/poker,rust/scale-generator,javascript/bowling,javascript/ledger,javascript/promises,javascript/react,cpp/all-your-base,cpp/binary-search-tree,cpp/dnd-character,cpp/meetup,cpp/yacht,java/connect,java/hangman,java/sgf-parsing,java/zebra-puzzle \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --db experiments/results/bv-aider-regression-27-reactive-full-smoke.db \
  --loop-budget 20 \
  --retry-loop-budget 12 \
  --selection-attempts 3 \
  --diagnostic-traces \
  --fetch-db \
  --fetch-artifacts
```

Full benchmark:

```bash
MCODE_LIVE_TRACE=1 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=420 \
uv run mcode bench aider-polyglot \
  --benchmark-root benchmarks/polyglot-benchmark \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --db experiments/results/bv-aider-full-reactive-run.db \
  --loop-budget 20 \
  --retry-loop-budget 12 \
  --selection-attempts 3 \
  --shards 8 \
  --fetch-db \
  --no-fetch-artifacts
```

## Results

Regression smoke:

| Slice | Passed | Total | Rate |
|-|-:|-:|-:|
| 27-task regression smoke | 20 | 27 | 74.1% |

Full run by language:

| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 32 | 34 | 94.1% |
| Go | 37 | 39 | 94.9% |
| Rust | 26 | 30 | 86.7% |
| JavaScript | 47 | 49 | 95.9% |
| C++ | 24 | 26 | 92.3% |
| Java | 41 | 47 | 87.2% |
| Total | 207 | 225 | 92.0% |

Terminal reasons:

| Reason | Count | Passed |
|-|-:|-:|
| submitted | 207 | 207 |
| unverified_diff_discarded | 15 | 0 |
| budget_exhausted | 3 | 0 |

Comparison:

| Metric | Previous kept Aider result | This run | Delta |
|-|-:|-:|-:|
| Passed | 190 | 207 | +17 |
| Pass rate | 84.4% | 92.0% | +7.6 points |
| Failed | 35 | 18 | -17 |

## Remaining failures

| Task | Terminal reason | Notes |
|-|-|-|
| `cpp/gigasecond` | `budget_exhausted` | Edited and verified, but ran out of turns before producing a verified final patch. |
| `cpp/sublist` | `budget_exhausted` | Edited late and never reached verification in the selected candidate. |
| `go/pov` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `go/robot-simulator` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `java/bowling` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `java/connect` | `unverified_diff_discarded` | Known near-miss from the smoke run; usually one Hex path or edge case remains. |
| `java/forth` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `java/hangman` | `unverified_diff_discarded` | Nondeterministic. It can pass in a single-task probe, but this full run still produced an unverified reactive-state implementation. |
| `java/pov` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `java/rational-numbers` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `javascript/bowling` | `unverified_diff_discarded` | Candidate edited and ran tests across three attempts, but no verified patch survived. |
| `javascript/complex-numbers` | `unverified_diff_discarded` | Candidate edited and ran tests across three attempts, but no verified patch survived. |
| `python/bowling` | `unverified_diff_discarded` | Candidate edited and ran tests, but no verified patch survived. |
| `python/connect` | `unverified_diff_discarded` | Nondeterministic. Passed in targeted probes, failed here. |
| `rust/fizzy` | `budget_exhausted` | Edited and verified late, then ran out of turns. |
| `rust/forth` | `unverified_diff_discarded` | Nondeterministic. Passed in targeted probes, failed here. |
| `rust/react` | `unverified_diff_discarded` | Candidate edited and ran tests across three attempts, but no verified patch survived. |
| `rust/xorcism` | `unverified_diff_discarded` | High-token failure. Edited and ran tests across three attempts but did not converge. |

The full run did not fetch artifacts, so the table above comes from DB terminal
metrics plus the live run log. The failure list is exported in `failures.csv`.

## Overfit check

There is no canned exercise solution in this result. The code changes were at
the harness level: source snippets for failed tests, richer JUnit details, stale
timeout-report suppression, syntax and delimiter guards, final close-failure
repair attempts, and generic reactive-stream guidance when a repository imports
Observable APIs.

The caveat is the evaluation setting. `--selection-attempts 3` is a real compute
multiplier. The reactive-stream note was motivated by `java/hangman`; it is
technology-specific rather than exercise-specific, but it is still a narrow
harness prior and should be reported that way.

The 225-task run is useful evidence that the changes did not just memorize the
27-task smoke set. The right citation is: Qwen3.6-35B-A3B on Aider Polyglot,
Blue Vela harness, 20+12 turn budgets, 3 selected attempts, 207/225.

## Files

- Full merged DB: `results.db`
- 27-task smoke DB: `regression-27-smoke.db`
- Failure export: `failures.csv`
