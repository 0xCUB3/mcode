# Aider Polyglot workspace context full run

This run tested the generic workspace context collector on the full Aider Polyglot benchmark with Qwen3.6-35B-A3B. The collector is not Polyglot-specific. It scans local authoritative Markdown docs such as `AGENTS.md`, `README.md`, `SPEC.md`, and `.docs/instructions.md`, then injects a small cited context block into the agent goal.

## Setup

- Model: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM endpoint on Blue Vela through local tunnel `http://127.0.0.1:18325/v1`
- Benchmark root: `/Users/skula/Documents/polyglot-benchmark`
- mcode commits under test: `e7fbecf` (`add workspace context discovery`) for run 1, `00bdec3` (`clarify run tests default argument`) for run 2, `10452ff` for the `go/poker` infra rerun, `e18df56` plus `--loop-budget 20` for run 3, and `528fce7` (`append test failure report snippets`) for run 5
- Baseline to beat: little-coder Qwen3.6 result, `177/225 = 78.67%`

## Commands

Run 1:
```bash
OPENAI_BASE_URL=http://127.0.0.1:18325/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=1800 \
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --benchmark-root /Users/skula/Documents/polyglot-benchmark \
  --db research/2026-04-24-aider-polyglot-workspace-context-full/run1-workspace-context/results.db
```

Run 2:
```bash
OPENAI_BASE_URL=http://127.0.0.1:18325/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=1800 \
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --benchmark-root /Users/skula/Documents/polyglot-benchmark \
  --db research/2026-04-24-aider-polyglot-workspace-context-full/run2-verification-prompt/results.db
```

`go/poker` infra rerun after the non-UTF8 output fix:
```bash
OPENAI_BASE_URL=http://127.0.0.1:18325/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=1800 \
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --task-ids go/poker \
  --benchmark-root /Users/skula/Documents/polyglot-benchmark \
  --db research/2026-04-24-aider-polyglot-workspace-context-full/run2-verification-prompt/go-poker-rerun.db
```

Run 3:
```bash
OPENAI_BASE_URL=http://127.0.0.1:18325/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=2400 \
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --loop-budget 20 \
  --benchmark-root /Users/skula/Documents/polyglot-benchmark \
  --db research/2026-04-24-aider-polyglot-workspace-context-full/run3-loop20/results.db
```

Run 5:
```bash
# The run was chunked by language to avoid losing progress on long benchmark invocations.
# The exact commands are in run5-failure-reports/commands.sh.
bash research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/commands.sh
```

## Results

Run 1:
| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 25 | 34 | 73.5% |
| Go | 25 | 39 | 64.1% |
| Rust | 17 | 30 | 56.7% |
| JavaScript | 37 | 49 | 75.5% |
| C++ | 18 | 26 | 69.2% |
| Java | 27 | 47 | 57.4% |
| Total | 149 | 225 | 66.2% |

Run 2:
| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 25 | 34 | 73.5% |
| Go | 26 | 39 | 66.7% |
| Rust | 19 | 30 | 63.3% |
| JavaScript | 42 | 49 | 85.7% |
| C++ | 18 | 26 | 69.2% |
| Java | 29 | 47 | 61.7% |
| Total | 159 | 225 | 70.7% |

Run 3:
| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 26 | 34 | 76.5% |
| Go | 29 | 39 | 74.4% |
| Rust | 20 | 30 | 66.7% |
| JavaScript | 43 | 49 | 87.8% |
| C++ | 20 | 26 | 76.9% |
| Java | 30 | 47 | 63.8% |
| Total | 168 | 225 | 74.7% |

Run 5:
| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 27 | 34 | 79.4% |
| Go | 31 | 39 | 79.5% |
| Rust | 20 | 30 | 66.7% |
| JavaScript | 43 | 49 | 87.8% |
| C++ | 19 | 26 | 73.1% |
| Java | 30 | 47 | 63.8% |
| Total | 170 | 225 | 75.6% |

Terminal reasons:

Run 1:
| Reason | Count | Passed |
|-|-:|-:|
| submitted | 149 | 149 |
| unverified_diff_discarded | 62 | 0 |
| budget_exhausted | 14 | 0 |

Run 2:
| Reason | Count | Passed |
|-|-:|-:|
| submitted | 159 | 159 |
| unverified_diff_discarded | 48 | 0 |
| budget_exhausted | 17 | 0 |
| infra_failure | 1 | 0 |

Run 3:
| Reason | Count | Passed |
|-|-:|-:|
| submitted | 168 | 168 |
| unverified_diff_discarded | 53 | 0 |
| budget_exhausted | 4 | 0 |

Run 5:
| Reason | Count | Passed |
|-|-:|-:|
| submitted | 170 | 170 |
| unverified_diff_discarded | 52 | 0 |
| budget_exhausted | 3 | 0 |

## Files

- Run 1 DB/log: `run1-workspace-context/results.db`, `run1-workspace-context/benchmark.log`
- Run 2 DB/log: `run2-verification-prompt/results.db`, `run2-verification-prompt/benchmark.log`
- `go/poker` rerun DB: `run2-verification-prompt/go-poker-rerun.db`
- Run 3 DB/log: `run3-loop20/results.db`, `run3-loop20/benchmark.log`
- Run 5 combined DB/logs: `run5-failure-reports/results.db`, chunk DB/log files, and `run5-failure-reports/commands.sh`
- HTML report: `aider-polyglot-workspace-context-report.html`

## Findings

The generic workspace context collector helped targeted A/B slices, but the first full run only reached `149/225`, far below little-coder's `177/225`. Compared with the prior corrected estimate of `148/225`, that was basically flat.

The second run changed only the generic verifier wording so the model is told to call `run_tests` with `test_cmd="default"`, rather than treating `run_tests default` as a literal argument. That moved the full run to `159/225`. The biggest gains were JavaScript (`37/49` to `42/49`), Rust (`17/30` to `19/30`), Go (`25/39` to `26/39`), and Java (`27/47` to `29/47`). It still missed little-coder by 18 tasks.

Run 2 had one infrastructure failure, `go/poker`, caused by non-UTF8 command output. Commit `10452ff` fixed command decoding with replacement characters and the single-task rerun produced a normal `unverified_diff_discarded` failure, not an infra failure. The corrected score stays `159/225`.

After run 2, the main remaining failure bucket was still unverified diffs. That pointed the next iteration toward generic control-loop behavior and better failed-test feedback, not more benchmark-local prompt content.

Run 3 raised the loop budget from the default to `20` and extended the per-task timeout to `2400`. That was a generic control-loop change, not benchmark prompt content, and it moved the full run from `159/225` to `168/225`. It also cut budget exhaustion from 17 tasks to 4 tasks, which confirms that some earlier failures were agents running out of turns while still working.

The gap to little-coder is now 10 tasks. The remaining misses are mostly `unverified_diff_discarded`, so the next iteration should not be another raw budget increase. The useful target is better failure information after a verification failure, especially for projects whose test runners hide the actual assertion in report files.


Run 4 tested cached tool-validator normalization and landed at `168/225`, the same as run 3. That change was reverted in `e390b14` because it did not improve the full metric.

Run 5 appends short test failure report snippets to failed `run_tests` output when common report artifacts are present. This is generic verifier feedback, not Polyglot-specific prompt content. It moved the full combined run to `170/225`, with gains in Python (`26/34` to `27/34`) and Go (`29/39` to `31/39`), while C++ dropped one task. The net gain is only two tasks, so it is useful but not enough. The next iteration needs to target the remaining `unverified_diff_discarded` cases without adding benchmark-local knowledge.