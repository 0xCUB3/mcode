# Aider Polyglot workspace context full run

This run tested the generic workspace context collector on the full Aider Polyglot benchmark with Qwen3.6-35B-A3B. The collector is not Polyglot-specific. It scans local authoritative Markdown docs such as `AGENTS.md`, `README.md`, `SPEC.md`, and `.docs/instructions.md`, then injects a small cited context block into the agent goal.

## Setup

- Model: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM endpoint on Blue Vela through local tunnel `http://127.0.0.1:18325/v1`
- Benchmark root: `/Users/skula/Documents/polyglot-benchmark`
- mcode commits under test: `e7fbecf` (`add workspace context discovery`) for run 1, `00bdec3` (`clarify run tests default argument`) for run 2, plus `10452ff` for the `go/poker` infra rerun
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

## Files

- Run 1 DB/log: `run1-workspace-context/results.db`, `run1-workspace-context/benchmark.log`
- Run 2 DB/log: `run2-verification-prompt/results.db`, `run2-verification-prompt/benchmark.log`
- `go/poker` rerun DB: `run2-verification-prompt/go-poker-rerun.db`
- HTML report: `aider-polyglot-workspace-context-report.html`

## Findings

The generic workspace context collector helped targeted A/B slices, but the first full run only reached `149/225`, far below little-coder's `177/225`. Compared with the prior corrected estimate of `148/225`, that was basically flat.

The second run changed only the generic verifier wording so the model is told to call `run_tests` with `test_cmd="default"`, rather than treating `run_tests default` as a literal argument. That moved the full run to `159/225`. The biggest gains were JavaScript (`37/49` to `42/49`), Rust (`17/30` to `19/30`), Go (`25/39` to `26/39`), and Java (`27/47` to `29/47`). It still missed little-coder by 18 tasks.

Run 2 had one infrastructure failure, `go/poker`, caused by non-UTF8 command output. Commit `10452ff` fixed command decoding with replacement characters and the single-task rerun produced a normal `unverified_diff_discarded` failure, not an infra failure. The corrected score stays `159/225`.

The main remaining failure bucket is still unverified diffs. The next iteration should focus on malformed tool-call recovery and better failed-test feedback, not more benchmark-local prompt content.
