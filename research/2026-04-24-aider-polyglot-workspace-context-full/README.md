# Aider Polyglot workspace context full run

This run tested the generic workspace context collector on the full Aider Polyglot benchmark with Qwen3.6-35B-A3B. The collector is not Polyglot-specific. It scans local authoritative Markdown docs such as `AGENTS.md`, `README.md`, `SPEC.md`, and `.docs/instructions.md`, then injects a small cited context block into the agent goal.

## Setup

- Model: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM endpoint on Blue Vela through local tunnel `http://127.0.0.1:18325/v1`
- Benchmark root: `/Users/skula/Documents/polyglot-benchmark`
- mcode commit under test: `e7fbecf` (`add workspace context discovery`)
- Baseline to beat: little-coder Qwen3.6 result, `177/225 = 78.67%`

## Command

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

## Results

| Language | Passed | Total | Rate |
|-|-:|-:|-:|
| Python | 25 | 34 | 73.5% |
| Go | 25 | 39 | 64.1% |
| Rust | 17 | 30 | 56.7% |
| JavaScript | 37 | 49 | 75.5% |
| C++ | 18 | 26 | 69.2% |
| Java | 27 | 47 | 57.4% |
| Total | 149 | 225 | 66.2% |

Terminal reasons:

| Reason | Count | Passed |
|-|-:|-:|
| submitted | 149 | 149 |
| unverified_diff_discarded | 62 | 0 |
| budget_exhausted | 14 | 0 |

## Files

- DB: `run1-workspace-context/results.db`
- Log: `run1-workspace-context/benchmark.log`
- HTML report: `aider-polyglot-workspace-context-report.html`

## Findings

The generic workspace context collector helped targeted A/B slices, but the full run only reached `149/225`, far below little-coder's `177/225`. Compared with the prior corrected estimate of `148/225`, this is basically flat. C++ improved slightly (`17/26` to `18/26`) and Python improved (`21/34` in the old full run to `25/34` here), but Java regressed relative to the Java-only corrected run (`30/47` to `27/47`).

The main remaining failure bucket is still unverified diffs, not infrastructure. The next iteration should focus on verification feedback quality and malformed tool-call recovery rather than more benchmark-local prompt content.
