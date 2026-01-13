# Benchmarks

## HumanEval

- Loader: `src/mcode/bench/humaneval.py`
- Source: `https://github.com/openai/human-eval` (downloaded as `HumanEval.jsonl.gz`)
- Evaluation: runs the provided `check(...)` function in the sandbox.

## MBPP

- Loader: `src/mcode/bench/mbpp.py`
- Source: `https://github.com/google-research/google-research/tree/master/mbpp` (downloaded as `mbpp.jsonl`)
- Evaluation: concatenates model code with the provided `assert ...` tests and executes in the sandbox.

## SWE-Bench Lite

Deferred for Phase 1.
