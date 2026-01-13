# Mellea Integration

Implementation: `src/mcode/llm/session.py`.

How it’s used during benchmarking:

- A single `mellea.start_session(...)` is opened per benchmark run (reused for all tasks/samples).
- Each model call uses `m.instruct(..., format=CodeOutput)` to request structured output containing a `code` field.
- Validation is done by executing in the Docker sandbox; failures are fed back via a separate “debug” prompt.
