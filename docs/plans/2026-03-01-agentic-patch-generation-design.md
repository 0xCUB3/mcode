# Agentic patch generation

Replace the current "dump 60K of source into context and ask for line edits" approach with Mellea's built-in ReAct agent. The model gets a problem statement, a file tree, and tools to explore the repo itself.

## Problem

The current pipeline stuffs BM25-ranked source files (60K chars) into the prompt and asks the model to produce line-range edits in one shot. Small models (7B-30B) can't handle this: they drown in irrelevant code, hallucinate line numbers, and produce broken patches. Every successful coding agent (Claude Code, Cursor, etc.) uses agentic tool use instead.

## Design

### Flow

1. BM25 produces a ranked file list (names only, no content)
2. `react(goal, tools=[search_code, read_file, apply_edit], loop_budget=N)` via Mellea's ReAct framework
3. Model searches for relevant code, reads specific sections, applies targeted edits
4. On `final_answer`, diff the working copy against the original to produce a unified patch
5. Run patch through Docker eval (SWE-bench) or syntax gate (local)

### Tools

Three tools, each a plain Python function converted via `MelleaTool.from_callable`:

- `search_code(query: str) -> str` -- grep the repo for a pattern, return file:line snippets (capped at 20 results)
- `read_file(path: str, start_line: int, end_line: int) -> str` -- read a range of lines with line numbers
- `apply_edit(path: str, start_line: int, end_line: int, replacement: str) -> str` -- apply a line-range edit to the working copy, run syntax gate on Python files, return success/error message

### What gets deleted

- `generate_patch()` method on LLMSession
- `LinePatchOutput`, `LineEdit` Pydantic models
- `line_edits_to_patch()` and `edits_to_patch()` functions
- The system prompt explaining line-range edit format
- LLM file localization (`localize_files()`, `FileLocalization` model)
- `localize()` LLM narrowing path (keep BM25 for file tree)
- All tests for line_edits_to_patch

### What stays

- BM25 `localize()` for producing the ranked file tree (gives the agent a starting point)
- Syntax gate (moves inside `apply_edit`)
- `SWEbenchLiveSandbox.evaluate_patch()` (still takes a unified diff)
- SOFAI and RepairTemplate strategies can wrap the react call for retry/escalation
- `scripts/claude_swebench_test.py` gets rewritten to use the new agentic path

### Key details

- The react agent works on a temp copy of the repo (the existing `repo_context()` already provides this)
- After the react loop, `git diff` on the working copy produces the unified patch
- `apply_edit` validates Python syntax before accepting the edit, returns the error if rejected so the model can fix it
- `search_code` uses ripgrep or Python grep, not the model -- keeps search fast and accurate
- `read_file` caps output at ~200 lines per call to keep context small
- The `final_answer` tool is built into Mellea's ReAct framework
- `loop_budget` controls max react turns (default 15, configurable)
