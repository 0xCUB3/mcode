# Agentic Patch Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the "dump 60K of code in context" patch generation with Mellea's ReAct agent using search/read/edit tools.

**Architecture:** Define 3 repo tools as plain Python functions, wire them into Mellea's `react()` framework via `MelleaTool.from_callable`. The model explores the repo itself instead of getting everything upfront. After the react loop, `git diff` produces the unified patch.

**Tech Stack:** Mellea (react framework, MelleaTool, ChatContext), Python subprocess for git diff, existing BM25 for file tree ranking.

---

### Task 1: Add repo tools module

**Files:**
- Create: `src/mcode/agent/tools.py`
- Test: `tests/test_agent_tools.py`

**Step 1: Write failing tests**

Create `tests/test_agent_tools.py`:

```python
from __future__ import annotations

import subprocess

from mcode.agent.tools import make_tools


def test_search_code_finds_match(tmp_path):
    (tmp_path / "foo.py").write_text("def hello_world():\n    pass\n")
    tools = make_tools(str(tmp_path))
    result = tools["search_code"]("hello_world")
    assert "foo.py" in result
    assert "hello_world" in result


def test_search_code_no_match(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\n")
    tools = make_tools(str(tmp_path))
    result = tools["search_code"]("nonexistent_symbol")
    assert "No matches" in result


def test_search_code_caps_results(tmp_path):
    # Create 30 files each containing "target"
    for i in range(30):
        (tmp_path / f"mod_{i}.py").write_text(f"target = {i}\n")
    tools = make_tools(str(tmp_path))
    result = tools["search_code"]("target")
    # Should cap at 20 matches
    assert result.count("mod_") <= 20


def test_read_file_basic(tmp_path):
    (tmp_path / "foo.py").write_text("line1\nline2\nline3\nline4\nline5\n")
    tools = make_tools(str(tmp_path))
    result = tools["read_file"]("foo.py", 2, 4)
    assert "2: line2" in result
    assert "3: line3" in result
    assert "4: line4" in result
    assert "line1" not in result
    assert "line5" not in result


def test_read_file_not_found(tmp_path):
    tools = make_tools(str(tmp_path))
    result = tools["read_file"]("nope.py", 1, 10)
    assert "not found" in result.lower()


def test_read_file_caps_lines(tmp_path):
    content = "\n".join(f"line{i}" for i in range(500))
    (tmp_path / "big.py").write_text(content)
    tools = make_tools(str(tmp_path))
    result = tools["read_file"]("big.py", 1, 500)
    # Should cap at 200 lines
    assert result.count("\n") <= 201


def test_apply_edit_basic(tmp_path):
    (tmp_path / "foo.py").write_text("a = 1\nb = 2\nc = 3\n")
    tools = make_tools(str(tmp_path))
    result = tools["apply_edit"]("foo.py", 2, 2, "b = 42\n")
    assert "ok" in result.lower() or "success" in result.lower()
    assert (tmp_path / "foo.py").read_text() == "a = 1\nb = 42\nc = 3\n"


def test_apply_edit_syntax_error(tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    pass\n")
    tools = make_tools(str(tmp_path))
    result = tools["apply_edit"]("foo.py", 1, 1, "def hello(\n")
    assert "SyntaxError" in result
    # File should not be modified
    assert (tmp_path / "foo.py").read_text() == "def hello():\n    pass\n"


def test_apply_edit_non_python_skips_syntax(tmp_path):
    (tmp_path / "data.txt").write_text("line1\nline2\n")
    tools = make_tools(str(tmp_path))
    result = tools["apply_edit"]("data.txt", 1, 1, "def hello(\n")
    assert "ok" in result.lower() or "success" in result.lower()


def test_apply_edit_invalid_range(tmp_path):
    (tmp_path / "foo.py").write_text("a\nb\n")
    tools = make_tools(str(tmp_path))
    result = tools["apply_edit"]("foo.py", 5, 6, "x\n")
    assert "invalid" in result.lower() or "range" in result.lower()


def test_apply_edit_file_not_found(tmp_path):
    tools = make_tools(str(tmp_path))
    result = tools["apply_edit"]("nope.py", 1, 1, "x\n")
    assert "not found" in result.lower()


def test_diff_after_edits(tmp_path):
    # Init a git repo so we can diff
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("a = 1\nb = 2\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
    )

    tools = make_tools(str(tmp_path))
    tools["apply_edit"]("foo.py", 2, 2, "b = 42\n")

    from mcode.agent.tools import get_diff
    patch = get_diff(str(tmp_path))
    assert "-b = 2" in patch
    assert "+b = 42" in patch
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_tools.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mcode.agent'"

**Step 3: Write implementation**

Create `src/mcode/agent/__init__.py` (empty).

Create `src/mcode/agent/tools.py`:

```python
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def make_tools(repo_root: str) -> dict[str, callable]:
    """Create repo tools bound to a specific repo root. Returns name->callable dict."""
    root = Path(repo_root)

    def search_code(query: str) -> str:
        """Search for code matching a pattern in the repository.

        Args:
            query: regex pattern or literal string to search for
        """
        matches = []
        try:
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                if any(
                    d in p.parts
                    for d in (".git", "__pycache__", "node_modules", ".tox", "build", "dist")
                ):
                    continue
                if p.suffix not in (".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".h", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt", ".md", ".rst"):
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(query, line, re.IGNORECASE):
                        rel = str(p.relative_to(root))
                        matches.append(f"{rel}:{i}: {line.strip()}")
                        if len(matches) >= 20:
                            break
                if len(matches) >= 20:
                    break
        except re.error:
            # Fall back to literal search if regex is invalid
            return search_code(re.escape(query))
        if not matches:
            return "No matches found."
        return "\n".join(matches)

    def read_file(path: str, start_line: int, end_line: int) -> str:
        """Read lines from a file with line numbers.

        Args:
            path: file path relative to repo root
            start_line: first line to read (1-indexed)
            end_line: last line to read (1-indexed, inclusive)
        """
        fp = root / path
        if not fp.is_file():
            return f"Error: file not found: {path}"
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return f"Error: cannot read {path}"
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)
        # Cap at 200 lines per read
        if end_line - start_line + 1 > 200:
            end_line = start_line + 199
        selected = lines[start_line - 1 : end_line]
        numbered = [f"{start_line + i}: {line}" for i, line in enumerate(selected)]
        header = f"--- {path} (lines {start_line}-{end_line} of {len(lines)}) ---"
        return header + "\n" + "\n".join(numbered)

    def apply_edit(path: str, start_line: int, end_line: int, replacement: str) -> str:
        """Replace lines in a file. Validates Python syntax before accepting.

        Args:
            path: file path relative to repo root
            start_line: first line to replace (1-indexed, inclusive)
            end_line: last line to replace (1-indexed, inclusive)
            replacement: new text to replace those lines with
        """
        fp = root / path
        if not fp.is_file():
            return f"Error: file not found: {path}"
        try:
            original = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return f"Error: cannot read {path}"
        lines = original.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            return f"Error: invalid line range {start_line}-{end_line} (file has {len(lines)} lines)"
        end_line = min(end_line, len(lines))

        replace_lines = replacement.splitlines(keepends=True)
        if replacement and not replacement.endswith("\n"):
            replace_lines[-1] += "\n"

        modified_lines = lines[: start_line - 1] + replace_lines + lines[end_line:]
        modified = "".join(modified_lines)

        # Syntax gate for Python files
        if path.endswith(".py"):
            try:
                compile(modified, path, "exec")
            except SyntaxError as exc:
                return f"Error: SyntaxError in {path} line {exc.lineno}: {exc.msg}. Edit rejected, file unchanged."

        fp.write_text(modified, encoding="utf-8")
        return f"OK: replaced lines {start_line}-{end_line} in {path} ({len(replace_lines)} new lines)"

    return {
        "search_code": search_code,
        "read_file": read_file,
        "apply_edit": apply_edit,
    }


def get_diff(repo_root: str) -> str:
    """Get unified diff of all changes in the repo working tree."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.stdout
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_tools.py -v`
Expected: all PASS

**Step 5: Lint**

Run: `uv run ruff check src/mcode/agent/ tests/test_agent_tools.py && uv run ruff format --check src/mcode/agent/ tests/test_agent_tools.py`

**Step 6: Commit**

```bash
git add src/mcode/agent/ tests/test_agent_tools.py
git commit -m "add repo tools for agentic patch generation"
```

---

### Task 2: Replace generate_patch with react-based agent

**Files:**
- Modify: `src/mcode/llm/session.py`
- Test: `tests/test_agent_generate.py`

**Step 1: Write failing test**

Create `tests/test_agent_generate.py`:

```python
from __future__ import annotations

import subprocess
import os
from unittest.mock import MagicMock, patch


def test_generate_patch_calls_react(tmp_path):
    """generate_patch should call Mellea's react() with tools."""
    # Set up a git repo
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)

    from mcode.llm.session import LLMSession

    session = LLMSession(model_id="test", backend_name="ollama")

    # Mock the Mellea session and react
    mock_mellea = MagicMock()
    session._m = mock_mellea

    mock_thunk = MagicMock()
    mock_thunk.value = "done"

    with patch("mcode.llm.session.asyncio") as mock_asyncio:
        mock_asyncio.run.return_value = (mock_thunk, MagicMock())
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    # react was called via asyncio.run
    mock_asyncio.run.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_generate.py -v`
Expected: FAIL (generate_patch still uses old instruct-based approach)

**Step 3: Rewrite session.py**

Replace `generate_patch` and delete old models/functions. The full modified `src/mcode/llm/session.py`:

Delete these from session.py:
- `class FileEdit` (lines 16-19)
- `class PatchOutput` (lines 22-23)
- `class LineEdit` (lines 26-32)
- `class FileLocalization` (lines 35-36)
- `class LinePatchOutput` (lines 39-40)
- `def edits_to_patch` (lines 43-228)
- `def line_edits_to_patch` (lines 231-312)
- `def localize_files` method (lines 454-494)
- `def generate_patch` method (lines 507-551) -- replace with new version

Keep: `class CodeOutput`, `class LLMSession` (minus deleted methods), `def generate_code`, `def _code_system_prompt`, `open()`, `_backend_kwargs`, `_model_options`, `_strategy`, `check_available`.

New `generate_patch` method on LLMSession:

```python
    def generate_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        file_paths: list[str] | None = None,
        requirements: list | None = None,
        repo_root: str | None = None,
    ) -> str:
        """Run a react agent to produce a unified diff patch.

        Returns the unified diff string (empty string if no changes made).
        """
        import asyncio

        from mellea.backends.tools import MelleaTool
        from mellea.stdlib.context import ChatContext
        from mellea.stdlib.frameworks.react import react

        from mcode.agent.tools import get_diff, make_tools

        if repo_root is None:
            raise ValueError("repo_root is required for agentic patch generation")

        tool_fns = make_tools(repo_root)
        tools = [MelleaTool.from_callable(fn, name) for name, fn in tool_fns.items()]

        # Build goal from problem statement + file hints
        file_hint = ""
        if file_paths:
            file_hint = "\n\nFiles likely relevant (from BM25 ranking):\n" + "\n".join(
                f"  - {f}" for f in file_paths
            )
        hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""

        goal = (
            f"You are fixing a bug in {repo}.\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{file_hint}{hints_block}\n\n"
            "Use the tools to find the relevant code, understand it, and apply a fix. "
            "Call search_code to find relevant symbols, read_file to examine code, "
            "and apply_edit to make changes. When done, call final_answer with a summary."
        )

        system_prompt = (
            "You are an expert software engineer fixing a bug in an open-source repository. "
            "Use the provided tools to explore the codebase and apply minimal, targeted fixes."
        )

        ctx = ChatContext()
        loop_budget = max(1, self.loop_budget) * 5  # react turns, not repair attempts

        try:
            thunk, _ = asyncio.run(
                react(
                    goal=goal,
                    context=ctx,
                    backend=self._m.backend,
                    tools=tools,
                    loop_budget=loop_budget,
                    model_options=self._model_options(system_prompt=system_prompt),
                )
            )
        except RuntimeError as e:
            if "could not complete react loop" in str(e):
                pass  # Ran out of budget, still return whatever diff exists
            else:
                raise

        return get_diff(repo_root)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_generate.py -v`
Expected: PASS

**Step 5: Lint**

Run: `uv run ruff check src/mcode/llm/session.py tests/test_agent_generate.py && uv run ruff format --check src/mcode/llm/session.py tests/test_agent_generate.py`

**Step 6: Commit**

```bash
git add src/mcode/llm/session.py tests/test_agent_generate.py
git commit -m "replace generate_patch with react-based agent"
```

---

### Task 3: Update runner.py and k8s script to use new generate_patch

**Files:**
- Modify: `src/mcode/bench/runner.py`
- Modify: `deploy/k8s/run-swebench-live-one.sh`

**Step 1: Update runner.py**

In `src/mcode/bench/runner.py`:

1. Remove import of `line_edits_to_patch`:
   - Change `from mcode.llm.session import LLMSession, line_edits_to_patch` to `from mcode.llm.session import LLMSession`

2. In both `_run_swebench_live_task` and `_run_swebench_task`, replace the pattern:
   ```python
   loc_files, loc_hints = localize_files(...)
   hints = ...
   result = self.llm.generate_patch(...)
   # later:
   patch, _ = line_edits_to_patch(result.value or "", ...)
   ```
   With:
   ```python
   loc_files, _ = localize_files(str(repo_root), task.problem_statement)
   patch = self.llm.generate_patch(
       repo=task.repo,
       problem_statement=task.problem_statement,
       hints_text=task.hints_text or "",
       file_paths=loc_files,
       repo_root=str(repo_root),
   )
   ```

3. Remove the `_patch_test` closure and `Requirement` wiring since the react agent handles the loop internally.

4. Remove `localize_files` import's `llm_session=self.llm` argument (no longer needed).

**Step 2: Update k8s script**

In `deploy/k8s/run-swebench-live-one.sh`, apply the same simplification:

- Remove `line_edits_to_patch` import
- Remove `_patch_test` closure
- Remove `Requirement` / `simple_validate` wiring
- Change `generate_patch` call to pass `repo_root=REPO_ROOT`
- Get `patch = session.generate_patch(...)` directly (returns string now)

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: tests that import deleted functions will fail -- fix in next task.

**Step 4: Commit**

```bash
git add src/mcode/bench/runner.py deploy/k8s/run-swebench-live-one.sh
git commit -m "wire agentic generate_patch into runner and k8s"
```

---

### Task 4: Delete old code and tests, update localize.py

**Files:**
- Modify: `src/mcode/context/localize.py` -- remove LLM narrowing path
- Delete tests: `tests/test_line_edits_to_patch.py`, `tests/test_edits_to_patch.py`
- Modify: `tests/test_localize_files.py` -- remove LLM session tests
- Modify: `scripts/local_smoke.py` -- simplify

**Step 1: Simplify localize.py**

In `src/mcode/context/localize.py`:

Remove the `llm_session` parameter and all LLM narrowing code from `localize()`. It should only do BM25 ranking and return `(file_paths, "")` -- no more hints text (the agent reads files itself).

```python
def localize(
    repo_root: str,
    problem_statement: str,
    *,
    bm25_top_n: int = 30,
) -> tuple[list[str], str]:
    """BM25 file localization. Returns (ranked_file_paths, "")."""
    all_files = collect_source_files(repo_root)
    if not all_files:
        return [], ""
    ranked = rank_bm25(all_files, problem_statement, repo_root, top_n=bm25_top_n)
    return ranked, ""
```

**Step 2: Delete old test files**

```bash
rm tests/test_line_edits_to_patch.py tests/test_edits_to_patch.py
```

**Step 3: Update test_localize_files.py**

Remove `test_localize_with_llm_session` and `test_localize_with_llm_session_fallback` tests. Remove `from unittest.mock import MagicMock`. Remove tests that check for `hints` content (the function no longer returns hints). Keep BM25 and file tree tests.

Update `test_localize_returns_files_and_hints` to just check files are returned:

```python
def test_localize_returns_files(tmp_path):
    (tmp_path / "foo.py").write_text("def foo():\n    pass\n")
    (tmp_path / "bar.py").write_text("def bar():\n    pass\n")
    files, hints = localize(str(tmp_path), "Fix the foo function")
    assert "foo.py" in files
    assert hints == ""
```

**Step 4: Update local_smoke.py**

Simplify `scripts/local_smoke.py` to use the new `generate_patch` which returns a diff string directly:

```python
with session.open():
    loc_files, _ = localize(repo_root, PROBLEM_STATEMENT)
    patch = session.generate_patch(
        repo="pylint-dev/pylint",
        problem_statement=PROBLEM_STATEMENT,
        file_paths=loc_files,
        repo_root=repo_root,
    )
print(f"patch ({len(patch)} chars):")
print(patch[:2000])
```

**Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

**Step 6: Lint everything**

Run: `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/`

**Step 7: Commit**

```bash
git add -A
git commit -m "delete old line-edit code, simplify localize to BM25 only"
```

---

### Task 5: Update claude_swebench_test.py and run smoke test

**Files:**
- Modify: `scripts/claude_swebench_test.py`

**Step 1: Rewrite claude test script**

The `claude -p` test no longer applies since `generate_patch` now uses Mellea's react internally. Rewrite the script to test the agentic pipeline with the local ollama model:

```python
#!/usr/bin/env python3
"""Smoke test: run 1-3 SWE-bench tasks through the agentic pipeline."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
os.environ["MCODE_CONTEXT_WINDOW"] = "65536"

from mcode.bench.swebench_live import load_swebench_live
from mcode.context.localize import localize
from mcode.execution.swebench_live import SWEbenchLiveSandbox
from mcode.llm.session import LLMSession

MODEL = os.environ.get("MODEL", "qwen3-coder:30b")
BACKEND = os.environ.get("BACKEND", "ollama")
N_TASKS = int(os.environ.get("N_TASKS", "3"))

tasks = load_swebench_live(None, split="verified")
tasks = [t for t in tasks if "conan" not in t.instance_id and "matplotlib" not in t.instance_id][:N_TASKS]

print(f"Agentic smoke test: {MODEL} ({BACKEND}), {len(tasks)} tasks")
results = []

for t in tasks:
    print(f"\n===== {t.instance_id} =====", flush=True)
    t0 = time.time()
    try:
        sandbox = SWEbenchLiveSandbox()
        with sandbox.repo_context(t) as repo_root:
            session = LLMSession(model_id=MODEL, backend_name=BACKEND, loop_budget=3)
            with session.open():
                loc_files, _ = localize(str(repo_root), t.problem_statement)
                patch = session.generate_patch(
                    repo=t.repo,
                    problem_statement=t.problem_statement,
                    file_paths=loc_files[:10],
                    repo_root=str(repo_root),
                )
            has_patch = bool(patch.strip())
            print(f"patch: {has_patch} ({len(patch)} chars)", flush=True)
            if has_patch:
                run = sandbox.evaluate_patch(task=t, patch=patch, run_id="smoke", timeout_s=600)
                resolved = run.resolved
            else:
                resolved = False
            elapsed = time.time() - t0
            print(f"  >> resolved={resolved} patch={has_patch} time={elapsed:.0f}s", flush=True)
            results.append({"id": t.instance_id, "resolved": resolved, "patch": has_patch})
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ERROR: {e}", flush=True)
        results.append({"id": t.instance_id, "resolved": False, "patch": False})

print("\n===== SUMMARY =====")
for r in results:
    print(f"  {r['id']}: patch={r['patch']} resolved={r['resolved']}")
```

**Step 2: Run smoke test locally**

Run: `PYTHONUNBUFFERED=1 MODEL=qwen3-coder:30b uv run python scripts/claude_swebench_test.py`

Observe: the model should make tool calls (search_code, read_file, apply_edit) visible in the Mellea logs. Check that patches are produced.

**Step 3: Commit**

```bash
git add scripts/claude_swebench_test.py
git commit -m "rewrite smoke test for agentic pipeline"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS

**Step 2: Lint**

Run: `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/`

**Step 3: Verify no stale references**

Run: `grep -r "line_edits_to_patch\|edits_to_patch\|LinePatchOutput\|LineEdit\|FileLocalization\|localize_files" src/ tests/ scripts/ --include="*.py"`

Expected: no matches (except possibly in git history).

**Step 4: Commit any fixups**

```bash
git add -A
git commit -m "final cleanup: remove stale references"
```
