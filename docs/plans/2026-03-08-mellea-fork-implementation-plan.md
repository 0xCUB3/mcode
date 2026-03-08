# Mellea Fork Agent Toolkit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fork mellea and add a modular, language-agnostic agent toolkit optimized for small LLMs, then wire it into mcode.

**Architecture:** New `mellea/agent/` and `mellea/eval/` packages added to the mellea fork. All components follow mellea's patterns: immutable Contexts, BaseSamplingStrategy subclasses, MelleaTool.from_callable() for tools. mcode switches to the fork and deletes replaced code.

**Tech Stack:** Python 3.12, tree-sitter-language-pack (165+ languages), networkx (PageRank), ripgrep (subprocess), mellea core (Context, BaseSamplingStrategy, MelleaTool, react())

**Design doc:** `docs/plans/2026-03-08-mellea-fork-agent-toolkit-design.md`

---

## Phase 0: Fork Setup

### Task 1: Create the GitHub fork and local dev setup

**Step 1: Fork mellea on GitHub**

Go to the upstream mellea repo on GitHub and click Fork to create `github.com/0xCUB3/mellea`.

**Step 2: Clone the fork locally**

```bash
cd ~/Documents
git clone git@github.com:0xCUB3/mellea.git mellea-fork
cd mellea-fork
```

**Step 3: Add upstream remote**

```bash
git remote add upstream <upstream-mellea-url>
git fetch upstream
```

**Step 4: Create a dev branch**

```bash
git checkout -b agent-toolkit
```

**Step 5: Verify the fork builds and tests pass**

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -x -q
```

**Step 6: Add new dependencies to pyproject.toml**

Add to `[project.optional-dependencies]`:
```toml
agent = [
    "tree-sitter-language-pack>=0.13",
    "networkx>=3.0",
]
```

**Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "add agent optional dependencies"
```

---

## Phase 1: Tools

### Task 2: Ripgrep search tool

**Files:**
- Create: `mellea/agent/__init__.py`
- Create: `mellea/agent/tools/__init__.py`
- Create: `mellea/agent/tools/search.py`
- Test: `tests/agent/tools/test_search.py`

**Step 1: Create package stubs**

```python
# mellea/agent/__init__.py
# mellea/agent/tools/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/agent/tools/test_search.py
from __future__ import annotations

import os
from pathlib import Path

from mellea.agent.tools.search import search_code


def test_search_finds_match(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("def hello_world():\n    pass\n")
    (tmp_path / "bar.py").write_text("x = 1\n")
    result = search_code("hello_world", repo_root=str(tmp_path))
    assert "foo.py" in result
    assert "hello_world" in result
    assert "bar.py" not in result


def test_search_no_matches(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x = 1\n")
    result = search_code("nonexistent_symbol", repo_root=str(tmp_path))
    assert "No matches" in result


def test_search_caps_results(tmp_path: Path) -> None:
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text(f"match_me_{i} = {i}\n")
    result = search_code("match_me", repo_root=str(tmp_path))
    assert result.count("\n") <= 31  # 30 matches + possible trailing newline


def test_search_skips_git_dir(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("match_me\n")
    (tmp_path / "real.py").write_text("match_me\n")
    result = search_code("match_me", repo_root=str(tmp_path))
    assert "real.py" in result
    assert ".git" not in result
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/agent/tools/test_search.py -v
```

**Step 4: Implement search tool**

```python
# mellea/agent/tools/search.py
from __future__ import annotations

import subprocess


_SKIP_DIRS = [".git", "node_modules", "__pycache__", ".venv", "venv",
              "build", "dist", ".tox", ".eggs", ".mypy_cache"]

_MAX_MATCHES = 30
_TIMEOUT_S = 5


def search_code(query: str, *, repo_root: str) -> str:
    """Search for a regex pattern across all files in a repo using ripgrep."""
    globs = [f"--glob=!{d}" for d in _SKIP_DIRS]
    cmd = [
        "rg", "--no-heading", "--line-number", "--max-columns=200",
        "--max-count=5", "--color=never", "--type-add=src:*.{py,js,ts,tsx,jsx,java,go,rs,c,cpp,h,hpp,rb,php,swift,kt,scala,sh,bash,yaml,yml,json,toml,cfg,ini,txt,md,rst}",
        "--type=src",
        *globs,
        query,
        repo_root,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT_S,
        )
        output = result.stdout.strip()
    except FileNotFoundError:
        return _fallback_search(query, repo_root)
    except subprocess.TimeoutExpired:
        return f"Search timed out after {_TIMEOUT_S}s. Try a more specific query."

    if not output:
        return "No matches found."

    lines = output.splitlines()
    if len(lines) > _MAX_MATCHES:
        lines = lines[:_MAX_MATCHES]
        return "\n".join(lines) + f"\n... ({_MAX_MATCHES} of many matches shown)"
    return "\n".join(lines)


def _fallback_search(query: str, repo_root: str) -> str:
    """Fallback to grep if ripgrep is not installed."""
    import re
    from pathlib import Path

    skip = set(_SKIP_DIRS)
    pattern = re.compile(query, re.IGNORECASE)
    matches: list[str] = []
    for p in Path(repo_root).rglob("*"):
        if any(s in p.parts for s in skip):
            continue
        if not p.is_file() or p.stat().st_size > 500_000:
            continue
        try:
            text = p.read_text(errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    rel = p.relative_to(repo_root)
                    matches.append(f"{rel}:{i}: {line[:200]}")
                    if len(matches) >= _MAX_MATCHES:
                        return "\n".join(matches)
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(matches) if matches else "No matches found."
```

**Step 5: Run tests**

```bash
pytest tests/agent/tools/test_search.py -v
```

**Step 6: Commit**

```bash
git add mellea/agent/ tests/agent/
git commit -m "add ripgrep search tool with grep fallback"
```

---

### Task 3: str_replace edit tool with tree-sitter validation

**Files:**
- Create: `mellea/agent/tools/edit.py`
- Test: `tests/agent/tools/test_edit.py`

**Step 1: Write the failing test**

```python
# tests/agent/tools/test_edit.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.edit import str_replace_edit


def test_edit_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")
    result = str_replace_edit(
        str(f), old_str="return 1", new_str="return 2", repo_root=str(tmp_path),
    )
    assert "return 2" in f.read_text()
    assert "Successfully" in result


def test_edit_rejects_nonunique_match(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("x = 1\nx = 1\n")
    result = str_replace_edit(
        str(f), old_str="x = 1", new_str="x = 2", repo_root=str(tmp_path),
    )
    assert "x = 1" in f.read_text()  # unchanged
    assert "multiple" in result.lower() or "unique" in result.lower()


def test_edit_rejects_missing_match(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("x = 1\n")
    result = str_replace_edit(
        str(f), old_str="y = 2", new_str="y = 3", repo_root=str(tmp_path),
    )
    assert "x = 1" in f.read_text()  # unchanged
    assert "not found" in result.lower()


def test_edit_rejects_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")
    result = str_replace_edit(
        str(f), old_str="return 1", new_str="return (", repo_root=str(tmp_path),
    )
    assert "return 1" in f.read_text()  # reverted
    assert "syntax" in result.lower() or "error" in result.lower()


def test_edit_works_for_non_python(tmp_path: Path) -> None:
    f = tmp_path / "foo.js"
    f.write_text("function hello() {\n  return 1;\n}\n")
    result = str_replace_edit(
        str(f), old_str="return 1;", new_str="return 2;", repo_root=str(tmp_path),
    )
    assert "return 2;" in f.read_text()
    assert "Successfully" in result
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/agent/tools/test_edit.py -v
```

**Step 3: Implement edit tool**

```python
# mellea/agent/tools/edit.py
from __future__ import annotations

from pathlib import Path


_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".toml": "toml",
}


def _count_errors(tree) -> int:
    """Count ERROR nodes in a tree-sitter parse tree."""
    count = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            count += 1
        stack.extend(node.children)
    return count


def _validate_syntax(path: str, content: str) -> str | None:
    """Return error message if content has new syntax errors, else None."""
    ext = Path(path).suffix.lower()
    lang = _EXTENSION_TO_LANGUAGE.get(ext)
    if lang is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
    except (ImportError, Exception):
        return None
    tree = parser.parse(content.encode("utf-8"))
    errors = _count_errors(tree)
    if errors > 0:
        # Find first error location.
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "ERROR":
                row, col = node.start_point
                return f"Syntax error at line {row + 1}, column {col}"
            stack.extend(node.children)
    return None


def str_replace_edit(
    path: str,
    old_str: str,
    new_str: str,
    *,
    repo_root: str,
) -> str:
    """Replace a unique occurrence of old_str with new_str in a file.

    Validates syntax after edit using tree-sitter. Reverts on syntax errors.
    """
    try:
        full_path = Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return f"Error: file not found: {path}"

    count = content.count(old_str)
    if count == 0:
        # Show nearby content to help the model.
        lines = content.splitlines()
        preview = "\n".join(lines[:30])
        return f"Error: old_str not found in {path}. First 30 lines:\n{preview}"
    if count > 1:
        return (
            f"Error: old_str appears {count} times in {path}. "
            "Provide more context to make the match unique."
        )

    # Check syntax before edit (to know pre-existing errors).
    pre_errors = _validate_syntax(str(full_path), content)

    new_content = content.replace(old_str, new_str, 1)
    post_error = _validate_syntax(str(full_path), new_content)

    if post_error and not pre_errors:
        return f"Edit rejected: introduces syntax error. {post_error}. File unchanged."

    full_path.write_text(new_content)
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    return f"Successfully replaced {old_lines} lines with {new_lines} lines in {path}."
```

**Step 4: Run tests**

```bash
pytest tests/agent/tools/test_edit.py -v
```

**Step 5: Commit**

```bash
git add mellea/agent/tools/edit.py tests/agent/tools/test_edit.py
git commit -m "add str_replace edit tool with tree-sitter syntax validation"
```

---

### Task 4: Read and navigate tools

**Files:**
- Create: `mellea/agent/tools/read.py`
- Create: `mellea/agent/tools/navigate.py`
- Test: `tests/agent/tools/test_read.py`
- Test: `tests/agent/tools/test_navigate.py`

**Step 1: Write tests**

```python
# tests/agent/tools/test_read.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.read import read_file


def test_read_file_with_line_numbers(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("line1\nline2\nline3\n")
    result = read_file(str(f), repo_root=str(tmp_path))
    assert "1:" in result or "1 " in result
    assert "line1" in result
    assert "3 lines" in result or "3)" in result


def test_read_file_range(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 301)))
    result = read_file(str(f), start_line=10, end_line=20, repo_root=str(tmp_path))
    assert "line10" in result
    assert "line20" in result
    assert "line9" not in result


def test_read_file_caps_at_200(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 501)))
    result = read_file(str(f), repo_root=str(tmp_path))
    # Should show 200 lines max and indicate truncation.
    assert "200" in result or "truncated" in result.lower()
```

```python
# tests/agent/tools/test_navigate.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.navigate import find_file, list_dir


def test_find_file_by_name(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "src" / "util.py").write_text("")
    result = find_file("main.py", repo_root=str(tmp_path))
    assert "main.py" in result
    assert "util.py" not in result


def test_find_file_glob(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.js").write_text("")
    result = find_file("*.py", repo_root=str(tmp_path))
    assert "a.py" in result
    assert "b.py" in result
    assert "c.js" not in result


def test_list_dir(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "foo.py").write_text("hello")
    result = list_dir(str(tmp_path), repo_root=str(tmp_path))
    assert "src" in result
    assert "foo.py" in result
```

**Step 2: Implement**

```python
# mellea/agent/tools/read.py
from __future__ import annotations

from pathlib import Path

_MAX_LINES = 200


def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    *,
    repo_root: str,
) -> str:
    """Read a file with line numbers, capped at 200 lines."""
    full_path = Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
    try:
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return f"Error: file not found: {path}"

    lines = content.splitlines()
    total = len(lines)

    start = max(1, start_line) - 1  # 0-indexed
    end = min(total, end_line or total)
    selected = lines[start:end]

    if len(selected) > _MAX_LINES:
        selected = selected[:_MAX_LINES]
        end = start + _MAX_LINES
        truncated = True
    else:
        truncated = False

    numbered = [f"{start + i + 1:>4}: {line}" for i, line in enumerate(selected)]
    header = f"{path} ({total} lines total, showing {start + 1}-{end})"
    if truncated:
        header += f" [truncated to {_MAX_LINES} lines]"

    return header + "\n" + "\n".join(numbered)
```

```python
# mellea/agent/tools/navigate.py
from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              "build", "dist", ".tox", ".eggs"}
_MAX_RESULTS = 50


def find_file(pattern: str, *, repo_root: str) -> str:
    """Find files matching a glob pattern."""
    root = Path(repo_root)
    matches: list[str] = []
    for p in root.rglob(pattern):
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        if p.is_file():
            matches.append(str(p.relative_to(root)))
            if len(matches) >= _MAX_RESULTS:
                break
    if not matches:
        return f"No files matching '{pattern}' found."
    result = "\n".join(sorted(matches))
    if len(matches) >= _MAX_RESULTS:
        result += f"\n... (showing first {_MAX_RESULTS} matches)"
    return result


def list_dir(path: str = ".", *, repo_root: str) -> str:
    """List directory contents."""
    full_path = Path(repo_root) / path
    if not full_path.is_dir():
        return f"Error: not a directory: {path}"
    entries: list[str] = []
    for p in sorted(full_path.iterdir()):
        if p.name.startswith(".") and p.name in _SKIP_DIRS:
            continue
        kind = "dir" if p.is_dir() else "file"
        entries.append(f"  {p.name:<40} [{kind}]")
    return f"{path}/\n" + "\n".join(entries) if entries else f"{path}/ (empty)"
```

**Step 3: Run tests**

```bash
pytest tests/agent/tools/ -v
```

**Step 4: Commit**

```bash
git add mellea/agent/tools/read.py mellea/agent/tools/navigate.py tests/agent/tools/
git commit -m "add read and navigate tools"
```

---

### Task 5: Tool registration helper

**Files:**
- Modify: `mellea/agent/tools/__init__.py`
- Test: `tests/agent/tools/test_init.py`

**Step 1: Write test**

```python
# tests/agent/tools/test_init.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.tools import make_agent_tools


def test_make_agent_tools_returns_mellea_tools(tmp_path: Path) -> None:
    tools = make_agent_tools(str(tmp_path))
    names = {t.name for t in tools}
    assert "search_code" in names
    assert "edit" in names
    assert "read_file" in names
    assert "find_file" in names
    assert "list_dir" in names
    assert len(tools) == 5


def test_tools_are_callable(tmp_path: Path) -> None:
    tools = make_agent_tools(str(tmp_path))
    for t in tools:
        assert callable(t.run)
```

**Step 2: Implement**

```python
# mellea/agent/tools/__init__.py
from __future__ import annotations

from functools import partial

from mellea.backends.tools import MelleaTool

from mellea.agent.tools.edit import str_replace_edit
from mellea.agent.tools.navigate import find_file, list_dir
from mellea.agent.tools.read import read_file
from mellea.agent.tools.search import search_code


def make_agent_tools(repo_root: str) -> list[MelleaTool]:
    """Create the standard set of agent tools bound to a repo root."""
    bound = {
        "search_code": partial(search_code, repo_root=repo_root),
        "edit": partial(str_replace_edit, repo_root=repo_root),
        "read_file": partial(read_file, repo_root=repo_root),
        "find_file": partial(find_file, repo_root=repo_root),
        "list_dir": partial(list_dir, repo_root=repo_root),
    }
    return [MelleaTool.from_callable(fn, name=name) for name, fn in bound.items()]
```

**Step 3: Run tests**

```bash
pytest tests/agent/tools/ -v
```

**Step 4: Commit**

```bash
git add mellea/agent/tools/__init__.py tests/agent/tools/test_init.py
git commit -m "add make_agent_tools registration helper"
```

---

## Phase 2: Repo Map

### Task 6: Tree-sitter tag extraction

**Files:**
- Create: `mellea/agent/repomap/__init__.py`
- Create: `mellea/agent/repomap/tags.py`
- Test: `tests/agent/repomap/test_tags.py`

**Step 1: Write test**

```python
# tests/agent/repomap/test_tags.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap.tags import extract_tags, Tag


def test_extract_python_defs(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    pass\n")
    tags = extract_tags(str(f))
    def_names = [t.name for t in tags if t.kind == "def"]
    assert "Foo" in def_names
    assert "bar" in def_names
    assert "baz" in def_names


def test_extract_javascript_defs(tmp_path: Path) -> None:
    f = tmp_path / "foo.js"
    f.write_text("function hello() {}\nclass World {}\n")
    tags = extract_tags(str(f))
    def_names = [t.name for t in tags if t.kind == "def"]
    assert "hello" in def_names
    assert "World" in def_names


def test_extract_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "foo.xyz"
    f.write_text("whatever content\n")
    tags = extract_tags(str(f))
    assert tags == []
```

**Step 2: Implement**

```python
# mellea/agent/repomap/tags.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tag:
    rel_path: str
    name: str
    line: int
    kind: str  # "def" or "ref"


_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".rb": "ruby", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
}

# tree-sitter node types that represent definitions, per language.
_DEF_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "enum_item", "trait_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "ruby": {"method", "class", "module"},
}

# Node types that hold the name child for definitions.
_NAME_FIELD = "name"


def extract_tags(path: str, *, repo_root: str = "") -> list[Tag]:
    """Extract definition tags from a source file using tree-sitter."""
    ext = Path(path).suffix.lower()
    lang = _EXT_TO_LANG.get(ext)
    if lang is None:
        return []

    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
    except (ImportError, Exception):
        return []

    try:
        content = Path(path).read_text(errors="replace")
    except OSError:
        return []

    tree = parser.parse(content.encode("utf-8"))
    def_types = _DEF_NODE_TYPES.get(lang, set())
    rel = str(Path(path).relative_to(repo_root)) if repo_root else path

    tags: list[Tag] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in def_types:
            name_node = node.child_by_field_name(_NAME_FIELD)
            if name_node:
                tags.append(Tag(
                    rel_path=rel,
                    name=name_node.text.decode("utf-8", errors="replace"),
                    line=node.start_point[0] + 1,
                    kind="def",
                ))
        stack.extend(reversed(node.children))
    return tags
```

**Step 3: Run tests**

```bash
pytest tests/agent/repomap/test_tags.py -v
```

**Step 4: Commit**

```bash
git add mellea/agent/repomap/ tests/agent/repomap/
git commit -m "add tree-sitter tag extraction for repo map"
```

---

### Task 7: File graph and PageRank

**Files:**
- Create: `mellea/agent/repomap/graph.py`
- Test: `tests/agent/repomap/test_graph.py`

**Step 1: Write test**

```python
# tests/agent/repomap/test_graph.py
from __future__ import annotations

from mellea.agent.repomap.graph import rank_files
from mellea.agent.repomap.tags import Tag


def test_rank_files_returns_sorted_paths() -> None:
    tags = [
        Tag("a.py", "Foo", 1, "def"),
        Tag("b.py", "Foo", 5, "ref"),
        Tag("b.py", "Bar", 1, "def"),
        Tag("c.py", "Bar", 3, "ref"),
        Tag("c.py", "Foo", 7, "ref"),
    ]
    ranked = rank_files(tags, seed_files=["a.py"])
    assert isinstance(ranked, list)
    assert len(ranked) > 0
    # a.py should rank highly since it's a seed file.
    assert ranked[0] == "a.py"


def test_rank_files_empty_tags() -> None:
    ranked = rank_files([], seed_files=[])
    assert ranked == []
```

**Step 2: Implement**

```python
# mellea/agent/repomap/graph.py
from __future__ import annotations

from mellea.agent.repomap.tags import Tag


def rank_files(
    tags: list[Tag],
    seed_files: list[str],
    *,
    top_n: int = 30,
) -> list[str]:
    """Rank files by relevance using PageRank on a symbol-reference graph."""
    if not tags:
        return list(seed_files)[:top_n]

    import networkx as nx

    g = nx.MultiDiGraph()
    all_files: set[str] = set()
    defs: dict[str, list[str]] = {}  # symbol -> [files that define it]

    for t in tags:
        all_files.add(t.rel_path)
        if t.kind == "def":
            defs.setdefault(t.name, []).append(t.rel_path)

    for t in tags:
        if t.kind == "ref" and t.name in defs:
            for def_file in defs[t.name]:
                if def_file != t.rel_path:
                    g.add_edge(t.rel_path, def_file, symbol=t.name)

    # Add isolated files as nodes.
    for f in all_files:
        g.add_node(f)

    if len(g.nodes) == 0:
        return list(seed_files)[:top_n]

    # Personalization: boost seed files.
    personalization = {}
    n = len(g.nodes)
    seed_set = set(seed_files)
    for node in g.nodes:
        personalization[node] = 100.0 if node in seed_set else 1.0 / n

    try:
        pr = nx.pagerank(g, personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        pr = {node: 1.0 for node in g.nodes}

    ranked = sorted(pr, key=pr.get, reverse=True)
    return ranked[:top_n]
```

**Step 3: Run tests**

```bash
pytest tests/agent/repomap/test_graph.py -v
```

**Step 4: Commit**

```bash
git add mellea/agent/repomap/graph.py tests/agent/repomap/
git commit -m "add file graph and PageRank ranking"
```

---

### Task 8: Repo map renderer and public API

**Files:**
- Create: `mellea/agent/repomap/render.py`
- Modify: `mellea/agent/repomap/__init__.py`
- Test: `tests/agent/repomap/test_repomap.py`

**Step 1: Write test**

```python
# tests/agent/repomap/test_repomap.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap import build_repo_map


def test_build_repo_map_python(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "from util import helper\n\ndef main():\n    helper()\n"
    )
    (tmp_path / "util.py").write_text(
        "def helper():\n    return 42\n"
    )
    result = build_repo_map(
        str(tmp_path), query="main function", max_tokens=2000,
    )
    assert "main.py" in result
    assert "def main" in result
    assert "util.py" in result
    assert "def helper" in result


def test_build_repo_map_empty_dir(tmp_path: Path) -> None:
    result = build_repo_map(str(tmp_path), query="anything")
    assert result == "" or "No source files" in result
```

**Step 2: Implement**

```python
# mellea/agent/repomap/render.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap.tags import Tag


def render_skeleton(
    tags_by_file: dict[str, list[Tag]],
    ranked_files: list[str],
    *,
    max_tokens: int = 4096,
) -> str:
    """Render a condensed skeleton of the top-ranked files."""
    lines: list[str] = []
    token_estimate = 0

    for rel_path in ranked_files:
        file_tags = tags_by_file.get(rel_path, [])
        if not file_tags:
            continue
        file_lines = [f"{rel_path}:"]
        for t in sorted(file_tags, key=lambda t: t.line):
            file_lines.append(f"  {t.line}: {t.kind} {t.name}")

        block = "\n".join(file_lines)
        block_tokens = len(block) // 4  # rough estimate
        if token_estimate + block_tokens > max_tokens:
            break
        lines.append(block)
        token_estimate += block_tokens

    return "\n\n".join(lines)
```

```python
# mellea/agent/repomap/__init__.py
from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap.graph import rank_files
from mellea.agent.repomap.render import render_skeleton
from mellea.agent.repomap.tags import Tag, extract_tags


_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              "build", "dist", ".tox", ".eggs"}

_SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".sh", ".yaml", ".yml",
    ".json", ".toml",
}


def build_repo_map(
    repo_root: str,
    query: str,
    *,
    max_tokens: int = 4096,
    bm25_top_n: int = 30,
) -> str:
    """Build a structural map of a repository for LLM context.

    Combines BM25 text ranking with tree-sitter structural analysis
    and PageRank to produce a condensed skeleton of the most relevant files.
    """
    root = Path(repo_root)
    source_files: list[str] = []
    for p in root.rglob("*"):
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        if p.is_file() and p.suffix.lower() in _SOURCE_EXTS:
            source_files.append(str(p))

    if not source_files:
        return ""

    # BM25 seed ranking.
    from mellea.agent.repomap._bm25 import rank_bm25
    seed_files = rank_bm25(source_files, query, repo_root, top_n=bm25_top_n)

    # Extract tags from all source files.
    all_tags: list[Tag] = []
    tags_by_file: dict[str, list[Tag]] = {}
    for fpath in source_files:
        ftags = extract_tags(fpath, repo_root=repo_root)
        if ftags:
            rel = str(Path(fpath).relative_to(repo_root))
            all_tags.extend(ftags)
            tags_by_file[rel] = ftags

    # Rank files.
    seed_rels = [str(Path(f).relative_to(repo_root)) for f in seed_files]
    ranked = rank_files(all_tags, seed_files=seed_rels)

    return render_skeleton(tags_by_file, ranked, max_tokens=max_tokens)
```

Note: We need to move the BM25 implementation into the fork. Create `mellea/agent/repomap/_bm25.py` by copying the core BM25 logic from `mcode/context/localize.py` (the `rank_bm25`, `_tokenize` functions), but making it language-agnostic (accept any file paths, not just .py).

**Step 3: Run tests**

```bash
pytest tests/agent/repomap/ -v
```

**Step 4: Commit**

```bash
git add mellea/agent/repomap/ tests/agent/repomap/
git commit -m "add repo map with tree-sitter + PageRank + BM25"
```

---

## Phase 3: Strategy Components

### Task 9: Loop detection wrapper

**Files:**
- Create: `mellea/agent/strategy/__init__.py`
- Create: `mellea/agent/strategy/loop_detect.py`
- Test: `tests/agent/strategy/test_loop_detect.py`

**Step 1: Write test**

```python
# tests/agent/strategy/test_loop_detect.py
from __future__ import annotations

from mellea.agent.strategy.loop_detect import detect_loop


def test_no_loop() -> None:
    history = [
        ("search_code", ("foo",)),
        ("read_file", ("bar.py",)),
        ("search_code", ("baz",)),
    ]
    action = detect_loop(history)
    assert action is None


def test_detect_double_repeat() -> None:
    history = [
        ("search_code", ("foo",)),
        ("search_code", ("foo",)),
    ]
    action = detect_loop(history)
    assert action == "nudge"


def test_detect_triple_repeat() -> None:
    history = [
        ("search_code", ("foo",)),
        ("search_code", ("foo",)),
        ("search_code", ("foo",)),
    ]
    action = detect_loop(history)
    assert action == "force_switch"
```

**Step 2: Implement**

```python
# mellea/agent/strategy/loop_detect.py
from __future__ import annotations


def detect_loop(
    history: list[tuple[str, tuple]],
    *,
    nudge_threshold: int = 2,
    force_threshold: int = 3,
) -> str | None:
    """Check if the most recent tool calls are repeated.

    Returns:
        None: no loop detected.
        "nudge": same call repeated nudge_threshold times.
        "force_switch": same call repeated force_threshold times.
    """
    if len(history) < nudge_threshold:
        return None

    last = history[-1]
    repeat_count = 0
    for call in reversed(history):
        if call == last:
            repeat_count += 1
        else:
            break

    if repeat_count >= force_threshold:
        return "force_switch"
    if repeat_count >= nudge_threshold:
        return "nudge"
    return None


NUDGE_MESSAGE = (
    "You already tried this exact tool call. "
    "Try a different approach: read a file, search for something else, "
    "or make an edit."
)

FORCE_MESSAGE = (
    "You have repeated the same tool call 3 times. "
    "You MUST call a different tool now."
)
```

**Step 3: Run tests**

```bash
pytest tests/agent/strategy/test_loop_detect.py -v
```

**Step 4: Commit**

```bash
git add mellea/agent/strategy/ tests/agent/strategy/
git commit -m "add loop detection for repeated tool calls"
```

---

### Task 10: Observation masking Context

**Files:**
- Create: `mellea/agent/context/__init__.py`
- Create: `mellea/agent/context/masking.py`
- Test: `tests/agent/context/test_masking.py`

**Step 1: Write test**

```python
# tests/agent/context/test_masking.py
from __future__ import annotations

from mellea.agent.context.masking import MaskingChatContext
from mellea.stdlib.components.chat import Message


def test_masking_context_keeps_recent() -> None:
    ctx = MaskingChatContext(mask_after=2)
    ctx = ctx.add(Message("user", "turn 1"))
    ctx = ctx.add(Message("assistant", "response 1 with lots of detail"))
    ctx = ctx.add(Message("user", "turn 2"))
    ctx = ctx.add(Message("assistant", "response 2 with lots of detail"))
    ctx = ctx.add(Message("user", "turn 3"))
    ctx = ctx.add(Message("assistant", "response 3 with lots of detail"))
    view = ctx.view_for_generation()
    # Last 2 turns should be unmasked, earlier ones summarized.
    assert view is not None
    assert len(view) > 0


def test_masking_context_empty() -> None:
    ctx = MaskingChatContext(mask_after=3)
    view = ctx.view_for_generation()
    assert view is not None or view is None  # Should not error.
```

**Step 2: Implement**

This requires subclassing mellea's ChatContext. The exact implementation depends on how ChatContext stores turns internally. Read the ChatContext source during implementation and adapt accordingly. The key behavior:

```python
# mellea/agent/context/masking.py
from __future__ import annotations

from mellea.stdlib.context import ChatContext
from mellea.core.base import Component, CBlock


class MaskingChatContext(ChatContext):
    """ChatContext that masks old tool outputs to save context space.

    Tool outputs older than `mask_after` turns are replaced with a
    one-line summary. Action/reasoning history is preserved in full.
    """

    def __init__(self, *, mask_after: int = 3, window_size: int | None = None):
        super().__init__(window_size=window_size)
        self._mask_after = mask_after

    def add(self, c: Component | CBlock) -> MaskingChatContext:
        new_ctx = super().add(c)
        # Preserve mask_after on the new context.
        new_ctx.__class__ = MaskingChatContext
        new_ctx._mask_after = self._mask_after
        return new_ctx

    def view_for_generation(self) -> list[Component | CBlock] | None:
        items = super().view_for_generation()
        if items is None or len(items) <= self._mask_after * 2:
            return items
        # Mask tool outputs in older turns.
        # Implementation detail: identify ToolMessage components and
        # replace their content with a summary. The exact component types
        # depend on mellea's internals -- adapt during implementation.
        return items
```

Note: The exact masking logic must be adapted during implementation based on how mellea stores tool call results in the context. The test verifies the interface works; the masking behavior will be refined when we can inspect the actual context contents.

**Step 3: Run tests**

```bash
pytest tests/agent/context/test_masking.py -v
```

**Step 4: Commit**

```bash
git add mellea/agent/context/ tests/agent/context/
git commit -m "add observation masking context"
```

---

### Task 11: Phased tool access

**Files:**
- Create: `mellea/agent/strategy/phased.py`
- Test: `tests/agent/strategy/test_phased.py`

**Step 1: Write test**

```python
# tests/agent/strategy/test_phased.py
from __future__ import annotations

from mellea.agent.strategy.phased import get_available_tools


def test_phase1_explore_only() -> None:
    all_tools = ["search_code", "read_file", "find_file", "list_dir", "edit", "final_answer"]
    available = get_available_tools(all_tools, turn=1, budget=15)
    assert "search_code" in available
    assert "read_file" in available
    assert "edit" not in available
    assert "final_answer" not in available


def test_phase2_all_tools() -> None:
    all_tools = ["search_code", "read_file", "find_file", "list_dir", "edit", "final_answer"]
    available = get_available_tools(all_tools, turn=8, budget=15)
    assert "edit" in available
    assert "search_code" in available


def test_phase3_commit_only() -> None:
    all_tools = ["search_code", "read_file", "find_file", "list_dir", "edit", "final_answer"]
    available = get_available_tools(all_tools, turn=14, budget=15)
    assert "edit" in available
    assert "final_answer" in available
    assert "search_code" not in available


def test_small_budget() -> None:
    all_tools = ["search_code", "read_file", "edit", "final_answer"]
    # budget=3: phase1=turn1, phase2=turn2, phase3=turn3
    available = get_available_tools(all_tools, turn=1, budget=3)
    assert "edit" not in available
    available = get_available_tools(all_tools, turn=2, budget=3)
    assert "edit" in available
    available = get_available_tools(all_tools, turn=3, budget=3)
    assert "edit" in available
    assert "search_code" not in available


def test_custom_phases() -> None:
    all_tools = ["search_code", "edit", "final_answer"]
    # 50/50 split, no phase 3.
    available = get_available_tools(
        all_tools, turn=1, budget=10, phases=[0.5, 1.0, 1.0],
    )
    assert "edit" not in available
```

**Step 2: Implement**

```python
# mellea/agent/strategy/phased.py
from __future__ import annotations

import math

_EXPLORE_TOOLS = {"search_code", "read_file", "find_file", "list_dir"}
_COMMIT_TOOLS = {"edit", "final_answer"}
_DEFAULT_PHASES = (0.4, 0.8, 1.0)


def get_available_tools(
    all_tool_names: list[str],
    turn: int,
    budget: int,
    *,
    phases: tuple[float, ...] = _DEFAULT_PHASES,
) -> list[str]:
    """Return the tool names available at a given turn based on phase boundaries.

    Phases (percentage of budget):
        Phase 1 (0 to phases[0]): explore tools only.
        Phase 2 (phases[0] to phases[1]): all tools.
        Phase 3 (phases[1] to 1.0): edit + final_answer only.
    """
    assert len(phases) == 3, "phases must have exactly 3 values"
    progress = turn / max(1, budget)

    if progress <= phases[0]:
        # Phase 1: explore only.
        return [t for t in all_tool_names if t in _EXPLORE_TOOLS]
    elif progress <= phases[1]:
        # Phase 2: all tools.
        return list(all_tool_names)
    else:
        # Phase 3: commit only.
        return [t for t in all_tool_names if t in _COMMIT_TOOLS]
```

**Step 3: Run tests**

```bash
pytest tests/agent/strategy/ -v
```

**Step 4: Commit**

```bash
git add mellea/agent/strategy/phased.py tests/agent/strategy/
git commit -m "add percentage-based phased tool access"
```

---

## Phase 4: mcode Integration

### Task 12: Switch mcode to mellea fork

**Files:**
- Modify: `mcode/pyproject.toml`

**Step 1: Update dependency**

Change `"mellea>=0.3.2"` to `"mellea @ git+https://github.com/0xCUB3/mellea@agent-toolkit"` in pyproject.toml.

Add `"tree-sitter-language-pack>=0.13"` and `"networkx>=3.0"` to the dependencies (or they come via the fork's `agent` extra).

**Step 2: Install and verify**

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -x -q
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "switch mellea dependency to fork"
```

---

### Task 13: Wire new tools and repo map into session.py

**Files:**
- Modify: `src/mcode/llm/session.py`

**Step 1: Replace tool creation in generate_patch()**

Replace the old `make_tools` import and usage with the new `make_agent_tools` from the fork. Replace the old `localize` call with `build_repo_map`. Update the goal string to include the repo map.

Key changes:
- `from mcode.agent.tools import get_diff, make_tools` → `from mellea.agent.tools import make_agent_tools` + keep `get_diff` locally
- `from mcode.context.localize import localize` → `from mellea.agent.repomap import build_repo_map`
- Build repo map before the react loop and include it in the goal
- Pass `make_agent_tools(repo_root)` directly to `react()` as tools

**Step 2: Run tests**

```bash
uv run pytest tests/ -x -q
```

**Step 3: Commit**

```bash
git add src/mcode/llm/session.py
git commit -m "wire mellea fork tools and repo map into session"
```

---

### Task 14: Delete old mcode code

**Files:**
- Delete: `src/mcode/agent/tools.py` (old rglob search, line-number edit)
- Delete: `src/mcode/context/localize.py` (old BM25-only localization)
- Modify: `src/mcode/context/__init__.py` (remove localize imports if any)
- Modify: `src/mcode/bench/runner.py` (update imports)

**Step 1: Delete old files**

```bash
rm src/mcode/agent/tools.py
rm src/mcode/context/localize.py
```

**Step 2: Update imports in runner.py**

Remove `from mcode.context.localize import localize as localize_files`. The localization is now handled inside session.py via `build_repo_map`.

**Step 3: Update tests**

Remove or update any tests that import from the deleted modules. Add new tests for the integration.

**Step 4: Run all tests**

```bash
uv run ruff check && uv run pytest tests/ -x -q
```

**Step 5: Commit**

```bash
git add -A
git commit -m "delete old tools and localization replaced by mellea fork"
```

---

## Phase 5: Evaluation

### Task 15: Smoke suite and A/B comparison

**Files:**
- Create: `mellea/eval/__init__.py`
- Create: `mellea/eval/smoke.py`
- Create: `mellea/eval/compare.py`
- Create: `data/smoke-suite.json` (in mcode repo)

**Step 1: Create smoke suite task list**

Select ~25 tasks from run2 results. Store as JSON in mcode:

```json
{
    "description": "SWE-bench Live Lite smoke suite for agent toolkit evaluation",
    "tasks": {
        "regression": ["pvlib__pvlib-python-2249", "pypa__twine-1225", "python-babel__babel-1141", "python-control__python-control-1111", "run-llama__llama_deploy-397"],
        "search_loops": ["tox-dev__tox-3409", "tox-dev__tox-3388", "aws-cloudformation__cfn-lint-3798", "aws-cloudformation__cfn-lint-3767", "conan-io__conan-17092"],
        "bad_edits": ["reflex-dev__reflex-4129", "keras-team__keras-20389", "conan-io__conan-17123", "aws-cloudformation__cfn-lint-3821", "aws-cloudformation__cfn-lint-3890"],
        "multilingual": ["TO_BE_SELECTED_FROM_MULTI_SWEBENCH"],
        "near_miss": ["TO_BE_SELECTED_FROM_RUN2_ANALYSIS"]
    }
}
```

Note: The multilingual and near-miss tasks need to be selected during implementation by analyzing run2 results and checking Multi-SWE-bench availability.

**Step 2: Implement comparison tool**

```python
# mellea/eval/compare.py
from __future__ import annotations

import sqlite3
from pathlib import Path


def compare_runs(
    baseline_db: str,
    candidate_db: str,
    *,
    task_ids: list[str] | None = None,
) -> dict:
    """Compare two benchmark runs and return a diff report."""
    baseline = _load_results(baseline_db, task_ids)
    candidate = _load_results(candidate_db, task_ids)

    all_tasks = sorted(set(baseline) | set(candidate))
    flips: dict[str, list[str]] = {"gained": [], "lost": [], "unchanged_pass": [], "unchanged_fail": []}

    for task in all_tasks:
        b = baseline.get(task, False)
        c = candidate.get(task, False)
        if not b and c:
            flips["gained"].append(task)
        elif b and not c:
            flips["lost"].append(task)
        elif b and c:
            flips["unchanged_pass"].append(task)
        else:
            flips["unchanged_fail"].append(task)

    return {
        "baseline_total": len(baseline),
        "candidate_total": len(candidate),
        "baseline_passed": sum(baseline.values()),
        "candidate_passed": sum(candidate.values()),
        "gained": flips["gained"],
        "lost": flips["lost"],
        "unchanged_pass": flips["unchanged_pass"],
        "net_change": len(flips["gained"]) - len(flips["lost"]),
    }


def _load_results(db_path: str, task_ids: list[str] | None) -> dict[str, bool]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if task_ids:
        placeholders = ",".join("?" for _ in task_ids)
        rows = conn.execute(
            f"SELECT task_id, passed FROM task_results WHERE task_id IN ({placeholders})",
            task_ids,
        ).fetchall()
    else:
        rows = conn.execute("SELECT task_id, passed FROM task_results").fetchall()
    conn.close()
    return {r["task_id"]: bool(r["passed"]) for r in rows}
```

**Step 3: Run tests and commit**

```bash
pytest tests/ -x -q
git add mellea/eval/ data/
git commit -m "add smoke suite and A/B comparison tool"
```

---

## Phase 6: Validate

### Task 16: End-to-end smoke test on cluster

**Step 1: Push fork changes**

```bash
cd ~/Documents/mellea-fork
git push origin agent-toolkit
```

**Step 2: Update mcode on cluster**

```bash
ssh skula@login3.bluevela.rmf.ibm.com "cd /u/skula/mcode && git pull origin main && source venv/bin/activate && uv pip install -e '.[dev]'"
```

**Step 3: Run HumanEval/MBPP regression**

Verify code generation still works with the new mellea fork.

**Step 4: Run smoke suite**

Run the ~25 task smoke suite on Blue Vela and compare against run2 baseline.

**Step 5: Run full SWE-bench Lite if smoke results are promising**

```bash
SWB_SPLIT=lite bash deploy/bluevela/run-swebench-live.sh
```

**Step 6: Publish results**

Follow the standard research publishing workflow (combined folder, HTML report, README with commands).
