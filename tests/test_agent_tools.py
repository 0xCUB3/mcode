from __future__ import annotations

import os
import subprocess

from mcode.agent.tools import get_diff, make_tools


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
    for i in range(30):
        (tmp_path / f"mod_{i}.py").write_text(f"target = {i}\n")
    tools = make_tools(str(tmp_path))
    result = tools["search_code"]("target")
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
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("a = 1\nb = 2\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)

    tools = make_tools(str(tmp_path))
    tools["apply_edit"]("foo.py", 2, 2, "b = 42\n")

    patch = get_diff(str(tmp_path))
    assert "-b = 2" in patch
    assert "+b = 42" in patch
