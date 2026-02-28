from __future__ import annotations

import json
from unittest.mock import MagicMock

from mcode.llm.session import LLMSession, build_file_tree


def test_build_file_tree_basic(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "bar.py").write_text("y = 2\n")
    (sub / "__init__.py").write_text("")

    tree = build_file_tree(str(tmp_path))
    lines = tree.strip().splitlines()
    paths = set(lines)
    assert "foo.py" in paths
    assert "pkg/bar.py" in paths or "pkg\\bar.py" in paths
    assert "pkg/__init__.py" in paths or "pkg\\__init__.py" in paths


def test_build_file_tree_excludes_non_source_dirs(tmp_path):
    (tmp_path / "good.py").write_text("")
    for d in [".git/hooks", "__pycache__", "build/lib/pkg", "dist", ".tox/py3", ".eggs"]:
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "junk.py").write_text("")

    tree = build_file_tree(str(tmp_path))
    assert "good.py" in tree
    for excluded in [".git", "__pycache__", "build", "dist", ".tox", ".eggs"]:
        assert excluded not in tree, f"{excluded} should be excluded"


def test_build_file_tree_truncates(tmp_path):
    for i in range(10):
        (tmp_path / f"mod_{i}.py").write_text("")

    tree = build_file_tree(str(tmp_path), max_files=5)
    lines = tree.strip().splitlines()
    assert any("more files" in line for line in lines)


def test_build_file_tree_empty(tmp_path):
    tree = build_file_tree(str(tmp_path))
    assert tree == ""


def test_localize_files_returns_paths():
    session = LLMSession(model_id="test", backend_name="ollama")

    mock_result = MagicMock()
    mock_result.value = json.dumps({"files": ["src/foo.py", "src/bar.py"]})

    mock_m = MagicMock()
    mock_m.instruct.return_value = mock_result
    session._m = mock_m

    result = session.localize_files(
        file_tree="src/foo.py\nsrc/bar.py\nsrc/baz.py",
        problem_statement="Fix the bug in foo",
    )
    assert result == ["src/foo.py", "src/bar.py"]
    mock_m.instruct.assert_called_once()


def test_localize_files_handles_invalid_json():
    session = LLMSession(model_id="test", backend_name="ollama")

    mock_result = MagicMock()
    mock_result.value = "not valid json"

    mock_m = MagicMock()
    mock_m.instruct.return_value = mock_result
    session._m = mock_m

    result = session.localize_files(
        file_tree="src/foo.py",
        problem_statement="Fix the bug",
    )
    assert result == []


def test_localize_files_handles_empty_files():
    session = LLMSession(model_id="test", backend_name="ollama")

    mock_result = MagicMock()
    mock_result.value = json.dumps({"files": []})

    mock_m = MagicMock()
    mock_m.instruct.return_value = mock_result
    session._m = mock_m

    result = session.localize_files(
        file_tree="src/foo.py",
        problem_statement="Fix the bug",
    )
    assert result == []
