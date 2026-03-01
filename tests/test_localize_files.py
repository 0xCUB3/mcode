from __future__ import annotations

from unittest.mock import MagicMock

from mcode.context.localize import (
    build_indented_tree,
    collect_source_files,
    localize,
    rank_bm25,
)


def test_collect_source_files_basic(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "bar.py").write_text("y = 2\n")

    files = collect_source_files(str(tmp_path))
    assert "foo.py" in files
    assert "pkg/bar.py" in files


def test_collect_source_files_excludes_noise(tmp_path):
    (tmp_path / "good.py").write_text("")
    noise_dirs = [
        ".git/hooks",
        "__pycache__",
        "build/lib/pkg",
        "dist",
        ".tox/py3",
        "doc/data",
        "docs/api",
        "tests/unit",
        "test/functional",
    ]
    for d in noise_dirs:
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "junk.py").write_text("")

    files = collect_source_files(str(tmp_path))
    assert files == ["good.py"]


def test_build_indented_tree_basic():
    paths = [
        "pylint/checkers/base_checker.py",
        "pylint/checkers/base/name_checker/checker.py",
        "pylint/lint/pylinter.py",
    ]
    tree = build_indented_tree(paths)
    assert "pylint/" in tree
    assert "    checkers/" in tree
    assert "        base_checker.py" in tree
    assert "        base/" in tree
    assert "            name_checker/" in tree
    assert "                checker.py" in tree
    assert "    lint/" in tree
    assert "        pylinter.py" in tree


def test_build_indented_tree_dirs_before_files():
    paths = ["a/z.py", "a/b/c.py"]
    tree = build_indented_tree(paths)
    lines = tree.splitlines()
    # b/ directory should come before z.py file
    b_idx = next(i for i, line in enumerate(lines) if "b/" in line)
    z_idx = next(i for i, line in enumerate(lines) if "z.py" in line)
    assert b_idx < z_idx


def test_rank_bm25_basic(tmp_path):
    (tmp_path / "name_checker.py").write_text(
        "class NameChecker:\n    def check_name(self, node):\n        pass\n"
    )
    (tmp_path / "base_checker.py").write_text(
        "class BaseChecker:\n    def run(self):\n        pass\n"
    )
    (tmp_path / "pylinter.py").write_text("class PyLinter:\n    def check(self):\n        pass\n")

    paths = ["name_checker.py", "base_checker.py", "pylinter.py"]
    ranked = rank_bm25(paths, "name checker naming convention", str(tmp_path))
    # name_checker.py should rank first (has "name" and "checker" in content)
    assert ranked[0] == "name_checker.py"


def test_rank_bm25_empty():
    assert rank_bm25([], "query", "/nonexistent") == []


def test_rank_bm25_respects_top_n(tmp_path):
    for i in range(10):
        (tmp_path / f"mod_{i}.py").write_text(f"x = {i}\n")
    paths = [f"mod_{i}.py" for i in range(10)]
    ranked = rank_bm25(paths, "module", str(tmp_path), top_n=3)
    assert len(ranked) == 3


def test_localize_returns_files_and_hints(tmp_path):
    (tmp_path / "foo.py").write_text("def foo():\n    pass\n")
    (tmp_path / "bar.py").write_text("def bar():\n    pass\n")

    files, hints = localize(str(tmp_path), "Fix the foo function")
    assert "foo.py" in files
    assert "--- foo.py ---" in hints
    assert "def foo():" in hints


def test_localize_includes_multiple_files(tmp_path):
    (tmp_path / "foo.py").write_text("def foo():\n    pass\n")
    (tmp_path / "bar.py").write_text("def bar():\n    pass\n")

    files, hints = localize(str(tmp_path), "Fix foo and bar")
    assert len(files) == 2


def test_localize_empty_repo(tmp_path):
    files, hints = localize(str(tmp_path), "Fix something")
    assert files == []
    assert hints == ""


def test_localize_with_llm_session(tmp_path):
    """LLM session narrows BM25 candidates to a subset."""
    (tmp_path / "checker.py").write_text("class NameChecker:\n    pass\n")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / "config.py").write_text("CONFIG = {}\n")

    mock_session = MagicMock()
    mock_session.localize_files.return_value = ["checker.py"]

    files, hints = localize(
        str(tmp_path),
        "Fix the name checker",
        llm_session=mock_session,
    )

    mock_session.localize_files.assert_called_once()
    assert files == ["checker.py"]
    assert "--- checker.py ---" in hints
    # utils.py should not be included since LLM narrowed to checker.py
    assert "--- utils.py ---" not in hints


def test_localize_with_llm_session_fallback(tmp_path):
    """When LLM returns files not on disk, they are skipped."""
    (tmp_path / "real.py").write_text("x = 1\n")

    mock_session = MagicMock()
    mock_session.localize_files.return_value = ["nonexistent.py", "real.py"]

    files, hints = localize(
        str(tmp_path),
        "Fix something",
        llm_session=mock_session,
    )

    assert files == ["real.py"]
