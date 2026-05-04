from __future__ import annotations

from pathlib import Path

from mcode.agent import tooling


def test_str_replace_edit_skips_cpp_syntax_guard(tmp_path: Path, monkeypatch) -> None:
    header = tmp_path / "sample.h"
    header.write_text("#pragma once\n\nnamespace sample {\n\n}  // namespace sample\n")

    calls: list[str] = []

    def fake_syntax_details(path: str, content: str):
        del content
        calls.append(path)
        return (0, None)

    monkeypatch.setattr(tooling, "_syntax_details", fake_syntax_details)

    result = tooling.str_replace_edit(
        str(header),
        "namespace sample {\n\n}  // namespace sample",
        "namespace sample {\nclass Thing;\n}  // namespace sample",
        repo_root=str(tmp_path),
    )

    assert "APPLIED" in result
    assert calls == []
    assert "class Thing;" in header.read_text()


def test_str_replace_edit_rejects_rust_unclosed_delimiters(tmp_path: Path) -> None:
    source = tmp_path / "lib.rs"
    source.write_text("fn value() -> i32 { 1 }\n")

    result = tooling.str_replace_edit(
        str(source),
        "fn value() -> i32 { 1 }\n",
        "fn value() -> i32 { (1 }\n",
        repo_root=str(tmp_path),
    )

    assert "REJECTED" in result
    assert "unclosed" in result or "unexpected" in result
    assert source.read_text() == "fn value() -> i32 { 1 }\n"


def test_str_replace_edit_rejects_python_ast_syntax_errors(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("value = (1)\n")

    result = tooling.str_replace_edit(
        str(source),
        "value = (1)\n",
        "value = (1\n",
        repo_root=str(tmp_path),
    )

    assert "REJECTED" in result
    assert "Syntax error" in result
    assert source.read_text() == "value = (1)\n"


def test_str_replace_edit_keeps_python_syntax_guard(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def value():\n    return 1\n")

    calls: list[str] = []

    def fake_syntax_details(path: str, content: str):
        calls.append(path)
        if "return 2" in content:
            return (1, "Syntax error at line 2, column 4")
        return (0, None)

    monkeypatch.setattr(tooling, "_syntax_details", fake_syntax_details)

    result = tooling.str_replace_edit(
        str(source),
        "def value():\n    return 1\n",
        "def value():\n    return 2\n",
        repo_root=str(tmp_path),
    )

    assert "REJECTED" in result
    assert len(calls) == 2
    assert source.read_text() == "def value():\n    return 1\n"
