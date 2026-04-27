from __future__ import annotations

from pathlib import Path

from mcode.agent.workspace_context import collect_workspace_context


def test_collect_workspace_context_includes_authoritative_docs(tmp_path: Path) -> None:
    docs = tmp_path / ".docs"
    docs.mkdir()
    (docs / "instructions.md").write_text(
        "# Allergies\n\nImplement score membership using bit values.\nEggs is 1 and peanuts is 2.\n"
    )
    (tmp_path / "README.md").write_text("# Exercise\n\nUse the existing tests as verifier.\n")

    context = collect_workspace_context(str(tmp_path), "implement allergies score")

    assert [entry.path for entry in context.entries] == [
        ".docs/instructions.md",
        "README.md",
    ]
    assert "Local workspace context:" in context.text
    assert ".docs/instructions.md (task instructions)" in context.text
    assert "Implement score membership" in context.text


def test_collect_workspace_context_returns_empty_without_docs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")

    context = collect_workspace_context(str(tmp_path), "fix hello")

    assert context.entries == ()
    assert context.text == ""


def test_collect_workspace_context_caps_large_docs(tmp_path: Path) -> None:
    (tmp_path / "SPEC.md").write_text("\n".join(f"line {i}" for i in range(200)))

    context = collect_workspace_context(str(tmp_path), "line", max_chars=300)

    assert len(context.text) < 700
    assert "line 0" in context.text
