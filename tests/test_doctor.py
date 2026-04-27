"""Top-level mcode doctor."""

from __future__ import annotations

from typer.testing import CliRunner

from mcode.cli import app
from mcode.doctor import render_check_lines, system_checks
from mcode.launch.models import Check


def test_system_checks_includes_results_dir():
    checks = system_checks()
    names = [c.name for c in checks]
    assert any("results dir writable" in n for n in names)
    assert any("container runtime" in n for n in names)
    assert any("mellea importable" in n for n in names)
    assert any("ruff" in n for n in names)


def test_render_check_lines_marks_failures():
    checks = [
        Check(name="ok-thing", ok=True, detail="all good"),
        Check(name="bad-thing", ok=False, detail="broken", next="fix it"),
    ]
    lines, any_failed = render_check_lines(checks)
    text = "\n".join(lines)
    assert "ok-thing" in text
    assert "bad-thing" in text
    assert "all good" in text
    assert "next: fix it" in text
    assert any_failed is True


def test_render_check_lines_clean():
    checks = [Check(name="a", ok=True), Check(name="b", ok=True)]
    _, any_failed = render_check_lines(checks)
    assert any_failed is False


def test_doctor_unknown_target_exits_nonzero():
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "no-such-target"])
    assert res.exit_code == 1
