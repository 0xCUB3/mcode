"""Top-level mcode doctor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_doctor_local_vllm_reports_checks(monkeypatch):
    monkeypatch.setattr("mcode.launch.config.load", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "mcode.launch.local_vllm.doctor",
        lambda _cfg: [Check(name="local-vllm", ok=True, detail="ok")],
    )
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "local-vllm"])
    assert res.exit_code == 0
    assert "local-vllm" in res.stdout
    assert "✓" in res.stdout


def test_doctor_init_writes_config(tmp_path: Path, monkeypatch):
    written: dict[str, Path] = {}

    def fake_init(*, login, cfg_path=None, **_):
        p = tmp_path / "launch.toml"
        p.write_text("[bluevela]\nlogin = '" + login + "'\n")
        written["path"] = p
        return p

    monkeypatch.setattr("mcode.launch.bluevela.doctor_init", fake_init)
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "bluevela", "--init", "--login", "alice@host"])
    assert res.exit_code == 0
    assert "wrote" in res.stdout
    assert written["path"].exists()


def test_doctor_init_rejects_non_bluevela_target():
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "local-vllm", "--init"])
    assert res.exit_code == 1
    assert "only supported for `bluevela`" in res.output
