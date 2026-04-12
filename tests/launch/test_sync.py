from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mcode.launch.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cfg_path(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "launch.toml"
    p.write_text(
        "[bluevela]\n"
        'login = "alice@host"\n'
        'workspace_root = "/u/alice/mcode"\n'
        'shared_root = "/u/alice/mcode-shared"\n'
        'group = "grp_x"\n'
        'queue_order = ["normal"]\n'
    )
    monkeypatch.setenv("MCODE_LAUNCH_CONFIG", str(p))
    return p


def test_sync_rejects_non_bluevela_target(runner: CliRunner) -> None:
    result = runner.invoke(app, ["sync", "local-vllm"])
    assert result.exit_code == 1
    assert "only supports target=bluevela" in result.output


def test_sync_requires_config(runner: CliRunner, tmp_path: Path, monkeypatch) -> None:
    empty_cfg = tmp_path / "empty.toml"
    empty_cfg.write_text("")
    monkeypatch.setenv("MCODE_LAUNCH_CONFIG", str(empty_cfg))
    result = runner.invoke(app, ["sync", "bluevela"])
    assert result.exit_code == 1
    assert "config incomplete" in result.output


def test_sync_invokes_rsync_to_correct_destination(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    (src / "README.md").write_text("hi")

    seen = {}

    def fake_run(argv, *args, **kwargs):
        if argv and argv[0] == "rsync":
            seen["argv"] = argv
            return type("R", (), {"returncode": 0})()
        # git rev-parse: simulate failure so fallback to --src path kicks in.
        return type("R", (), {"returncode": 1, "stdout": ""})()

    with patch(
        "mcode.launch.cli.subprocess.run" if False else "subprocess.run", side_effect=fake_run
    ):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 0, result.output
    argv = seen.get("argv")
    assert argv is not None, "rsync was not invoked"
    assert argv[0] == "rsync"
    assert "-az" in argv and "--delete" in argv
    assert any(a.startswith("--exclude=.git/") for a in argv)
    assert f"{str(src)}/" in argv
    assert "alice@host:/u/alice/mcode/" in argv


def test_sync_dry_run_adds_flag(runner: CliRunner, cfg_path: Path, tmp_path: Path) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    seen = {}

    def fake_run(argv, *args, **kwargs):
        if argv and argv[0] == "rsync":
            seen["argv"] = argv
            return type("R", (), {"returncode": 0})()
        return type("R", (), {"returncode": 1, "stdout": ""})()

    with patch("subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["sync", "bluevela", "--dry-run", "--src", str(src)])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in seen["argv"]


def test_sync_surfaces_rsync_failure_as_launch_error(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()

    def fake_run(argv, *args, **kwargs):
        if argv and argv[0] == "rsync":
            return type("R", (), {"returncode": 23})()
        return type("R", (), {"returncode": 1, "stdout": ""})()

    with patch("subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 23
    assert "rsync exited 23" in result.output
