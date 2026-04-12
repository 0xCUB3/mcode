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


def _fake_subprocess(rsync_rc: int = 0, marker_present: bool = True, ssh_rc: int = 0):
    """Mock subprocess.run handling the 3 kinds of calls cmd_sync makes:
    git rev-parse, ssh marker probe, ssh marker touch, rsync."""
    seen = {}

    def fake(argv, *args, **kwargs):
        if argv and argv[0] == "rsync":
            seen["rsync_argv"] = argv
            return type("R", (), {"returncode": rsync_rc})()
        if argv and argv[0] == "ssh":
            seen.setdefault("ssh_calls", []).append(argv)
            stdout = "yes\n" if marker_present else "no\n"
            return type("R", (), {"returncode": ssh_rc, "stdout": stdout, "stderr": ""})()
        # git rev-parse: fail so fallback fires (test uses --src explicitly)
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    return fake, seen


def test_sync_invokes_rsync_to_correct_destination(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    (src / "README.md").write_text("hi")

    fake, seen = _fake_subprocess()
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 0, result.output
    argv = seen.get("rsync_argv")
    assert argv is not None, "rsync was not invoked"
    assert argv[0] == "rsync"
    assert "-az" in argv and "--delete" in argv
    assert any(a.startswith("--exclude=.git/") for a in argv)
    assert f"{str(src)}/" in argv
    assert "alice@host:/u/alice/mcode/" in argv
    # Codex review fix: rsync must use SSH with safety options (BatchMode etc).
    assert "-e" in argv
    ssh_cmd = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in ssh_cmd
    assert "ConnectTimeout=10" in ssh_cmd


def test_sync_fails_closed_when_not_in_git_repo(
    runner: CliRunner, cfg_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """Codex review fix: refuse to sync when git rev-parse fails — the old
    cwd fallback combined with --delete could wipe an unrelated remote dir."""
    # Make git rev-parse fail; also don't pass --src.
    fake, _ = _fake_subprocess()
    with patch("subprocess.run", side_effect=fake):
        # Invoke from a directory guaranteed not to be a git repo.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["sync", "bluevela"])
    assert result.exit_code == 1
    assert "cannot determine source repo" in result.output


def test_sync_creates_marker_on_first_run(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    """Codex review fix: on first sync, the launcher writes a marker file
    so subsequent --delete runs have a safety net."""
    src = tmp_path / "repo"
    src.mkdir()

    fake, seen = _fake_subprocess(marker_present=False)
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 0, result.output
    # At least two ssh calls: probe + touch marker.
    ssh_calls = seen.get("ssh_calls", [])
    assert len(ssh_calls) >= 2
    assert any(
        "touch" in " ".join(c) and ".mcode-launch-workspace" in " ".join(c) for c in ssh_calls
    )


def test_sync_dry_run_adds_flag_v2(runner: CliRunner, cfg_path: Path, tmp_path: Path) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    fake, seen = _fake_subprocess()
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--dry-run", "--src", str(src)])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in seen["rsync_argv"]


def test_sync_surfaces_rsync_failure_as_launch_error_v2(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    fake, _ = _fake_subprocess(rsync_rc=23)
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 23
    assert "rsync exited 23" in result.output
