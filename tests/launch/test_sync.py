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


def _fake_subprocess(
    rsync_rc: int = 0,
    remote_state: str = "marker",
    ssh_rc: int = 0,
    deps_rc: int = 0,
    deps_stderr: str = "remote uv sync failed",
):
    """Mock subprocess.run for cmd_sync. remote_state drives the probe response:
    - "marker": launcher-owned workspace; rsync proceeds
    - "empty": first sync, marker gets created; rsync proceeds
    - "populated": non-empty unmarked dir; rsync refused without --bootstrap
    """
    seen = {}

    def fake(argv, *args, **kwargs):
        if argv and argv[0] == "rsync":
            seen["rsync_argv"] = argv
            return type("R", (), {"returncode": rsync_rc, "stdout": "", "stderr": ""})()
        if argv and argv[0] == "ssh":
            seen.setdefault("ssh_calls", []).append(argv)
            remote = argv[-1]
            if "uv sync" in remote:
                return type(
                    "R",
                    (),
                    {
                        "returncode": deps_rc,
                        "stdout": "",
                        "stderr": deps_stderr if deps_rc else "",
                    },
                )()
            # The probe command contains the state-detection logic. The touch
            # command is a follow-up — return empty stdout for that.
            if "elif" in remote:  # the 3-state probe
                return type(
                    "R", (), {"returncode": ssh_rc, "stdout": f"{remote_state}\n", "stderr": ""}
                )()
            return type("R", (), {"returncode": ssh_rc, "stdout": "", "stderr": ""})()
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
    assert any(a.startswith("--exclude=benchmarks/") for a in argv)
    assert any(a.startswith("--exclude=.uv-cache/") for a in argv)
    assert any(a.startswith("--exclude=.bluevela-reruns/") for a in argv)
    assert any(a.startswith("--exclude=research/") for a in argv)
    assert any(a.startswith("--exclude=experiments/") for a in argv)
    assert any(a.startswith("--exclude=podman-tmp/") for a in argv)
    assert f"{str(src)}/" in argv
    assert "alice@host:/u/alice/mcode/" in argv
    # Codex review fix: rsync must use SSH with safety options (BatchMode etc).
    assert "-e" in argv
    ssh_cmd = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in ssh_cmd
    assert "ConnectTimeout=10" in ssh_cmd
    ssh_calls = seen.get("ssh_calls", [])
    assert any(
        "uv sync --extra dev --extra swebench --extra datasets --extra observability" in c[-1]
        for c in ssh_calls
    )


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


def test_sync_creates_marker_on_empty_remote(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    """First sync into an EMPTY remote workspace is safe: create the marker
    and proceed."""
    src = tmp_path / "repo"
    src.mkdir()

    fake, seen = _fake_subprocess(remote_state="empty")
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 0, result.output
    ssh_calls = seen.get("ssh_calls", [])
    assert any(
        "touch" in " ".join(c) and ".mcode-launch-workspace" in " ".join(c) for c in ssh_calls
    )
    assert "rsync_argv" in seen, "rsync should have run after marker creation"


def test_sync_refuses_populated_unmarked_remote(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    """Codex verify-pass fix: `rsync --delete` into a non-empty remote dir
    without our marker could wipe unrelated data. Must refuse unless
    --bootstrap is passed."""
    src = tmp_path / "repo"
    src.mkdir()

    fake, seen = _fake_subprocess(remote_state="populated")
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 1
    assert "non-empty" in result.output and "marker" in result.output
    assert "rsync_argv" not in seen, "rsync MUST NOT run when destination is unvetted"


def test_sync_bootstrap_allows_populated_remote(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    """--bootstrap is the explicit opt-in for populated unmarked remote."""
    src = tmp_path / "repo"
    src.mkdir()

    fake, seen = _fake_subprocess(remote_state="populated")
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--bootstrap", "--src", str(src)])
    assert result.exit_code == 0, result.output
    assert "rsync_argv" in seen
    assert "⚠" in result.output or "bootstrap" in result.output.lower()


def test_sync_dry_run_adds_flag_v2(runner: CliRunner, cfg_path: Path, tmp_path: Path) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    fake, seen = _fake_subprocess()
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--dry-run", "--src", str(src)])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in seen["rsync_argv"]


def test_sync_dry_run_skips_remote_dependency_refresh(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    fake, seen = _fake_subprocess()
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--dry-run", "--src", str(src)])
    assert result.exit_code == 0, result.output
    ssh_calls = seen.get("ssh_calls", [])
    assert not any("uv sync" in call[-1] for call in ssh_calls)


def test_sync_surfaces_remote_dependency_failure(
    runner: CliRunner, cfg_path: Path, tmp_path: Path
) -> None:
    src = tmp_path / "repo"
    src.mkdir()
    fake, _ = _fake_subprocess(deps_rc=1, deps_stderr="uv sync boom")
    with patch("subprocess.run", side_effect=fake):
        result = runner.invoke(app, ["sync", "bluevela", "--src", str(src)])
    assert result.exit_code == 1
    assert "remote dependency sync failed" in result.output
    assert "uv sync boom" in result.output


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
