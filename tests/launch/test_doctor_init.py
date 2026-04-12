from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcode.launch import bluevela
from mcode.launch import config as config_mod
from mcode.launch.models import LaunchError
from mcode.launch.progress import TransportError
from mcode.launch.ssh import SshResult


def _ok(stdout: str = "", stderr: str = "") -> SshResult:
    return SshResult(returncode=0, stdout=stdout, stderr=stderr, duration_s=0.01)


def test_parse_bugroup_filters_by_user() -> None:
    """Codex/live-probe fix: bugroup (no args) lists ALL groups. Must filter
    to rows that actually contain the user, otherwise the first row wins
    regardless of membership (e.g. 'lsfadmins' swept in incorrectly)."""
    raw = (
        "GROUP_NAME    USERS                     GROUP_ADMIN\n"
        "lsfadmins     bmbelgod lsfadmin jcolino ( - )\n"
        "grp_runtime   alice bob skula carol     (admin)\n"
        "grp_models    xdang issei skula         ( admin )\n"
        "grp_weird!    skula                     -\n"  # invalid name -> skipped
    )
    # With user filter, lsfadmins excluded; skula is in runtime+models.
    assert bluevela._parse_bugroup(raw, user="skula") == ["grp_runtime", "grp_models"]
    # Without user filter, returns every well-formed row (internal use).
    assert bluevela._parse_bugroup(raw) == ["lsfadmins", "grp_runtime", "grp_models"]


def test_parse_bugroup_rejects_substring_match() -> None:
    """Whole-word member matching: `skula` must NOT match `skulapp`."""
    raw = "grp_x  skulapp other  ( - )\n"
    assert bluevela._parse_bugroup(raw, user="skula") == []


def test_parse_bqueues_orders_open_queues_by_priority() -> None:
    raw = (
        "QUEUE_NAME PRIO STATUS\n"
        "night       40   Closed:Inact\n"
        "normal      30   Open:Active\n"
        "preempt     20   Open:Active\n"
        "owners      43   Closed:Inact\n"
    )
    rows = bluevela._parse_bqueues(raw)
    # Open queues first, sorted by prio desc.
    open_names = [name for name, _, status in rows if status.startswith("Open")]
    assert open_names == ["normal", "preempt"]


def test_doctor_init_preflight_failure_raises_actionable_error() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(login="alice@host", ssh_client=ssh)
    assert "preflight" in ei.value.what.lower()
    assert ei.value.next  # actionable hint


def test_doctor_init_rejects_login_without_at_sign() -> None:
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(login="nohost", ssh_client=MagicMock())
    assert "user@host" in ei.value.what


def test_doctor_init_writes_config_with_probed_values(tmp_path: Path) -> None:
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if "bugroup" in cmd:
            return _ok(stdout="GROUP_NAME USERS GROUP_ADMIN\ngrp_runtime testuser (admin)\n")
        if "bqueues" in cmd:
            return _ok(
                stdout="QUEUE_NAME PRIO STATUS\nnormal 30 Open:Active\npreempt 20 Open:Active\n"
            )
        return _ok()

    ssh.run.side_effect = run
    dst = tmp_path / "launch.toml"
    written = bluevela.doctor_init(dst, login="testuser@testhost", ssh_client=ssh)
    assert written == dst
    cfg = config_mod.load(dst)
    assert cfg.bluevela.login == "testuser@testhost"
    assert cfg.bluevela.workspace_root == "/u/testuser/mcode-launch"
    assert cfg.bluevela.shared_root == "/u/testuser/mcode-shared"
    # Filtered: lsfadmins excluded (testuser not a member), grp_runtime picked.
    assert cfg.bluevela.group == "grp_runtime"
    # Filtered: `interactive` queue dropped via ONLY_INTERACTIVE policy check.
    assert cfg.bluevela.queue_order == ["normal", "preempt"]
    assert cfg.bluevela.gpu_mode == "exclusive_process"


def test_doctor_init_falls_back_to_normal_when_no_queues_parsed(tmp_path: Path) -> None:
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if "bugroup" in cmd:
            return _ok(stdout="")
        if "bqueues" in cmd:
            return _ok(stdout="")  # parse failure / cluster quirks
        return _ok()

    ssh.run.side_effect = run
    dst = tmp_path / "launch.toml"
    bluevela.doctor_init(dst, login="testuser@testhost", ssh_client=ssh)
    cfg = config_mod.load(dst)
    assert cfg.bluevela.queue_order == ["normal"]
    assert cfg.bluevela.group == ""  # not fabricated


def test_doctor_init_rejects_weird_home_path(tmp_path: Path) -> None:
    """If $HOME contains characters that can't safely round-trip through TOML,
    hard-fail instead of writing a bad config."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/alice; rm -rf /\n")  # malicious
        return _ok()

    ssh.run.side_effect = run
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(tmp_path / "x.toml", login="a@h", ssh_client=ssh)
    assert "$HOME" in ei.value.what or "unexpected" in ei.value.what.lower()
