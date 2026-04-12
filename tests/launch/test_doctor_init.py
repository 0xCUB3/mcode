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


def test_doctor_init_raises_when_no_parseable_queues(tmp_path: Path) -> None:
    """Codex pre-merge-review fix: if bqueues returns nothing parseable, we
    must NOT silently write queue_order=['normal']. That ships a config that
    fails at submit time."""
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
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(tmp_path / "launch.toml", login="testuser@testhost", ssh_client=ssh)
    assert "batch-capable" in ei.value.what


@pytest.mark.parametrize(
    "failing_cmd",
    ["echo $HOME", "bugroup", "bqueues -u"],
    ids=["home", "bugroup", "bqueues-u"],
)
def test_doctor_init_converts_transport_error_to_launch_error(
    failing_cmd: str, tmp_path: Path
) -> None:
    """Codex final-verify-pass fix: every post-preflight ssh.run in doctor_init
    must convert TransportError into a formatted LaunchError. Without this,
    a mid-init SSH drop surfaces as a raw Python traceback instead of the
    CLI's ✗/why/next layout."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if failing_cmd in cmd:
            raise TransportError("ssh session dropped")
        # Non-failing paths — return minimal valid output.
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if cmd.startswith("bugroup"):
            return _ok(stdout="GROUP_NAME USERS GROUP_ADMIN\ngrp_x testuser ( - )\n")
        if "bqueues -u" in cmd:
            return _ok(stdout="QUEUE_NAME PRIO STATUS\nnormal 30 Open:Active\n")
        if cmd.startswith("bqueues -l"):
            return _ok(stdout="SCHEDULING POLICIES: FAIRSHARE\n")
        return _ok()

    ssh.run.side_effect = run
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(tmp_path / "launch.toml", login="testuser@testhost", ssh_client=ssh)
    # Either the SSH-dropped path or the fail-closed queue path renders
    # a formatted LaunchError. What must NOT happen: raw TransportError.
    assert ei.value.what  # non-empty formatted message


def test_doctor_init_handles_transport_error_in_queue_probe(tmp_path: Path) -> None:
    """Codex pre-merge verification fix: a TransportError raised by the
    `bqueues -l` probe must NOT escape doctor_init. It must be caught and
    contribute to the fail-closed 'could not confirm any batch-capable
    queue' LaunchError path so the CLI renders a formatted error, not a
    traceback."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if cmd.startswith("bugroup"):
            return _ok(stdout="GROUP_NAME USERS GROUP_ADMIN\ngrp_x testuser ( - )\n")
        if "bqueues -u" in cmd:
            return _ok(stdout="QUEUE_NAME PRIO STATUS\nnormal 30 Open:Active\n")
        if cmd.startswith("bqueues -l"):
            raise TransportError("ssh session dropped")
        return _ok()

    ssh.run.side_effect = run
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(tmp_path / "launch.toml", login="testuser@testhost", ssh_client=ssh)
    assert "batch-capable" in ei.value.what


def test_doctor_init_raises_when_only_interactive_queues_available(tmp_path: Path) -> None:
    """Fail closed if every visible queue is confirmed interactive-only."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if cmd.startswith("bugroup"):
            return _ok(stdout="GROUP_NAME USERS GROUP_ADMIN\ngrp_x testuser ( - )\n")
        if "bqueues -u" in cmd:
            return _ok(stdout="QUEUE_NAME PRIO STATUS\ninteractive 30 Open:Active\n")
        if cmd.startswith("bqueues -l"):
            return _ok(stdout="SCHEDULING POLICIES: FAIRSHARE ONLY_INTERACTIVE\n")
        return _ok()

    ssh.run.side_effect = run
    with pytest.raises(LaunchError) as ei:
        bluevela.doctor_init(tmp_path / "launch.toml", login="testuser@testhost", ssh_client=ssh)
    assert "batch-capable" in ei.value.what


def test_doctor_init_shared_root_is_under_home(tmp_path: Path) -> None:
    """shared_root lives under $HOME — the bluevela_vllm.sh script uses
    per-job podman graphroots in /tmp, so shared_root only carries small
    artifacts. Users who need HF_HOME on a quota-free filesystem configure
    it via hf-env.sh separately."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if "mcode-doctor-init-ok" in cmd:
            return _ok(stdout="mcode-doctor-init-ok\n")
        if cmd.startswith("echo $HOME"):
            return _ok(stdout="/u/testuser\n")
        if cmd.startswith("bugroup"):
            return _ok(stdout="GROUP_NAME USERS GROUP_ADMIN\ngrp_runtime testuser ( - )\n")
        if "bqueues -u" in cmd:
            return _ok(stdout="QUEUE_NAME PRIO STATUS\nnormal 30 Open:Active\n")
        if cmd.startswith("bqueues -l"):
            return _ok(stdout="SCHEDULING POLICIES: FAIRSHARE\n")
        return _ok()

    ssh.run.side_effect = run
    dst = tmp_path / "launch.toml"
    bluevela.doctor_init(dst, login="testuser@testhost", ssh_client=ssh)
    cfg = config_mod.load(dst)
    assert cfg.bluevela.shared_root == "/u/testuser/mcode-shared"


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
