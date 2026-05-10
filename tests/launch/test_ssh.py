from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mcode.launch.progress import TransportError
from mcode.launch.ssh import (
    DEFAULT_SSH_OPTIONS,
    SshClient,
    SshError,
    SshResult,
    _is_transport_failure,
    _transport_message,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


# --- _is_transport_failure classifier ---------------------------------------
def test_transport_classifier_exit_255_known_marker() -> None:
    r = _completed(returncode=255, stderr="ssh: Connection timed out")
    assert _is_transport_failure(r) is True


def test_transport_classifier_exit_255_unknown_marker_is_remote_failure() -> None:
    # A remote command can legitimately exit 255 with no SSH-client marker.
    # Classifier must NOT treat this as transport (regression).
    r = _completed(returncode=255, stderr="remote: something weird")
    assert _is_transport_failure(r) is False


def test_transport_classifier_remote_nonzero_is_not_transport() -> None:
    # Remote command returning non-zero: transport is fine.
    r = _completed(returncode=1, stderr="bsub: queue closed")
    assert _is_transport_failure(r) is False


def test_remote_exit_255_returns_failed_result_not_transport_error() -> None:
    """End-to-end regression: remote exit 255 without an SSH
    transport marker must surface as SshResult(ok=False), not TransportError."""
    from unittest.mock import patch

    r = _completed(returncode=255, stderr="my-remote-tool: fatal error")
    with patch("mcode.launch.ssh.subprocess.run", return_value=r):
        c = SshClient("user@host")
        result = c.run("weird-tool")
    assert not result.ok
    assert result.returncode == 255
    assert "fatal error" in result.stderr


def test_transport_classifier_timeout_is_transport() -> None:
    t = subprocess.TimeoutExpired(cmd="ssh", timeout=10.0)
    assert _is_transport_failure(t) is True


def test_transport_message_extracts_marker() -> None:
    r = _completed(
        returncode=255,
        stderr="debug1: ssh something\nssh: connect to host x port 22: Connection refused",
    )
    assert "Connection refused" in _transport_message(r)


# --- SshClient.run() --------------------------------------------------------
@patch("mcode.launch.ssh.subprocess.run")
def test_run_returns_result_on_success(mock_run) -> None:
    mock_run.return_value = _completed(returncode=0, stdout="hello", stderr="")
    c = SshClient("user@host")
    result = c.run("echo hello")
    assert result.ok
    assert result.stdout == "hello"
    assert result.returncode == 0
    # Verify we invoked ssh with the expected options.
    args, _ = mock_run.call_args
    argv = args[0]
    assert argv[0] == "ssh"
    for opt in DEFAULT_SSH_OPTIONS:
        assert opt in argv
    assert argv[-2] == "user@host"
    assert argv[-1] == "echo hello"


@patch("mcode.launch.ssh.subprocess.run")
def test_run_remote_nonzero_returns_failed_result_not_transport_error(mock_run) -> None:
    mock_run.return_value = _completed(returncode=2, stdout="", stderr="queue closed")
    c = SshClient("user@host")
    result = c.run("bsub ...")
    assert not result.ok
    assert result.returncode == 2
    assert "queue closed" in result.stderr


@patch("mcode.launch.ssh.subprocess.run")
def test_run_transport_failure_raises(mock_run) -> None:
    mock_run.return_value = _completed(
        returncode=255,
        stderr="ssh: connect to host x port 22: Connection refused",
    )
    c = SshClient("user@host")
    with pytest.raises(TransportError) as ei:
        c.run("echo hi")
    assert "Connection refused" in str(ei.value)


@patch("mcode.launch.ssh.subprocess.run")
def test_run_timeout_raises_transport_error(mock_run) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10.0)
    c = SshClient("user@host")
    with pytest.raises(TransportError) as ei:
        c.run("echo hi", timeout=10.0)
    assert "timeout" in str(ei.value).lower()


@patch("mcode.launch.ssh.subprocess.run")
def test_raise_for_status_on_remote_failure(mock_run) -> None:
    mock_run.return_value = _completed(returncode=5, stderr="nope")
    c = SshClient("user@host")
    with pytest.raises(SshError):
        c.run("bsub").raise_for_status()


# --- scp upload/download ----------------------------------------------------
@patch("mcode.launch.ssh.subprocess.run")
def test_upload_invokes_scp(mock_run, tmp_path) -> None:
    mock_run.return_value = _completed(returncode=0)
    src = tmp_path / "env.json"
    src.write_text("{}")
    c = SshClient("user@host")
    c.upload(src, "/remote/env.json")
    argv = mock_run.call_args[0][0]
    assert argv[0] == "scp"
    assert str(src) in argv
    assert "user@host:/remote/env.json" in argv


@patch("mcode.launch.ssh.subprocess.run")
def test_upload_transport_failure_raises(mock_run, tmp_path) -> None:
    mock_run.return_value = _completed(returncode=255, stderr="ssh: No route to host")
    src = tmp_path / "x"
    src.write_text("")
    c = SshClient("user@host")
    with pytest.raises(TransportError):
        c.upload(src, "/remote/x")


@patch("mcode.launch.ssh.subprocess.run")
def test_upload_remote_error_raises_ssh_error(mock_run, tmp_path) -> None:
    mock_run.return_value = _completed(returncode=1, stderr="scp: permission denied")
    src = tmp_path / "x"
    src.write_text("")
    c = SshClient("user@host")
    with pytest.raises(SshError):
        c.upload(src, "/remote/x")


# --- SshResult.ok convenience -----------------------------------------------
def test_ssh_result_ok_property() -> None:
    assert SshResult(returncode=0, stdout="", stderr="", duration_s=0.1).ok
    assert not SshResult(returncode=1, stdout="", stderr="", duration_s=0.1).ok


# --- real cluster round-trip (gated) ----------------------------------------
@pytest.mark.skipif(
    not os.environ.get("MCODE_TEST_SSH_LOGIN"),
    reason="set MCODE_TEST_SSH_LOGIN=user@host to enable",
)
def test_real_ssh_roundtrip() -> None:
    c = SshClient(os.environ["MCODE_TEST_SSH_LOGIN"])
    r = c.run("echo mcode-ssh-test", timeout=20.0)
    assert r.ok
    assert "mcode-ssh-test" in r.stdout
