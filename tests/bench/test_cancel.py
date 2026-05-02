"""bench list / bench cancel."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from mcode.bench import cancel as cancel_mod
from mcode.launch import state as launch_state
from mcode.launch.models import RunRecord, RunStatus, Target
from mcode.ui.errors import ExitCode, MCodeError


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(p))
    return p


def _put(run: RunRecord) -> None:
    launch_state.update(None, lambda s: s.upsert_run(run))


def test_list_runs_empty(isolated_state: Path, capsys) -> None:
    rc = cancel_mod.list_runs()
    assert rc == 0
    out = capsys.readouterr().out
    assert "no runs recorded" in out


def test_list_runs_json_includes_cancel_reason(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.STOPPED,
            metadata={"cancel_reason": "user"},
            shard_pids=[42],
            remote={
                "remote_artifact_dir": "/remote/artifacts",
                "local_artifact_dir": "artifacts",
            },
        )
    )
    rc = cancel_mod.list_runs(json_mode=True)
    assert rc == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload[0]["cancel_reason"] == "user"
    assert payload[0]["shards"] == 1
    assert payload[0]["artifacts_fetchable"] is True
    assert payload[0]["local_artifact_dir"] == "artifacts"

def test_cancel_unknown_run_raises(isolated_state: Path) -> None:
    with pytest.raises(MCodeError, match="no run with id"):
        cancel_mod.cancel_run("nope")


def test_list_runs_table_marks_fetchable_artifacts(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.BLUEVELA,
            benchmark="suite",
            status=RunStatus.DONE,
            remote={"remote_artifact_dir": "/remote/artifacts"},
        )
    )
    rc = cancel_mod.list_runs()
    assert rc == 0
    out = capsys.readouterr().out
    assert "artifacts" in out.lower()
    assert "yes" in out.lower()


def test_cancel_already_terminal_returns_ok(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="x",
            status=RunStatus.DONE,
        )
    )
    rc = cancel_mod.cancel_run("r1")
    assert rc == 0
    assert "already done" in capsys.readouterr().out


def test_cancel_not_cancellable_for_in_process_run(isolated_state: Path) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="x",
            status=RunStatus.RUNNING,
            # No shard_pids and no remote — single in-process run.
        )
    )
    with pytest.raises(MCodeError) as exc:
        cancel_mod.cancel_run("r1")
    assert exc.value.exit_code == ExitCode.NOT_CANCELLABLE


def test_cancel_local_signals_each_pid(isolated_state: Path, monkeypatch, capsys) -> None:
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError  # report dead so loop exits fast

    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(cancel_mod, "_pid_alive", lambda pid: False)
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="x",
            status=RunStatus.RUNNING,
            shard_pids=[1111, 2222],
        )
    )
    rc = cancel_mod.cancel_run("r1")
    assert rc == 0
    sent = {(pid, sig) for pid, sig in signals}
    assert (1111, signal.SIGTERM) in sent
    assert (2222, signal.SIGTERM) in sent

    # State now reflects the cancellation.
    snap = launch_state.load()
    rec = snap.run("r1")
    assert rec is not None
    assert rec.status == RunStatus.STOPPED
    assert rec.metadata["cancel_reason"] == "user"


def test_cancel_remote_calls_ssh_kill(isolated_state: Path, monkeypatch) -> None:
    calls: list[str] = []

    class _FakeSsh:
        def __init__(self, login: str) -> None:
            self.login = login

        def run(self, cmd: str, *, timeout: float = 30.0):
            calls.append(cmd)
            return type("_R", (), {"ok": True, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("mcode.launch.ssh.SshClient", _FakeSsh)
    monkeypatch.setattr("mcode.bench.cancel.time.sleep", lambda s: None)
    _put(
        RunRecord(
            id="r1",
            target=Target.BLUEVELA,
            benchmark="smoke",
            status=RunStatus.RUNNING,
            remote={"login": "skula@bv", "pid": "4242", "run_dir": "/u/skula/x"},
        )
    )
    rc = cancel_mod.cancel_run("r1")
    assert rc == 0
    assert any("kill -TERM -4242" in c for c in calls)
    assert any("kill -KILL -4242" in c for c in calls)


def test_cancel_remote_lsf_calls_bkill(isolated_state: Path, monkeypatch) -> None:
    calls: list[str] = []

    class _FakeSsh:
        def __init__(self, login: str) -> None:
            self.login = login

        def run(self, cmd: str, *, timeout: float = 30.0):
            calls.append(cmd)
            return type("_R", (), {"ok": True, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("mcode.launch.ssh.SshClient", _FakeSsh)
    monkeypatch.setattr("mcode.bench.cancel.time.sleep", lambda s: None)
    _put(
        RunRecord(
            id="r1",
            target=Target.BLUEVELA,
            benchmark="smoke",
            status=RunStatus.RUNNING,
            remote={"login": "skula@bv", "job_id": "871884", "run_dir": "/u/skula/x"},
        )
    )

    rc = cancel_mod.cancel_run("r1")

    assert rc == 0
    assert any(c == "bkill 871884" for c in calls)
    assert any("bjobs -noheader -o stat 871884" in c for c in calls)
