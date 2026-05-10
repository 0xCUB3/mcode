"""bench list / bench cancel."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from mcode.bench import cancel as cancel_mod
from mcode.bench import cli as bench_cli
from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig
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
                "artifacts_fetched_at": 123.0,
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
    assert payload[0]["artifacts_fetched_at"] == 123.0
    assert payload[0]["local_artifact_dir"] == "artifacts"


def test_list_runs_filters_benchmark_status_and_artifacts(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.BLUEVELA,
            benchmark="suite",
            status=RunStatus.DONE,
            remote={"remote_artifact_dir": "/remote/artifacts"},
        )
    )
    _put(
        RunRecord(
            id="r2",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.RUNNING,
        )
    )
    rc = cancel_mod.list_runs(
        json_mode=True,
        benchmark="suite",
        status="done",
        artifacts_only=True,
    )
    assert rc == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert [row["id"] for row in payload] == ["r1"]


def test_list_runs_limit_returns_latest_first(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="older",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            started_at=1.0,
        )
    )
    _put(
        RunRecord(
            id="newer",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            started_at=2.0,
        )
    )
    rc = cancel_mod.list_runs(json_mode=True, limit=1)
    assert rc == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert [row["id"] for row in payload] == ["newer"]


def test_cancel_unknown_run_raises(isolated_state: Path) -> None:
    with pytest.raises(MCodeError, match="no run with id"):
        cancel_mod.cancel_run("nope")


def test_show_accepts_compact_list_id(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="bench-1234567890-abcdef12-aider-polyglot",
            target=Target.LOCAL_VLLM,
            benchmark="aider-polyglot",
            status=RunStatus.DONE,
            started_at=1.0,
        )
    )

    rc = cancel_mod.show_run("1234567890-abcdef12")

    assert rc == 0
    out = capsys.readouterr().out
    assert "bench-1234567890-abcdef12-aider-polyglot" in out


def test_show_latest_picks_newest_run(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="older",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            started_at=1.0,
        )
    )
    _put(
        RunRecord(
            id="newer",
            target=Target.LOCAL_VLLM,
            benchmark="suite",
            status=RunStatus.FAILED,
            started_at=2.0,
        )
    )

    rc = cancel_mod.show_run(latest=True)

    assert rc == 0
    out = capsys.readouterr().out
    assert "newer" in out
    assert "suite" in out


def test_show_run_includes_db_summary_and_commands(
    isolated_state: Path, tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        results_run_id = rdb.start_run(
            "smoke",
            {"model_id": "m", "backend_name": "openai", "loop_budget": 2, "timeout_s": 30},
        )
        rdb.save_task_result(
            results_run_id,
            {
                "task_id": "task-1",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 10,
                "exit_code": 1,
                "timed_out": False,
                "error": "Tests failed",
                "terminal_reason": "budget_exhausted",
            },
        )
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            db_path=str(db_path),
            metadata={"results_run_id": results_run_id, "command": "mcode bench smoke"},
        )
    )

    rc = cancel_mod.show_run("r1")

    assert rc == 0
    out = capsys.readouterr().out
    assert "r1" in out
    assert "0/1 passed" in out
    assert "task-1" in out
    assert "mcode export-csv" in out


def test_show_run_does_not_guess_result_id(isolated_state: Path, tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        rdb.start_run(
            "smoke",
            {"model_id": "m", "backend_name": "openai", "loop_budget": 2, "timeout_s": 30},
        )
    _put(
        RunRecord(
            id="r1",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.RUNNING,
            db_path=str(db_path),
        )
    )

    rc = cancel_mod.show_run("r1")

    assert rc == 0
    out = capsys.readouterr().out
    assert "results=" not in out
    assert "results_run_id" not in out


def test_single_benchmark_closes_state_when_setup_fails(
    isolated_state: Path, tmp_path: Path, monkeypatch
) -> None:
    class BrokenResultsDB:
        def __init__(self, _path: Path) -> None:
            raise OSError("db unavailable")

    monkeypatch.setattr(bench_cli, "ResultsDB", BrokenResultsDB)

    with pytest.raises(OSError, match="db unavailable"):
        bench_cli._run_single_benchmark(
            benchmark="smoke",
            config=BenchConfig(model_id="m", backend_name="openai"),
            db=tmp_path / "missing" / "results.db",
            limit=None,
            task_ids=None,
            backend="openai",
            model="m",
            loop_budget=1,
            timeout_s=30,
        )

    runs = launch_state.load().runs
    assert len(runs) == 1
    assert runs[0].status is RunStatus.FAILED


def test_list_runs_table_marks_fetchable_artifacts(isolated_state: Path, capsys) -> None:
    _put(
        RunRecord(
            id="r1",
            target=Target.BLUEVELA,
            benchmark="suite",
            status=RunStatus.DONE,
            remote={
                "remote_artifact_dir": "/remote/artifacts",
                "artifacts_fetched_at": 123.0,
            },
        )
    )
    rc = cancel_mod.list_runs(wide=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert out.lower().count("yes") >= 2


def test_prune_dry_run_does_not_delete(isolated_state: Path, tmp_path: Path, capsys) -> None:
    missing_db = tmp_path / "missing.db"
    existing_db = tmp_path / "existing.db"
    existing_db.write_text("placeholder")
    _put(
        RunRecord(
            id="stale",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.FAILED,
            db_path=str(missing_db),
            started_at=1.0,
        )
    )
    _put(
        RunRecord(
            id="kept",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.FAILED,
            db_path=str(existing_db),
            started_at=1.0,
        )
    )

    rc = cancel_mod.prune_runs(status="failed")

    assert rc == 0
    out = capsys.readouterr().out
    assert "would remove 1" in out
    assert "stale" in out
    assert {run.id for run in launch_state.load().runs} == {"stale", "kept"}


def test_prune_deletes_matching_old_missing_db_runs(
    isolated_state: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cancel_mod.time, "time", lambda: 10_000.0)
    _put(
        RunRecord(
            id="old-missing",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            db_path=str(tmp_path / "missing.db"),
            started_at=1.0,
        )
    )
    _put(
        RunRecord(
            id="new-missing",
            target=Target.LOCAL_VLLM,
            benchmark="smoke",
            status=RunStatus.DONE,
            db_path=str(tmp_path / "missing-new.db"),
            started_at=9_999.0,
        )
    )

    rc = cancel_mod.prune_runs(older_than="1h", yes=True)

    assert rc == 0
    assert [run.id for run in launch_state.load().runs] == ["new-missing"]


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
            remote={"login": "testuser@host", "pid": "4242", "run_dir": "/u/testuser/x"},
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
            remote={"login": "testuser@host", "job_id": "871884", "run_dir": "/u/testuser/x"},
        )
    )

    rc = cancel_mod.cancel_run("r1")

    assert rc == 0
    assert any(c == "bkill 871884" for c in calls)
    assert any("bjobs -noheader -o stat 871884" in c for c in calls)
