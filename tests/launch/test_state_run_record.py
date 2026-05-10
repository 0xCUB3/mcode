"""Schema-bump compat: old state files load; STOPPED + cancel_reason works."""

from __future__ import annotations

import json
from pathlib import Path

from mcode.launch import state
from mcode.launch.models import RunRecord, RunStatus, Target


def test_old_state_file_loads_with_no_new_fields(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))

    state_path.write_text(
        json.dumps(
            {
                "servers": [],
                "runs": [
                    {
                        "id": "old-run",
                        "target": "bluevela",
                        "benchmark": "swebench-live",
                        "status": "running",
                        "shard_job_ids": ["1", "2"],
                        "metadata": {},
                    }
                ],
            }
        )
    )

    snap = state.load()
    assert len(snap.runs) == 1
    rec = snap.runs[0]
    assert rec.id == "old-run"
    assert rec.shard_pids == []  # default for new field
    assert rec.remote == {}
    assert rec.progress == {}
    assert rec.started_at is None
    assert rec.ended_at is None
    assert rec.db_path is None


def test_stopped_with_cancel_reason_round_trip(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))

    def _write(s: state.State) -> None:
        s.upsert_run(
            RunRecord(
                id="r1",
                target=Target.LOCAL_VLLM,
                benchmark="smoke",
                status=RunStatus.STOPPED,
                shard_pids=[111, 222],
                metadata={"cancel_reason": "user"},
            )
        )

    state.update(None, _write)

    snap = state.load()
    rec = snap.runs[0]
    assert rec.status == RunStatus.STOPPED
    assert rec.metadata["cancel_reason"] == "user"
    assert rec.shard_pids == [111, 222]


def test_no_new_run_status_values_introduced():
    """Cancellation invariant: cancellation must reuse STOPPED, not a
    new enum value, so older mcode binaries don't drop records on load."""
    values = {v.value for v in RunStatus}
    assert values == {"submitted", "running", "done", "failed", "stopped"}
