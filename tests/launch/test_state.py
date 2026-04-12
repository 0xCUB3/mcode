from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcode.launch import state
from mcode.launch.models import RunRecord, RunStatus, ServerRecord, Target


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "launch-state.json"


def test_load_missing_returns_empty(state_path: Path) -> None:
    s = state.load(state_path)
    assert s.servers == []
    assert s.runs == []


def test_update_roundtrip(state_path: Path) -> None:
    def mutate(s: state.State) -> str:
        s.upsert_server(
            ServerRecord(
                id="server-1",
                target=Target.BLUEVELA,
                endpoint="http://x:8000/v1",
                model="Qwen/Qwen3.5-27B",
                config_hash="abc",
            )
        )
        s.upsert_run(
            RunRecord(
                id="run-1",
                target=Target.BLUEVELA,
                benchmark="swebench-live",
                server_id="server-1",
                shard_job_ids=["42", "43"],
            )
        )
        return "ok"

    result = state.update(state_path, mutate)
    assert result == "ok"

    reloaded = state.load(state_path)
    assert len(reloaded.servers) == 1
    assert reloaded.servers[0].id == "server-1"
    assert reloaded.servers[0].target == Target.BLUEVELA
    assert len(reloaded.runs) == 1
    run = reloaded.runs[0]
    assert run.id == "run-1"
    assert run.status == RunStatus.SUBMITTED
    assert run.shard_job_ids == ["42", "43"]


def test_upsert_replaces_by_id(state_path: Path) -> None:
    def first(s: state.State) -> None:
        s.upsert_server(
            ServerRecord(
                id="s",
                target=Target.LOCAL_VLLM,
                endpoint="a",
                model="m",
                config_hash="h1",
            )
        )

    def second(s: state.State) -> None:
        s.upsert_server(
            ServerRecord(
                id="s",
                target=Target.LOCAL_VLLM,
                endpoint="b",
                model="m",
                config_hash="h2",
            )
        )

    state.update(state_path, first)
    state.update(state_path, second)
    s = state.load(state_path)
    assert len(s.servers) == 1
    assert s.servers[0].endpoint == "b"
    assert s.servers[0].config_hash == "h2"


def test_atomic_write_leaves_no_tmp(state_path: Path) -> None:
    def mutate(s: state.State) -> None:
        s.upsert_run(RunRecord(id="r", target=Target.LOCAL_VLLM, benchmark="x"))

    state.update(state_path, mutate)
    # No leftover .tmp files in the state dir
    tmp_leftovers = list(state_path.parent.glob(f".{state_path.name}.*.tmp"))
    assert tmp_leftovers == []


def test_lock_serializes_concurrent_updates(state_path: Path) -> None:
    # Simulate two in-process mutators; fcntl makes them serialize.
    # We just assert both mutations land; deeper concurrency test needs subprocess.
    def add_run(run_id: str):
        def m(s: state.State) -> None:
            s.upsert_run(RunRecord(id=run_id, target=Target.BLUEVELA, benchmark="b"))

        return m

    state.update(state_path, add_run("a"))
    state.update(state_path, add_run("b"))
    s = state.load(state_path)
    assert {r.id for r in s.runs} == {"a", "b"}


def test_json_schema_matches_expected(state_path: Path) -> None:
    def mutate(s: state.State) -> None:
        s.upsert_server(
            ServerRecord(
                id="sv",
                target=Target.BLUEVELA,
                endpoint="http://h:8321/v1",
                model="google/gemma-4-31B-it",
                config_hash="deadbeef",
                refs=["run-1"],
                metadata={"queue": "normal"},
            )
        )

    state.update(state_path, mutate)
    data = json.loads(state_path.read_text())
    assert set(data.keys()) == {"servers", "runs"}
    server = data["servers"][0]
    assert server["target"] == "bluevela"
    assert server["refs"] == ["run-1"]
    assert server["metadata"] == {"queue": "normal"}
