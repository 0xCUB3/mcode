"""mcode launch wait — block until server reaches a terminal state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcode.launch import state
from mcode.launch.cli import app
from mcode.launch.models import ServerRecord, Target


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(p))
    return p


def _put(state_path: Path, srv: ServerRecord) -> None:
    state.update(state_path, lambda s: s.upsert_server(srv))


def test_wait_exits_zero_when_already_healthy(runner: CliRunner, isolated_state: Path) -> None:
    _put(
        isolated_state,
        ServerRecord(
            id="server-x",
            target=Target.LOCAL_VLLM,
            endpoint="http://127.0.0.1:8321/v1",
            model="x",
            config_hash="h",
            status="healthy",
        ),
    )
    res = runner.invoke(app, ["wait", "server-x", "--timeout", "5"])
    assert res.exit_code == 0
    assert "healthy" in res.output


def test_wait_exits_one_when_failed(runner: CliRunner, isolated_state: Path) -> None:
    _put(
        isolated_state,
        ServerRecord(
            id="server-x",
            target=Target.LOCAL_VLLM,
            endpoint="",
            model="x",
            config_hash="h",
            status="failed",
        ),
    )
    res = runner.invoke(app, ["wait", "server-x", "--timeout", "5"])
    assert res.exit_code == 1


def test_wait_exits_three_when_id_unknown(runner: CliRunner, isolated_state: Path) -> None:
    res = runner.invoke(app, ["wait", "no-such-id", "--timeout", "5"])
    assert res.exit_code == 3


def test_wait_json_mode_emits_payload(runner: CliRunner, isolated_state: Path) -> None:
    _put(
        isolated_state,
        ServerRecord(
            id="server-x",
            target=Target.LOCAL_VLLM,
            endpoint="http://e",
            model="x",
            config_hash="h",
            status="healthy",
        ),
    )
    res = runner.invoke(app, ["wait", "server-x", "--json", "--timeout", "5"])
    assert res.exit_code == 0
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert payload == {"id": "server-x", "status": "healthy", "endpoint": "http://e"}


def test_wait_times_out(runner: CliRunner, isolated_state: Path, monkeypatch) -> None:
    _put(
        isolated_state,
        ServerRecord(
            id="server-x",
            target=Target.LOCAL_VLLM,
            endpoint="",
            model="x",
            config_hash="h",
            status="pending",
        ),
    )
    # Skip the real sleep so the test is fast.
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    res = runner.invoke(app, ["wait", "server-x", "--timeout", "1", "--poll", "1.0"])
    assert res.exit_code == 2
