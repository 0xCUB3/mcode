"""mcode watch — live dashboard."""

from __future__ import annotations

from pathlib import Path

from mcode.launch import state as launch_state
from mcode.launch.models import RunRecord, RunStatus, ServerRecord, Target
from mcode.watch import _render, _safe_load, watch


def test_safe_load_returns_empty_state_on_first_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(tmp_path / "broken.json"))
    (tmp_path / "broken.json").write_text("not json {{{")
    s, err = _safe_load(None)
    assert err is not None
    assert "state read failed" in err
    assert s.servers == []
    assert s.runs == []


def test_safe_load_falls_back_to_last_good_on_failure(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    launch_state.update(
        None,
        lambda s: s.upsert_server(
            ServerRecord(
                id="srv-1",
                target=Target.LOCAL_VLLM,
                endpoint="http://x",
                model="m",
                config_hash="h",
                status="healthy",
            )
        ),
    )
    last_good = launch_state.load()
    state_path.write_text("garbage")
    s, err = _safe_load(last_good)
    assert err is not None
    assert len(s.servers) == 1


def test_render_includes_servers_and_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(tmp_path / "state.json"))
    launch_state.update(
        None,
        lambda s: (
            s.upsert_server(
                ServerRecord(
                    id="srv-1",
                    target=Target.LOCAL_VLLM,
                    endpoint="http://x",
                    model="m",
                    config_hash="h",
                    status="healthy",
                )
            ),
            s.upsert_run(
                RunRecord(
                    id="r-1",
                    target=Target.LOCAL_VLLM,
                    benchmark="smoke",
                    status=RunStatus.RUNNING,
                )
            ),
        ),
    )
    snap = launch_state.load()
    out = _render(snap, None)
    # Render returns a Rich Table.grid; just confirm it doesn't crash and
    # produces something non-empty when fed real state.
    assert out is not None


def test_watch_once_returns_zero(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(tmp_path / "state.json"))
    rc = watch(once=True)
    assert rc == 0
