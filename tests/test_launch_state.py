from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from mcode.launch.models import RunHandle
from mcode.launch.state import LauncherState, WorkspaceHandle, load_state, merge_run, update_state


def _update_state_worker(state_path: str, signature: str, delay_s: float) -> None:
    def _update(state: LauncherState) -> None:
        time.sleep(delay_s)
        state.workspaces.append(WorkspaceHandle(signature=signature, path=f"/tmp/{signature}"))

    update_state(Path(state_path), _update)


def test_load_state_treats_empty_file_as_empty_state(tmp_path: Path) -> None:
    state_path = tmp_path / "launch-state.json"
    state_path.write_text("")

    state = load_state(state_path)

    assert state == LauncherState()


def test_locked_state_serializes_concurrent_updates(tmp_path: Path) -> None:
    state_path = tmp_path / "launch-state.json"
    ctx = multiprocessing.get_context("spawn")
    first = ctx.Process(target=_update_state_worker, args=(str(state_path), "one", 0.3))
    second = ctx.Process(target=_update_state_worker, args=(str(state_path), "two", 0.0))

    first.start()
    time.sleep(0.05)
    second.start()
    first.join(5)
    second.join(5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    state = load_state(state_path)
    assert [workspace.signature for workspace in state.workspaces] == ["one", "two"]


def test_merge_run_replaces_existing_run_by_id(tmp_path: Path) -> None:
    state_path = tmp_path / "launch-state.json"
    merge_run(
        state_path,
        RunHandle(
            id="run-1",
            target="openai-compatible",
            benchmark="swebench-lite",
            status="planned",
            metadata={"step": "first"},
            log_path="a.log",
        ),
    )

    merge_run(
        state_path,
        RunHandle(
            id="run-1",
            target="openai-compatible",
            benchmark="swebench-lite",
            status="running",
            metadata={"step": "second"},
            log_path="b.log",
        ),
    )

    state = load_state(state_path)

    assert len(state.runs) == 1
    assert state.runs[0].status == "running"
    assert state.runs[0].metadata == {"step": "second"}
    assert state.runs[0].log_path == "b.log"
