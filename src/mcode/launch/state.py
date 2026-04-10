from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from mcode.launch.models import RunHandle, ServerHandle, WorkspaceHandle, default_state_path


@dataclass
class LauncherState:
    servers: list[ServerHandle] = field(default_factory=list)
    runs: list[RunHandle] = field(default_factory=list)
    workspaces: list[WorkspaceHandle] = field(default_factory=list)


T = TypeVar("T")


def _resolve_state_path(path: Path | None = None) -> Path:
    return path or Path(os.environ.get("MCODE_LAUNCH_STATE", default_state_path()))


def _load_state_from_path(state_path: Path) -> LauncherState:
    if not state_path.exists():
        return LauncherState()
    raw = state_path.read_text().strip()
    if not raw:
        return LauncherState()
    data = json.loads(raw)
    return LauncherState(
        servers=[ServerHandle(**entry) for entry in data.get("servers", [])],
        runs=[RunHandle(**entry) for entry in data.get("runs", [])],
        workspaces=[WorkspaceHandle(**entry) for entry in data.get("workspaces", [])],
    )


def _save_state_to_path(state_path: Path, state: LauncherState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [asdict(server) for server in state.servers],
        "runs": [asdict(run) for run in state.runs],
        "workspaces": [asdict(workspace) for workspace in state.workspaces],
    }
    with tempfile.NamedTemporaryFile(
        "w",
        dir=state_path.parent,
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, state_path)


@contextmanager
def _locked_state(path: Path | None = None):
    state_path = _resolve_state_path(path)
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = _load_state_from_path(state_path)
        try:
            yield state
        except Exception:
            raise
        else:
            _save_state_to_path(state_path, state)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_state(path: Path | None = None) -> LauncherState:
    return _load_state_from_path(_resolve_state_path(path))


def update_state(path: Path | None, updater) -> T:
    with _locked_state(path) as state:
        return updater(state)


def merge_workspace(path: Path | None, workspace: WorkspaceHandle) -> WorkspaceHandle:
    def _update(state: LauncherState) -> WorkspaceHandle:
        state.workspaces = [
            entry for entry in state.workspaces if entry.signature != workspace.signature
        ] + [workspace]
        return workspace

    return update_state(path, _update)


def merge_server(path: Path | None, server: ServerHandle) -> ServerHandle:
    def _update(state: LauncherState) -> ServerHandle:
        state.servers = [
            entry for entry in state.servers if entry.reuse_key != server.reuse_key
        ] + [server]
        return server

    return update_state(path, _update)


def merge_run(path: Path | None, run: RunHandle) -> RunHandle:
    def _update(state: LauncherState) -> RunHandle:
        state.runs = [entry for entry in state.runs if entry.id != run.id] + [run]
        return run

    return update_state(path, _update)
