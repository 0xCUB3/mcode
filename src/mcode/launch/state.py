from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mcode.launch.models import RunHandle, ServerHandle, WorkspaceHandle, default_state_path


@dataclass
class LauncherState:
    servers: list[ServerHandle] = field(default_factory=list)
    runs: list[RunHandle] = field(default_factory=list)
    workspaces: list[WorkspaceHandle] = field(default_factory=list)


def load_state(path: Path | None = None) -> LauncherState:
    state_path = path or Path(os.environ.get("MCODE_LAUNCH_STATE", default_state_path()))
    if not state_path.exists():
        return LauncherState()
    data = json.loads(state_path.read_text())
    return LauncherState(
        servers=[ServerHandle(**entry) for entry in data.get("servers", [])],
        runs=[RunHandle(**entry) for entry in data.get("runs", [])],
        workspaces=[WorkspaceHandle(**entry) for entry in data.get("workspaces", [])],
    )


def save_state(path: Path | None, state: LauncherState) -> None:
    state_path = path or Path(os.environ.get("MCODE_LAUNCH_STATE", default_state_path()))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [asdict(server) for server in state.servers],
        "runs": [asdict(run) for run in state.runs],
        "workspaces": [asdict(workspace) for workspace in state.workspaces],
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
