"""Atomic JSON state file with fcntl locking.

State shape (JSON):

    {"servers": [ServerRecord, ...], "runs": [RunRecord, ...]}

Invariants:

- Writes are atomic (tmp file + rename within the same directory).
- All reads/writes go through `update_state()` which holds an fcntl exclusive
  lock on a sibling .lock file. No concurrent writers on the same machine.
- Workspaces are NOT tracked here — the rewrite rsyncs the working tree every
  launch (see plan "Sync" section).

The fcntl + atomic-write pattern is consciously ported from main's launcher;
it survived 19 bug-fix commits untouched.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

from mcode.launch.models import (
    RunRecord,
    RunStatus,
    ServerRecord,
    Target,
    default_state_path,
)

T = TypeVar("T")


def _resolve_state_path(path: Path | None = None) -> Path:
    return path or Path(os.environ.get("MCODE_LAUNCH_STATE", default_state_path()))


def _filter_fields(cls, data: dict) -> dict:
    # Drop unknown keys so state files written by older launcher versions
    # (e.g. pre-rewrite `reuse_key`) load cleanly.
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in allowed}


def _run_from_dict(data: dict) -> RunRecord:
    data = dict(data)
    data["target"] = Target(data["target"])
    data["status"] = RunStatus(data["status"])
    return RunRecord(**_filter_fields(RunRecord, data))


def _server_from_dict(data: dict) -> ServerRecord:
    data = dict(data)
    data["target"] = Target(data["target"])
    return ServerRecord(**_filter_fields(ServerRecord, data))


def _load(state_path: Path) -> tuple[list[ServerRecord], list[RunRecord]]:
    if not state_path.exists():
        return [], []
    raw = state_path.read_text().strip()
    if not raw:
        return [], []
    data = json.loads(raw)
    servers: list[ServerRecord] = []
    for s in data.get("servers", []):
        try:
            servers.append(_server_from_dict(s))
        except (TypeError, ValueError):
            # Incompatible record from a prior launcher schema; drop it.
            continue
    runs: list[RunRecord] = []
    for r in data.get("runs", []):
        try:
            runs.append(_run_from_dict(r))
        except (TypeError, ValueError):
            continue
    return servers, runs


def _save(state_path: Path, servers: list[ServerRecord], runs: list[RunRecord]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [asdict(s) for s in servers],
        "runs": [asdict(r) for r in runs],
    }
    with tempfile.NamedTemporaryFile(
        "w",
        dir=state_path.parent,
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
        tmp_path = Path(fh.name)
    os.replace(tmp_path, state_path)


class State:
    """Mutable view of the state file held open under an fcntl lock."""

    def __init__(self, servers: list[ServerRecord], runs: list[RunRecord]) -> None:
        self.servers = servers
        self.runs = runs

    def server(self, server_id: str) -> ServerRecord | None:
        return next((s for s in self.servers if s.id == server_id), None)

    def run(self, run_id: str) -> RunRecord | None:
        return next((r for r in self.runs if r.id == run_id), None)

    def upsert_server(self, server: ServerRecord) -> ServerRecord:
        self.servers = [s for s in self.servers if s.id != server.id] + [server]
        return server

    def upsert_run(self, run: RunRecord) -> RunRecord:
        self.runs = [r for r in self.runs if r.id != run.id] + [run]
        return run


@contextmanager
def _locked(path: Path | None = None) -> Iterator[tuple[Path, State]]:
    state_path = _resolve_state_path(path)
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            servers, runs = _load(state_path)
            state = State(servers, runs)
            yield state_path, state
            _save(state_path, state.servers, state.runs)
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def load(path: Path | None = None) -> State:
    """Read-only snapshot (no lock held after return)."""
    servers, runs = _load(_resolve_state_path(path))
    return State(servers, runs)


def update(path: Path | None, mutator: Callable[[State], T]) -> T:
    """Hold the lock, mutate, persist. Return whatever the mutator returned."""
    with _locked(path) as (_, state):
        return mutator(state)
