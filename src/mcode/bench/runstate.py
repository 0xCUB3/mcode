"""RunRecord lifecycle helpers for bench invocations.

Wave 1 wires every bench entrypoint (local sharded, local single, Blue Vela
remote) through these helpers so `mcode bench list` and `mcode bench cancel`
(Wave 4) have a consistent view of what's running. Old state files load
unchanged because every new field on `RunRecord` has a default and no new
`RunStatus` value is introduced — cancellation reuses `RunStatus.STOPPED`
plus `metadata["cancel_reason"]`.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from mcode.launch import state as launch_state
from mcode.launch.models import RunRecord, RunStatus, Target


def make_run_id(benchmark: str) -> str:
    """Stable run id format used by every bench entrypoint."""
    return f"bench-{int(time.time())}-{uuid.uuid4().hex[:8]}-{benchmark}"


def open_run(
    *,
    run_id: str,
    benchmark: str,
    target: Target,
    db_path: Path,
    shard_pids: list[int] | None = None,
    remote: dict | None = None,
) -> None:
    """Create the initial RunRecord with status=running."""

    def _mutator(s: launch_state.State) -> None:
        rec = RunRecord(
            id=run_id,
            target=target,
            benchmark=benchmark,
            status=RunStatus.RUNNING,
            db_path=str(db_path),
            shard_pids=list(shard_pids or []),
            remote=dict(remote or {}),
            started_at=time.time(),
            updated_at=str(time.time()),
        )
        s.upsert_run(rec)

    launch_state.update(None, _mutator)


def patch_run(
    *,
    run_id: str,
    shard_pids: list[int] | None = None,
    remote: dict | None = None,
    progress: dict | None = None,
) -> None:
    """Patch a run in flight (e.g. record shard pids once subprocesses launch)."""

    def _mutator(s: launch_state.State) -> None:
        rec = s.run(run_id)
        if rec is None:
            return
        if shard_pids is not None:
            rec.shard_pids = list(shard_pids)
        if remote is not None:
            rec.remote = {**rec.remote, **remote}
        if progress is not None:
            rec.progress = {**rec.progress, **progress}
        rec.updated_at = str(time.time())
        s.upsert_run(rec)

    launch_state.update(None, _mutator)


def close_run(
    *,
    run_id: str,
    status: RunStatus,
    cancel_reason: str | None = None,
) -> None:
    """Final close. Sets ended_at and (optionally) metadata.cancel_reason."""

    def _mutator(s: launch_state.State) -> None:
        rec = s.run(run_id)
        if rec is None:
            return
        rec.status = status
        rec.ended_at = time.time()
        if cancel_reason is not None:
            rec.metadata = {**rec.metadata, "cancel_reason": cancel_reason}
        rec.updated_at = str(time.time())
        s.upsert_run(rec)

    launch_state.update(None, _mutator)


__all__ = ["close_run", "make_run_id", "open_run", "patch_run"]
