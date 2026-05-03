"""bench list + bench cancel implementations.

Cancel design (per Wave 4 of the plan): three-way dispatch on the RunRecord
shape.

  if shard_pids   → local sharded run; SIGTERM/SIGKILL each pid
  elif remote.job_id → Blue Vela run; bkill the LSF job
  elif remote.pid → legacy Blue Vela run; ssh kill -TERM/-KILL the captured pid
  else            → in-process single run; not cancellable from another shell

State transitions to RunStatus.STOPPED with metadata.cancel_reason="user" so
`bench list` can distinguish user cancellation from server stop.
"""

from __future__ import annotations

import os
import shlex
import signal
import time
from datetime import datetime

from mcode.launch import state as launch_state
from mcode.launch.models import RunRecord, RunStatus
from mcode.ui.console import console
from mcode.ui.errors import ExitCode, MCodeError

_KILL_GRACE_S = 10.0


def list_runs(
    *,
    json_mode: bool = False,
    benchmark: str | None = None,
    status: str | None = None,
    artifacts_only: bool = False,
    limit: int | None = None,
) -> int:
    """Print known runs, optionally filtered."""
    s = launch_state.load()
    runs = list(s.runs)
    if benchmark:
        runs = [run for run in runs if run.benchmark == benchmark]
    if status:
        runs = [run for run in runs if run.status.value == status]
    if artifacts_only:
        runs = [run for run in runs if (run.remote or {}).get("remote_artifact_dir")]
    runs.sort(key=lambda run: float(run.started_at or 0.0), reverse=True)
    if limit is not None:
        runs = runs[:limit]
    if json_mode:
        import json

        payload = [
            {
                "id": r.id,
                "benchmark": r.benchmark,
                "target": r.target.value,
                "status": r.status.value,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "shards": len(r.shard_job_ids) or len(r.shard_pids),
                "db_path": r.db_path,
                "progress": r.progress,
                "cancel_reason": (r.metadata or {}).get("cancel_reason"),
                "artifacts_fetchable": bool((r.remote or {}).get("remote_artifact_dir")),
                "artifacts_fetched_at": (r.remote or {}).get("artifacts_fetched_at"),
                "local_artifact_dir": (r.remote or {}).get("local_artifact_dir"),
            }
            for r in runs
        ]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not runs:
        print("no runs recorded")
        return 0

    from rich.table import Table

    table = Table(title=f"runs ({len(runs)})")
    table.add_column("id")
    table.add_column("benchmark")
    table.add_column("target")
    table.add_column("status")
    table.add_column("shards", justify="right")
    table.add_column("artifacts")
    table.add_column("fetched")
    table.add_column("started")
    table.add_column("progress")
    for r in runs:
        shards = len(r.shard_job_ids) or len(r.shard_pids)
        started = _format_ts(r.started_at)
        progress = _format_progress(r.progress)
        status_value = r.status.value
        if (r.metadata or {}).get("cancel_reason"):
            status_value = f"{status_value} ({r.metadata['cancel_reason']})"
        artifacts = "yes" if (r.remote or {}).get("remote_artifact_dir") else "-"
        fetched = "yes" if (r.remote or {}).get("artifacts_fetched_at") else "-"
        table.add_row(
            r.id,
            r.benchmark,
            r.target.value,
            status_value,
            str(shards),
            artifacts,
            fetched,
            started,
            progress,
        )
    console.print(table)
    return 0


def cancel_run(run_id: str) -> int:
    """Cancel a run by id. Three-way dispatch on RunRecord shape.

    Returns:
      0 — cancellation succeeded (or run was already terminal).
      Raises MCodeError otherwise; caller maps to typer.Exit.
    """
    s = launch_state.load()
    run = s.run(run_id)
    if run is None:
        raise MCodeError(
            what=f"no run with id {run_id!r}",
            why="",
            next="`mcode bench list` to see runs",
        )

    if run.status not in (RunStatus.RUNNING, RunStatus.SUBMITTED):
        print(f"run {run_id} is already {run.status.value}; nothing to cancel")
        return 0

    if run.shard_pids:
        return _cancel_local(run)
    if run.remote:
        return _cancel_remote(run)

    # In-process single run: no separate PID to signal from another shell.
    err = MCodeError(
        what=f"run {run_id} is not cancellable from another shell",
        why="single non-sharded local runs execute in-process — there is no child to signal",
        next="Ctrl+C in the running terminal",
    )
    err.exit_code = ExitCode.NOT_CANCELLABLE
    raise err


def _cancel_local(run: RunRecord) -> int:
    """SIGTERM each shard pid; SIGKILL stragglers after _KILL_GRACE_S.

    Best-effort identity check: skip pids whose `started_at` precedes
    `run.started_at` because they're almost certainly recycled OS pids
    pointing at unrelated processes. The kernel's pid-recycle window is
    small but not zero, and a stale state record after a reboot/laptop-
    sleep cycle can otherwise nuke an unrelated user process.
    """
    surviving: list[int] = []
    for pid in run.shard_pids:
        if not _pid_owned_by_run(pid, run):
            print(f"⚠ pid {pid} looks recycled (started before run); skipping")
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            surviving.append(pid)
        except ProcessLookupError:
            pass  # already exited
        except PermissionError:
            print(f"⚠ no permission to signal pid {pid}; skipping")
    if surviving:
        deadline = time.monotonic() + _KILL_GRACE_S
        while surviving and time.monotonic() < deadline:
            time.sleep(0.5)
            surviving = [p for p in surviving if _pid_alive(p)]
        for pid in surviving:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    _mark_cancelled(run.id)
    print(f"✓ cancelled {run.id} (local, {len(run.shard_pids)} shard pids)")
    return 0


def _cancel_remote(run: RunRecord) -> int:
    """Cancel a Blue Vela run by LSF job id, or legacy process group pid."""
    from mcode.launch.ssh import SshClient

    login = run.remote.get("login", "")
    job_id = run.remote.get("job_id", "")
    if login and job_id:
        return _cancel_remote_lsf(run, login=str(login), job_id=str(job_id))

    pid = run.remote.get("pid", "")
    if not login or not pid:
        raise MCodeError(
            what=f"run {run.id} has incomplete remote metadata",
            why=f"remote={run.remote!r}; expected login + job_id or login + pid",
            next="run already terminated, or state file was hand-edited",
        )
    pid_str = shlex.quote(str(pid))
    ssh = SshClient(login)
    ssh_errors: list[str] = []
    try:
        r1 = ssh.run(f"kill -TERM -{pid_str} 2>/dev/null || true", timeout=10)
        if not getattr(r1, "ok", True):
            ssh_errors.append((r1.stderr or "kill -TERM failed").strip())
    except Exception as e:
        ssh_errors.append(f"TERM: {e}")
    time.sleep(min(_KILL_GRACE_S, 5.0))
    try:
        r2 = ssh.run(f"kill -KILL -{pid_str} 2>/dev/null || true", timeout=10)
        if not getattr(r2, "ok", True):
            ssh_errors.append((r2.stderr or "kill -KILL failed").strip())
    except Exception as e:
        ssh_errors.append(f"KILL: {e}")

    alive = True
    try:
        check = ssh.run(f"! kill -0 {pid_str} 2>/dev/null", timeout=10)
        alive = not bool(getattr(check, "ok", True))
    except Exception as e:
        ssh_errors.append(f"verify: {e}")

    if alive:
        raise MCodeError(
            what=f"failed to kill remote run {run.id} (pid {pid} on {login} still alive)",
            why="; ".join(ssh_errors) or "kill verification failed",
            next=(
                f"check VPN/SSH; manually `ssh {login} kill -KILL -{pid}`; "
                f"only then retry `mcode bench cancel {run.id}`"
            ),
        )

    _mark_cancelled(run.id)
    print(f"✓ cancelled {run.id} (remote pid {pid} on {login})")
    if ssh_errors:
        print(f"  note: SSH had transient errors during cancel: {'; '.join(ssh_errors)}")
    print(
        f"  note: orphaned podman containers possible; verify with "
        f"`ssh {login} 'pgrep -af mcode|podman'`"
    )
    return 0


def _cancel_remote_lsf(run: RunRecord, *, login: str, job_id: str) -> int:
    from mcode.launch.ssh import SshClient

    job_id_q = shlex.quote(job_id)
    ssh = SshClient(login)
    ssh_errors: list[str] = []
    try:
        r1 = ssh.run(f"bkill {job_id_q}", timeout=10)
        if not getattr(r1, "ok", True):
            ssh_errors.append((r1.stderr or r1.stdout or "bkill failed").strip())
    except Exception as e:
        ssh_errors.append(f"bkill: {e}")
    time.sleep(min(_KILL_GRACE_S, 5.0))

    active = True
    try:
        check = ssh.run(
            "STAT=$(bjobs -noheader -o stat "
            f"{job_id_q} 2>/dev/null | tr -d '[:space:]' || true); "
            'case "$STAT" in PEND|RUN|PSUSP|USUSP|SSUSP) exit 1 ;; *) exit 0 ;; esac',
            timeout=10,
        )
        active = not bool(getattr(check, "ok", True))
    except Exception as e:
        ssh_errors.append(f"verify: {e}")

    if active:
        raise MCodeError(
            what=f"failed to bkill remote run {run.id} (job {job_id} on {login} still active)",
            why="; ".join(ssh_errors) or "bkill verification failed",
            next=(
                f"check VPN/SSH; manually `ssh {login} bkill {job_id}`; "
                f"only then retry `mcode bench cancel {run.id}`"
            ),
        )

    _mark_cancelled(run.id)
    print(f"✓ cancelled {run.id} (LSF job {job_id} on {login})")
    if ssh_errors:
        print(f"  note: SSH had transient errors during cancel: {'; '.join(ssh_errors)}")
    return 0


def _mark_cancelled(run_id: str) -> None:
    def _mut(s: launch_state.State) -> None:
        rec = s.run(run_id)
        if rec is None:
            return
        rec.status = RunStatus.STOPPED
        rec.ended_at = time.time()
        rec.metadata = {**rec.metadata, "cancel_reason": "user"}
        s.upsert_run(rec)

    launch_state.update(None, _mut)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    return True


def _pid_owned_by_run(pid: int, run: RunRecord) -> bool:
    """Best-effort check that `pid` is the same process the run started.

    On Linux/macOS, `psutil` (if installed) gives us a create-time we can
    compare to `run.started_at`. Without psutil, we fall back to "trust the
    state file" — the state file is fcntl-locked and only the orchestrator
    writes shard_pids, so within a single boot the pid is reliable.

    Returns True when we can't disprove ownership.
    """
    if run.started_at is None:
        return True
    try:
        import psutil  # type: ignore[import-not-found]

        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return False
        try:
            return proc.create_time() >= float(run.started_at) - 5.0
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        return True


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return str(ts)


def _format_progress(p: dict) -> str:
    if not p:
        return "—"
    cur = p.get("current", 0)
    total = p.get("total", 0)
    if total:
        return f"{cur}/{total}"
    return str(cur)


__all__ = ["cancel_run", "list_runs"]
