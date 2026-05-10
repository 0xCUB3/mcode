"""bench list and bench cancel implementations.

Cancellation dispatches from the stored RunRecord shape:

  if shard_pids      → local sharded run; SIGTERM/SIGKILL each pid
  elif remote.job_id → Blue Vela run; bkill the LSF job
  elif remote.pid    → older Blue Vela run; ssh kill -TERM/-KILL the captured pid
  else               → in-process single run; not cancellable from another shell

State transitions to RunStatus.STOPPED with metadata.cancel_reason="user" so
`bench list` can distinguish user cancellation from server stop.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcode.bench.results import ResultsDB
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
    wide: bool = False,
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
    table.add_column("id", no_wrap=True)
    table.add_column("bench", no_wrap=True)
    if wide:
        table.add_column("target", no_wrap=True)
    table.add_column("status", no_wrap=True)
    if wide:
        table.add_column("shards", justify="right")
        table.add_column("artifacts", no_wrap=True)
        table.add_column("fetched", no_wrap=True)
    table.add_column("started", no_wrap=True)
    table.add_column("progress", no_wrap=True, overflow="ellipsis")
    if wide:
        table.add_column("db", overflow="fold")
    for r in runs:
        shards = len(r.shard_job_ids) or len(r.shard_pids)
        started = _format_ts(r.started_at) if wide else _format_short_ts(r.started_at)
        progress = _format_progress(r.progress, max_len=None if wide else 48)
        status_value = r.status.value
        if (r.metadata or {}).get("cancel_reason"):
            status_value = f"{status_value} ({r.metadata['cancel_reason']})"
        artifacts = "yes" if (r.remote or {}).get("remote_artifact_dir") else "-"
        fetched = "yes" if (r.remote or {}).get("artifacts_fetched_at") else "-"
        row = [_display_run_id(r), _display_benchmark(r.benchmark)]
        if wide:
            row.append(r.target.value)
        row.extend([status_value, started, progress])
        if wide:
            row = [
                _display_run_id(r),
                r.benchmark,
                r.target.value,
                status_value,
                str(shards),
                artifacts,
                fetched,
                started,
                progress,
                r.db_path or "-",
            ]
        table.add_row(*row)
    console.print(table)
    return 0


def show_run(run_id: str | None = None, *, latest: bool = False, json_mode: bool = False) -> int:
    """Print one run with state, result rows, and useful follow-up commands."""
    s = launch_state.load()
    run = _resolve_run(s, run_id=run_id, latest=latest)

    result_info = _result_info(run)
    if json_mode:
        print(json.dumps({"run": _run_payload(run), "results": result_info}, indent=2, default=str))
        return 0

    console.print(f"[bold]{run.id}[/bold]")
    details = _details_table()
    _add_detail(details, "status", run.status.value)
    _add_detail(details, "benchmark", run.benchmark)
    _add_detail(details, "target", run.target.value)
    _add_detail(details, "started", _format_ts(run.started_at))
    _add_detail(details, "ended", _format_ts(run.ended_at))
    if run.db_path:
        _add_detail(details, "db", run.db_path)
    if run.progress:
        _add_detail(details, "progress", _format_progress(run.progress))
    command = (run.metadata or {}).get("command")
    if command:
        _add_detail(details, "rerun", str(command))
    env = (run.metadata or {}).get("env")
    if isinstance(env, dict) and env:
        env_text = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())
        _add_detail(details, "env", env_text)
    if run.remote:
        _add_remote_details(details, run)
    if result_info.get("summary"):
        summary = result_info["summary"]
        _add_detail(
            details,
            "results",
            f"{summary['passed']}/{summary['total']} passed "
            f"({summary['pass_rate']:.1%}) results_run_id={summary['run_id']}",
        )
    console.print(details)
    failures = result_info.get("failures") or []
    if failures:
        console.print("failed tasks:")
        for row in failures[:10]:
            reason = row.get("terminal_reason") or row.get("error") or "failed"
            console.print(f"  - {row['task_id']}: {reason}")
    console.print("commands:")
    _print_command(f"mcode bench cancel {run.id}")
    if run.remote.get("remote_artifact_dir"):
        _print_command(f"mcode bench artifacts fetch {run.id}")
    if run.db_path:
        _print_command(f"mcode export-csv --db {shlex.quote(run.db_path)}")
    return 0


def prune_runs(
    *,
    json_mode: bool = False,
    status: str | None = None,
    older_than: str | None = None,
    missing_db: bool = True,
    yes: bool = False,
) -> int:
    """Remove stale run records from the launch-state file."""
    cutoff = _older_than_cutoff(older_than)
    removed: list[RunRecord] = []

    def _matches(run: RunRecord) -> bool:
        if status and run.status.value != status:
            return False
        if cutoff is not None and float(run.started_at or 0.0) > cutoff:
            return False
        if missing_db and not _db_missing(run):
            return False
        if not status and run.status in (RunStatus.RUNNING, RunStatus.SUBMITTED):
            return False
        return True

    snap = launch_state.load()
    for run in snap.runs:
        if _matches(run):
            removed.append(run)

    if yes and removed:

        def _mut(s: launch_state.State) -> None:
            ids = {run.id for run in removed}
            s.runs = [run for run in s.runs if run.id not in ids]

        launch_state.update(None, _mut)

    if json_mode:
        payload: dict[str, Any] = {
            "dry_run": not yes,
            "removed": len(removed),
            "runs": [_run_payload(run) for run in removed],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    action = "removed" if yes else "would remove"
    if not removed:
        print("no matching runs")
        return 0
    print(f"{action} {len(removed)} run record(s)")
    for run in sorted(removed, key=lambda r: float(r.started_at or 0.0), reverse=True)[:25]:
        reason = _prune_reason(run, missing_db=missing_db, cutoff=cutoff)
        print(f"  - {run.id} {run.benchmark} {run.status.value} {reason}")
    if len(removed) > 25:
        print(f"  … {len(removed) - 25} more")
    if not yes:
        print("dry run; add --yes to delete these records")
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


def _display_benchmark(benchmark: str) -> str:
    return {
        "aider-polyglot": "polyglot",
        "swebench-lite": "swe-lite",
        "swebench-live": "swe-live",
    }.get(benchmark, benchmark)


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return str(ts)


def _resolve_run(state: launch_state.State, *, run_id: str | None, latest: bool) -> RunRecord:
    if latest:
        if run_id:
            raise MCodeError(
                what="pass either a run id or --latest, not both",
                why="",
                next="use `mcode bench show --latest` or `mcode bench show <run-id>`",
            )
        if not state.runs:
            raise MCodeError(
                what="no runs recorded",
                why="",
                next="run a benchmark first",
            )
        return max(state.runs, key=lambda run: float(run.started_at or 0.0))
    if not run_id:
        raise MCodeError(
            what="missing run id",
            why="",
            next="use `mcode bench show --latest` or pass a run id from `mcode bench list`",
        )
    run = state.run(run_id)
    if run is not None:
        return run
    matches = [
        run for run in state.runs if run.id.startswith(f"bench-{run_id}") or run_id in run.id
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise MCodeError(
            what=f"run id {run_id!r} is ambiguous",
            why=f"matched {len(matches)} runs",
            next=(
                "pass the full run id from `mcode bench show --latest` or `mcode bench list --json`"
            ),
        )
    raise MCodeError(
        what=f"no run with id {run_id!r}",
        why="",
        next="`mcode bench list --limit 20` to see recent runs",
    )


def _display_run_id(run: RunRecord) -> str:
    parts = run.id.split("-")
    if len(parts) >= 3 and parts[0] == "bench":
        return f"{parts[1][-5:]}-{parts[2][:6]}"
    return run.id


def _db_missing(run: RunRecord) -> bool:
    if not run.db_path:
        return False
    return not Path(run.db_path).exists()


def _older_than_cutoff(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    suffix = value[-1]
    if suffix in units:
        number = value[:-1]
        multiplier = units[suffix]
    else:
        number = value
        multiplier = 86400
    try:
        seconds = float(number) * multiplier
    except ValueError as exc:
        raise MCodeError(
            what=f"invalid --older-than value {value!r}",
            why="expected a number with optional suffix s, m, h, d, or w",
            next="examples: --older-than 7d, --older-than 12h",
        ) from exc
    return time.time() - seconds


def _prune_reason(run: RunRecord, *, missing_db: bool, cutoff: float | None) -> str:
    reasons: list[str] = []
    if missing_db and _db_missing(run):
        reasons.append("missing db")
    if cutoff is not None and float(run.started_at or 0.0) <= cutoff:
        reasons.append("old")
    return f"({', '.join(reasons)})" if reasons else ""


def _format_short_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
    except (OSError, ValueError):
        return str(ts)


def _format_progress(p: dict, *, max_len: int | None = None) -> str:
    if not p:
        return "—"
    cur = p.get("current", 0)
    total = p.get("total", 0)
    head = f"{cur}/{total}" if total else str(cur)
    task_id = p.get("task_id")
    stage = p.get("stage")
    details = [str(value) for value in (task_id, stage) if value]
    text = f"{head} {' · '.join(details)}" if details else head
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _details_table():
    from rich.table import Table

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True, justify="right")
    table.add_column(ratio=1, overflow="fold")
    return table


def _add_detail(table, key: str, value: str) -> None:
    from rich.text import Text

    table.add_row(f"{key}:", Text(str(value), overflow="fold"))


def _add_remote_details(table, run: RunRecord) -> None:
    for key in (
        "login",
        "run_dir",
        "remote_db",
        "remote_log",
        "remote_script",
        "remote_artifact_dir",
        "local_artifact_dir",
    ):
        value = run.remote.get(key)
        if value:
            _add_detail(table, key, str(value))


def _print_command(command: str) -> None:
    console.print(f"  {command}", overflow="fold")


def _run_payload(run: RunRecord) -> dict:
    return {
        "id": run.id,
        "benchmark": run.benchmark,
        "target": run.target.value,
        "status": run.status.value,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "db_path": run.db_path,
        "progress": run.progress,
        "remote": run.remote,
        "metadata": run.metadata,
    }


def _result_info(run: RunRecord) -> dict:
    db_path = Path(run.db_path) if run.db_path else None
    if db_path is None or not db_path.exists():
        return {}
    try:
        with ResultsDB(db_path) as rdb:
            run_id = _results_run_id(run, rdb)
            if run_id is None:
                return {}
            summary = rdb.run_summary(run_id)
            failures = [
                dict(row)
                for row in rdb.conn.execute(
                    """
                    SELECT task_id, error, terminal_reason, timed_out, exit_code
                    FROM task_results
                    WHERE run_id = ? AND NOT passed
                    ORDER BY task_id
                    LIMIT 20
                    """,
                    (run_id,),
                ).fetchall()
            ]
    except (OSError, sqlite3.Error):
        return {}
    return {
        "summary": {
            "run_id": summary.run_id,
            "total": summary.total,
            "passed": summary.passed,
            "pass_rate": summary.pass_rate,
        },
        "failures": failures,
    }


def _results_run_id(run: RunRecord, rdb: ResultsDB) -> int | None:
    raw = (run.metadata or {}).get("results_run_id")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return None


__all__ = ["cancel_run", "list_runs", "prune_runs", "show_run"]
