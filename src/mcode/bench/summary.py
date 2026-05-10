"""Human-facing benchmark summaries and hints."""

from __future__ import annotations

import os
import shlex
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.table import Table

from mcode.bench.results import ResultsDB, RunSummary
from mcode.ui.console import console

_PROGRESS_ENV_NAMES = (
    "OPENAI_BASE_URL",
    "MCODE_CONTEXT_WINDOW",
    "MCODE_MAX_NEW_TOKENS",
    "MCODE_REACT_TIMEOUT",
    "MCODE_LIVE_TRACE",
)


@dataclass(frozen=True)
class RunPlan:
    benchmark: str
    backend: str
    model: str
    db: Path
    loop_budget: int
    timeout_s: int
    phase: str = "run"
    location: str = "local"
    artifact_dir: Path | None = None
    limit: int | None = None
    task_ids: str | None = None
    shards: int | None = None
    shard_count: int | None = None
    shard_index: int | None = None


def print_run_plan(plan: RunPlan) -> None:
    selector = _task_selector(plan)
    parts = [
        f"▶ {plan.benchmark} on {plan.location}",
        f"model={plan.model}",
        f"backend={plan.backend}",
        f"phase={plan.phase}",
        f"budget={plan.loop_budget}",
        f"timeout={plan.timeout_s}s",
        f"tasks={selector}",
        f"db={plan.db}",
    ]
    if plan.artifact_dir is not None:
        parts.append(f"artifacts={plan.artifact_dir}")
    console.print(" ".join(parts))


def print_run_summary(
    *,
    summary: RunSummary,
    benchmark: str,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
) -> None:
    table = Table(title="Run summary")
    table.add_column("run_id", justify="right")
    table.add_column("benchmark")
    table.add_column("backend")
    table.add_column("model")
    table.add_column("budget", justify="right")
    table.add_column("timeout", justify="right")
    table.add_column("total", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_row(
        str(summary.run_id),
        benchmark,
        backend,
        model,
        str(loop_budget),
        str(timeout_s),
        str(summary.total),
        str(summary.passed),
        f"{summary.pass_rate:.1%}",
    )
    console.print(table)


def print_run_footer(*, db: Path, summary: RunSummary, task_time_ms: int | None = None) -> None:
    duration = f" task_time={_format_duration_ms(task_time_ms)}" if task_time_ms else ""
    console.print(
        f"passed {summary.passed}/{summary.total} ({summary.pass_rate:.1%}){duration}\ndb={db}"
    )


def print_failure_hints(*, db: Path, run_id: int, max_rows: int = 8) -> None:
    rows = _failure_rows(db, run_id, max_rows=max_rows)
    if not rows:
        return
    console.print("failed tasks:")
    for row in rows:
        reason = row.terminal_reason or _short_error(row.error) or "failed"
        if row.timed_out:
            reason = f"timeout; {reason}"
        console.print(f"  - {row.task_id}: {reason}")
    hints = _hints_for_failures(rows)
    if hints:
        console.print("hints:")
        for hint in hints:
            console.print(f"  - {hint}")


def task_time_ms(db: Path, run_id: int) -> int | None:
    try:
        with ResultsDB(db) as rdb:
            row = rdb.conn.execute(
                "SELECT COALESCE(SUM(time_ms), 0) AS total_ms FROM task_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    if row is None:
        return None
    return int(row["total_ms"] or 0)


def safe_rerun_metadata() -> dict[str, Any]:
    env = {name: os.environ[name] for name in _PROGRESS_ENV_NAMES if os.environ.get(name)}
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    return {"command": command, "env": env}


@dataclass(frozen=True)
class _FailureRow:
    task_id: str
    error: str | None
    terminal_reason: str | None
    timed_out: bool
    exit_code: int | None


def _failure_rows(db: Path, run_id: int, *, max_rows: int) -> list[_FailureRow]:
    try:
        with ResultsDB(db) as rdb:
            rows = rdb.conn.execute(
                """
                SELECT task_id, error, terminal_reason, timed_out, exit_code
                FROM task_results
                WHERE run_id = ? AND NOT passed
                ORDER BY task_id
                LIMIT ?
                """,
                (run_id, max_rows),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return []
    return [
        _FailureRow(
            task_id=str(row["task_id"]),
            error=row["error"],
            terminal_reason=row["terminal_reason"],
            timed_out=bool(row["timed_out"]),
            exit_code=row["exit_code"],
        )
        for row in rows
    ]


def _hints_for_failures(rows: list[_FailureRow]) -> list[str]:
    hints: list[str] = []
    text = "\n".join(
        str(value or "")
        for row in rows
        for value in (row.error, row.terminal_reason, "timeout" if row.timed_out else "")
    ).lower()
    if "dockerunavailableerror" in text or "docker is required" in text:
        hints.append("Start Docker Desktop or set DOCKER_HOST, then rerun the same command.")
    if "budget_exhausted" in text:
        hints.append("The model ran out of turns; retry with a larger --loop-budget.")
    if "unverified_diff_discarded" in text:
        hints.append("The model edited but did not produce a verified patch; try a larger budget.")
    if "no patch" in text or "zero_edit" in text:
        hints.append("No usable patch was produced; inspect progress logs or try a stronger model.")
    if "aider polyglot benchmark not found" in text:
        hints.append("Clone the benchmark repo or pass --benchmark-root.")
    if any(row.timed_out for row in rows):
        hints.append(
            "At least one evaluation timed out; increase --timeout if the task is still running."
        )
    return list(dict.fromkeys(hints))


def _short_error(error: str | None) -> str | None:
    if not error:
        return None
    line = " ".join(error.strip().split())
    return line[:160]


def _format_duration_ms(ms: int | None) -> str:
    if not ms:
        return "0s"
    seconds = max(0, int(round(ms / 1000)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _task_selector(plan: RunPlan) -> str:
    if plan.task_ids:
        parsed = [item.strip() for item in plan.task_ids.split(",") if item.strip()]
        if len(parsed) > 3:
            return f"{', '.join(parsed[:3])}, … ({len(parsed)} explicit)"
        return ",".join(parsed)
    if plan.limit is not None:
        return f"first {plan.limit}"
    if plan.shards:
        return f"all via {plan.shards} shards"
    if plan.shard_count:
        return f"shard {plan.shard_index or 0}/{plan.shard_count}"
    return "all selected"


__all__ = [
    "RunPlan",
    "print_failure_hints",
    "print_run_footer",
    "print_run_plan",
    "print_run_summary",
    "safe_rerun_metadata",
    "task_time_ms",
]
