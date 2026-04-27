"""Live `mcode watch` dashboard combining servers + runs.

Refreshes every 2s. Quits cleanly on Ctrl+C. On state.load() failure
(json.JSONDecodeError, OSError) renders the last-known-good snapshot with a
warning footer and retries on the next tick — never crashes from a transient
lock contention or partial write.
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table

from mcode.launch import state as launch_state
from mcode.launch.state import State

_REFRESH_S = 2.0


def watch(*, refresh_s: float = _REFRESH_S, once: bool = False) -> int:
    """Run the live dashboard. `once=True` renders one frame and returns,
    used by tests."""
    console = Console()
    last_good: State | None = None
    last_error: str | None = None

    if once:
        s, err = _safe_load(last_good)
        console.print(_render(s, err))
        return 0

    try:
        with Live(_render(last_good, last_error), console=console, refresh_per_second=2) as live:
            while True:
                s, err = _safe_load(last_good)
                if err is None:
                    last_good = s
                    last_error = None
                else:
                    last_error = err
                # Render and update are wrapped so a transient Rich error
                # (terminal resize, broken stream) doesn't kill the watch
                # loop. The next tick will retry; if the failure persists
                # the user sees the warning footer.
                try:
                    live.update(_render(last_good, last_error))
                except Exception as e:
                    last_error = f"render failed: {e}"
                time.sleep(refresh_s)
    except KeyboardInterrupt:
        return 0


def _safe_load(last_good: State | None) -> tuple[State, str | None]:
    try:
        return launch_state.load(), None
    except Exception as e:
        # Keep showing the last good snapshot rather than crashing.
        if last_good is not None:
            return last_good, f"state read failed: {e}"
        return State(servers=[], runs=[]), f"state read failed: {e}"


def _render(s: State | None, error: str | None) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column()

    servers_table = Table(title=f"servers ({len(s.servers) if s else 0})")
    servers_table.add_column("id")
    servers_table.add_column("target")
    servers_table.add_column("model")
    servers_table.add_column("status")
    servers_table.add_column("endpoint")
    if s:
        for srv in s.servers:
            servers_table.add_row(
                srv.id,
                srv.target.value,
                srv.model,
                _color_status(srv.status),
                srv.endpoint or "—",
            )

    runs_table = Table(title=f"runs ({len(s.runs) if s else 0})")
    runs_table.add_column("id")
    runs_table.add_column("benchmark")
    runs_table.add_column("status")
    runs_table.add_column("progress")
    runs_table.add_column("started")
    if s:
        for r in s.runs:
            runs_table.add_row(
                r.id,
                r.benchmark,
                _color_status(r.status.value),
                _format_progress(r.progress),
                _format_ts(r.started_at),
            )

    grid.add_row(servers_table)
    grid.add_row(runs_table)
    if error:
        grid.add_row(f"[yellow]⚠ {error}[/yellow]")
    grid.add_row(f"[dim]refreshing every {_REFRESH_S:.0f}s · Ctrl+C to quit[/dim]")
    return grid


def _color_status(status: str) -> str:
    color = {
        "healthy": "green",
        "running": "cyan",
        "submitted": "cyan",
        "pending": "yellow",
        "done": "green",
        "stopped": "yellow",
        "failed": "red",
    }.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _format_ts(ts) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (OSError, ValueError, TypeError):
        return str(ts)


def _format_progress(p: dict) -> str:
    if not p:
        return "—"
    cur = p.get("current", 0)
    total = p.get("total", 0)
    if total:
        return f"{cur}/{total}"
    return str(cur)


__all__ = ["watch"]
