"""Sharded-bench dashboard: single writer thread + per-shard state.

The dashboard owns one queue.Queue and one writer thread. Every event
source — the parent orchestrator, per-shard stdout streamers, infra-failure
detector — calls Dashboard.post(...) which is non-blocking. The writer
thread drains the queue, increments a monotonic seq counter, updates
internal per-shard state, and emits to the chosen renderer:

- Rich (TTY): a `rich.live.Live` table that updates in place. Multi-shard
  rows + an overall row with ok/fail/infra counters and a running ETA.
- Plain (non-TTY / CI / MCODE_NO_TTY=1): one human-readable line per
  state change. Format-equivalent (not byte-equivalent) to the
  pre-Wave-2 _run_sharded_benchmark output.
- Json (`--json`): one JSON object per line with strictly monotonic seq.

Threading invariants:
- Writer thread is a daemon; main thread joins it via close() in finally.
- A second Ctrl+C during teardown does not deadlock — the queue has a
  timeout-based get(), and close() drains best-effort.
- post() is lock-free against the writer; only the queue is shared state
  between producers and consumer.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, Any, Literal

EventKind = Literal[
    "run_start",
    "shard_start",
    "shard_stdout",
    "shard_done",
    "shard_failed",
    "shard_infra",
    "infra_failure",
    "merged",
    "summary",
    "remote_stdout",
    "info",
]


@dataclass
class DashEvent:
    seq: int
    ts: float
    kind: EventKind
    shard: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ShardState:
    index: int
    started_at: float | None = None
    ended_at: float | None = None
    db_path: str | None = None
    log_path: str | None = None
    last_line: str = ""
    status: Literal["pending", "running", "done", "failed", "infra"] = "pending"


class Dashboard:
    """Owner of the writer thread and per-shard state."""

    def __init__(
        self,
        *,
        mode: Literal["rich", "plain", "json"],
        total_shards: int,
        benchmark: str,
        model: str,
        stream: IO[str] | None = None,
    ) -> None:
        self._mode = mode
        self._stream = stream or sys.stderr
        self._benchmark = benchmark
        self._model = model
        self._total_shards = total_shards
        self._shards: dict[int, _ShardState] = {
            i: _ShardState(index=i) for i in range(total_shards)
        }
        self._counters = {"ok": 0, "fail": 0, "infra": 0, "completed": 0}
        # Unbounded queue so a producer burst never silently drops events.
        # Memory cost is bounded by total events posted in a run.
        self._queue: queue.Queue[DashEvent | None] = queue.Queue()
        self._seq = 0
        # _seq_lock guards the increment AND the enqueue together so that
        # seq order matches enqueue order under concurrent producers.
        self._seq_lock = threading.Lock()
        # _live_lock serializes Rich Live access between the writer thread
        # (live.update) and close() (live.stop) to prevent the closing race.
        self._live_lock = threading.Lock()
        self._started_at = time.monotonic()
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None
        self._closed = False
        # Renderer-specific state. Rich Live is constructed lazily so plain
        # callers don't pay for it.
        self._rich_console: Any | None = None
        self._rich_live: Any | None = None

    # ---- producer-side API -------------------------------------------------
    def post(self, kind: EventKind, *, shard: int | None = None, **data: Any) -> None:
        """Submit an event. Queue is unbounded so this never blocks or drops.
        seq increment + enqueue happen under one lock so order is monotonic
        across concurrent producers."""
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
            evt = DashEvent(seq=seq, ts=time.time(), kind=kind, shard=shard, data=dict(data))
            self._queue.put_nowait(evt)

    def patch_shard_meta(
        self, shard: int, *, db: str | None = None, log: str | None = None
    ) -> None:
        """Update bookkeeping fields without queuing an event (used by the
        orchestrator right after subprocess.Popen so the table can show the
        shard's db/log paths even though those don't deserve a JSON event)."""
        s = self._shards.get(shard)
        if s is None:
            return
        if db is not None:
            s.db_path = db
        if log is not None:
            s.log_path = log

    # ---- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Dashboard:
        self.post(
            "run_start", benchmark=self._benchmark, model=self._model, shards=self._total_shards
        )
        if self._mode == "rich":
            self._open_rich()
        self._writer = threading.Thread(
            target=self._writer_loop, daemon=True, name="mcode-bench-dashboard"
        )
        self._writer.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Sentinel tells the writer to drain and exit.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._stop.set()
        t = self._writer
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        # Stop Rich Live under the same lock the writer uses for update(),
        # so we can't race against a concurrent in-flight update.
        with self._live_lock:
            if self._rich_live is not None:
                try:
                    self._rich_live.stop()
                except Exception:
                    pass
                self._rich_live = None

    # ---- writer thread -----------------------------------------------------
    def _writer_loop(self) -> None:
        while True:
            try:
                evt = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                if self._mode == "rich":
                    self._render_rich()
                continue
            if evt is None:
                # Drain remaining events before exit.
                self._drain_remaining()
                if self._mode == "rich":
                    self._render_rich(final=True)
                return
            try:
                self._apply(evt)
                self._render(evt)
            except Exception:
                # Renderer errors must never kill the writer thread.
                pass

    def _drain_remaining(self) -> None:
        while True:
            try:
                evt = self._queue.get_nowait()
            except queue.Empty:
                return
            if evt is None:
                continue
            try:
                self._apply(evt)
                self._render(evt)
            except Exception:
                pass

    # ---- state machine -----------------------------------------------------
    def _apply(self, evt: DashEvent) -> None:
        s = self._shards.get(evt.shard) if evt.shard is not None else None
        if evt.kind == "shard_start" and s is not None:
            s.started_at = time.monotonic()
            s.status = "running"
            if "db" in evt.data:
                s.db_path = evt.data["db"]
            if "log" in evt.data:
                s.log_path = evt.data["log"]
        elif evt.kind == "shard_stdout" and s is not None:
            line = str(evt.data.get("line", ""))
            if line:
                s.last_line = line[:200]
        elif evt.kind == "shard_done" and s is not None:
            s.ended_at = time.monotonic()
            s.status = "done"
            self._counters["completed"] += 1
            self._counters["ok"] += 1
        elif evt.kind == "shard_failed" and s is not None:
            s.ended_at = time.monotonic()
            s.status = "failed"
            self._counters["completed"] += 1
            self._counters["fail"] += 1
        elif evt.kind == "shard_infra" and s is not None:
            s.ended_at = time.monotonic()
            s.status = "infra"
            self._counters["completed"] += 1
            self._counters["infra"] += 1

    # ---- renderers ---------------------------------------------------------
    def _render(self, evt: DashEvent) -> None:
        if self._mode == "json":
            self._render_json(evt)
        elif self._mode == "plain":
            self._render_plain(evt)
        else:
            self._render_rich()

    def _render_json(self, evt: DashEvent) -> None:
        import json

        payload: dict[str, Any] = {
            "seq": evt.seq,
            "ts": evt.ts,
            "kind": evt.kind,
        }
        if evt.shard is not None:
            payload["shard"] = evt.shard
        if evt.data:
            payload["data"] = evt.data
        print(json.dumps(payload, sort_keys=True), file=self._stream, flush=True)

    def _render_plain(self, evt: DashEvent) -> None:
        # Lines preserved verbatim (or near-verbatim) from the pre-Wave-2
        # _run_sharded_benchmark output so existing log scrapers stay valid.
        # run_start carries data but doesn't print in plain mode — the
        # orchestrator prints the equivalent "▶ sharded run command=..."
        # info line, matching the pre-Wave-2 format exactly.
        if evt.kind == "run_start":
            return
        if evt.kind == "shard_start":
            s = evt.shard
            d = evt.data
            print(
                f"▶ shard {(s or 0) + 1}/{self._total_shards} db={d.get('db')} log={d.get('log')}",
                file=self._stream,
                flush=True,
            )
        elif evt.kind == "shard_stdout":
            s = evt.shard
            line = str(evt.data.get("line", "")).rstrip()
            if line:
                print(f"[shard {s}] {line}", file=self._stream, flush=True)
        elif evt.kind == "shard_done":
            s = evt.shard
            print(
                f"✓ shard {(s or 0) + 1}/{self._total_shards} finished",
                file=self._stream,
                flush=True,
            )
        elif evt.kind == "shard_failed":
            s = evt.shard
            d = evt.data
            print(
                f"✗ shard {(s or 0) + 1}/{self._total_shards} failed "
                f"exit={d.get('rc')} log={d.get('log')}",
                file=self._stream,
                flush=True,
            )
        elif evt.kind == "shard_infra":
            s = evt.shard
            d = evt.data
            print(
                f"✗ shard {(s or 0) + 1}/{self._total_shards} hit retryable infra "
                f"exit={d.get('rc')} log={d.get('log')}",
                file=self._stream,
                flush=True,
            )
        elif evt.kind == "infra_failure":
            d = evt.data
            print(
                "✗ infra failure detected "
                f"task={d.get('task_id')} db={d.get('db')} "
                f"reason={d.get('reason')}: {d.get('detail') or ''}",
                file=self._stream,
                flush=True,
            )
        elif evt.kind == "merged":
            print(f"✓ merged shards into {evt.data.get('db')}", file=self._stream, flush=True)
        elif evt.kind == "summary":
            print(evt.data.get("text", ""), file=self._stream, flush=True)
        elif evt.kind == "remote_stdout":
            line = str(evt.data.get("line", "")).rstrip()
            if line:
                print(line, file=self._stream, flush=True)
        elif evt.kind == "info":
            print(evt.data.get("text", ""), file=self._stream, flush=True)

    # ---- rich -------------------------------------------------------------
    def _open_rich(self) -> None:
        from rich.console import Console
        from rich.live import Live

        self._rich_console = Console(file=self._stream, force_terminal=True)
        self._rich_live = Live(
            self._build_rich_table(),
            console=self._rich_console,
            refresh_per_second=4,
            transient=False,
        )
        self._rich_live.start()

    def _render_rich(self, *, final: bool = False) -> None:
        with self._live_lock:
            if self._rich_live is None:
                return
            try:
                self._rich_live.update(self._build_rich_table())
            except Exception:
                pass

    def _build_rich_table(self) -> Any:
        from rich.table import Table

        t = Table(title=f"{self._benchmark} · {self._model} · {self._total_shards} shards")
        t.add_column("shard")
        t.add_column("status")
        t.add_column("last")
        t.add_column("elapsed", justify="right")
        for idx in range(self._total_shards):
            s = self._shards[idx]
            if s.started_at is None:
                elapsed = "—"
            else:
                end = s.ended_at if s.ended_at is not None else time.monotonic()
                elapsed = f"{end - s.started_at:.0f}s"
            symbol = {
                "pending": "·",
                "running": "▶",
                "done": "✓",
                "failed": "✗",
                "infra": "⚠",
            }.get(s.status, "?")
            color = {
                "pending": "dim",
                "running": "cyan",
                "done": "green",
                "failed": "red",
                "infra": "yellow",
            }.get(s.status, "white")
            t.add_row(
                str(idx),
                f"[{color}]{symbol} {s.status}[/{color}]",
                s.last_line[:80],
                elapsed,
            )

        c = self._counters
        eta = self._eta()
        eta_str = f" · est {eta}" if eta else ""
        t.caption = (
            f"overall ok {c['ok']}  fail {c['fail']}  infra {c['infra']}  "
            f"completed {c['completed']}/{self._total_shards}{eta_str}"
        )
        return t

    def _eta(self) -> str:
        # Naive ETA. Only shown after >=1 shard has completed.
        completed = self._counters["completed"]
        if completed < 1 or completed >= self._total_shards:
            return ""
        elapsed = time.monotonic() - self._started_at
        rate = elapsed / completed
        remaining_s = rate * (self._total_shards - completed)
        if remaining_s < 60:
            return f"{remaining_s:.0f}s remaining"
        return f"{remaining_s / 60:.0f}m remaining"


@contextmanager
def open_dashboard(
    *,
    json_mode: bool,
    total_shards: int,
    benchmark: str,
    model: str,
    stream: IO[str] | None = None,
) -> Iterator[Dashboard]:
    """Pick mode and yield a Dashboard. Selection mirrors task_reporter.choose:
    json wins, else Rich on TTY, else Plain."""
    out = stream or sys.stderr
    if json_mode:
        mode: Literal["rich", "plain", "json"] = "json"
    else:
        is_tty = bool(getattr(out, "isatty", lambda: False)())
        mode = "rich" if is_tty else "plain"
    dash = Dashboard(
        mode=mode,
        total_shards=total_shards,
        benchmark=benchmark,
        model=model,
        stream=stream,
    )
    with dash:
        yield dash


__all__ = ["DashEvent", "Dashboard", "EventKind", "open_dashboard"]
