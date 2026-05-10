"""Counter-style progress UI for bench-shaped work."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import IO, Any, Literal, Protocol

EventLevel = Literal["ok", "warn", "fail", "info"]


@dataclass
class Event:
    """One state change. Posted to the dashboard writer thread; rendered to
    the chosen reporter; serialized in --json mode with monotonic seq."""

    seq: int
    ts: float
    kind: str
    shard: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class TaskReporter(Protocol):
    def total(self, n: int) -> None: ...
    def advance(self, n: int = 1, *, detail: str = "") -> None: ...
    def event(self, level: EventLevel, text: str, *, shard: int | None = None) -> None: ...
    def finish(self, *, ok: bool = True, summary: str = "") -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> TaskReporter: ...
    def __exit__(self, *exc: Any) -> None: ...


class _Base:
    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stderr
        self._lock = threading.RLock()
        self._total: int | None = None
        self._current = 0
        self._closed = False
        self._started_at = time.monotonic()
        self._seq = 0

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def __enter__(self) -> _Base:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def total(self, n: int) -> None:
        with self._lock:
            self._total = n
        self._emit(Event(seq=self._next_seq(), ts=time.time(), kind="total", data={"n": n}))

    def advance(self, n: int = 1, *, detail: str = "") -> None:
        with self._lock:
            if self._closed:
                return
            self._current += n
            current = self._current
            total = self._total
        self._emit(
            Event(
                seq=self._next_seq(),
                ts=time.time(),
                kind="advance",
                data={"current": current, "total": total, "detail": detail},
            )
        )

    def event(self, level: EventLevel, text: str, *, shard: int | None = None) -> None:
        if self._closed:
            return
        self._emit(
            Event(
                seq=self._next_seq(),
                ts=time.time(),
                kind="event",
                shard=shard,
                data={"level": level, "text": text},
            )
        )

    def finish(self, *, ok: bool = True, summary: str = "") -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._emit(
            Event(
                seq=self._next_seq(),
                ts=time.time(),
                kind="finish",
                data={"ok": ok, "summary": summary},
            )
        )

    def close(self) -> None:
        # Idempotent. Subclasses override for terminal cleanup; the default
        # implementation just makes sure finish() was called.
        with self._lock:
            if self._closed:
                return
            self._closed = True

    # Subclass hook.
    def _emit(self, event: Event) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class NullReporter(_Base):
    """Drop everything. Used in tests."""

    def _emit(self, event: Event) -> None:
        pass


class JsonReporter(_Base):
    """One JSON object per event, line-delimited. Strictly monotonic seq."""

    def _emit(self, event: Event) -> None:
        payload = asdict(event)
        payload = {k: v for k, v in payload.items() if v is not None and v != {}}
        print(json.dumps(payload, sort_keys=True), file=self._stream, flush=True)


class PlainReporter(_Base):
    """Non-TTY output. One readable line per event."""

    def _emit(self, event: Event) -> None:
        if event.kind == "total":
            print(f"-- total: {event.data.get('n')}", file=self._stream, flush=True)
        elif event.kind == "advance":
            current = event.data.get("current")
            total = event.data.get("total")
            detail = event.data.get("detail") or ""
            tail = f" — {detail}" if detail else ""
            print(f"  [{current}/{total}]{tail}", file=self._stream, flush=True)
        elif event.kind == "event":
            level = event.data.get("level", "info")
            text = event.data.get("text", "")
            symbol = {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "·"}.get(level, "·")
            shard = event.shard
            tag = f"[shard {shard}] " if shard is not None else ""
            print(f"{symbol} {tag}{text}", file=self._stream, flush=True)
        elif event.kind == "finish":
            ok = event.data.get("ok", True)
            summary = event.data.get("summary", "")
            symbol = "✓" if ok else "✗"
            print(f"{symbol} {summary}".rstrip(), file=self._stream, flush=True)


class RichReporter(_Base):
    """TTY output with Rich colors and the same line shape as PlainReporter."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        super().__init__(stream=stream)
        from rich.console import Console

        self._console = Console(file=self._stream, force_terminal=True)

    def _emit(self, event: Event) -> None:
        if event.kind == "advance":
            current = event.data.get("current")
            total = event.data.get("total")
            detail = event.data.get("detail") or ""
            tail = f" [dim]{detail}[/dim]" if detail else ""
            self._console.print(f"  [bold cyan][{current}/{total}][/bold cyan]{tail}")
        elif event.kind == "event":
            level = event.data.get("level", "info")
            text = event.data.get("text", "")
            color = {"ok": "green", "warn": "yellow", "fail": "red", "info": "dim"}.get(
                level, "dim"
            )
            symbol = {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "·"}.get(level, "·")
            shard = event.shard
            tag = f"[shard {shard}] " if shard is not None else ""
            self._console.print(f"[{color}]{symbol}[/{color}] {tag}{text}")
        elif event.kind == "finish":
            ok = event.data.get("ok", True)
            summary = event.data.get("summary", "")
            color = "green" if ok else "red"
            symbol = "✓" if ok else "✗"
            self._console.print(f"[{color}]{symbol}[/{color}] {summary}".rstrip())
        elif event.kind == "total":
            n = event.data.get("n")
            self._console.print(f"[dim]-- total: {n}[/dim]")


def choose(
    *,
    json_mode: bool = False,
    stream: IO[str] | None = None,
) -> _Base:
    """Pick a reporter.

    JSON mode wins. Otherwise Rich if stderr is a TTY, else Plain. Rich
    construction failure falls back to Plain.
    """
    if json_mode:
        return JsonReporter(stream=stream)
    out = stream or sys.stderr
    is_tty = bool(getattr(out, "isatty", lambda: False)())
    if is_tty:
        try:
            return RichReporter(stream=stream)
        except Exception:
            pass
    return PlainReporter(stream=stream)


__all__ = [
    "Event",
    "EventLevel",
    "JsonReporter",
    "NullReporter",
    "PlainReporter",
    "RichReporter",
    "TaskReporter",
    "choose",
]
