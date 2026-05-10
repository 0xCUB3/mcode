"""Phase-list progress UI with heartbeat.

Design invariants for the phase-based progress UI with heartbeat:

- Each target declares a list[Phase]. Exactly one phase is active at a time.
- The active phase has a background heartbeat thread that refreshes its detail
  line on an adaptive schedule: 2 Hz for the first 30 s, 0.5 Hz from 30 s-2 min,
  0.1 Hz afterward. Queue-like phases can opt into 0.1 Hz from the start.
- Phase transitions come from explicit caller signals (start/finish). Log
  markers or bjobs changes decorate the detail line but never drive transitions.
- Transport failures render distinctly: detail becomes "⚠ ssh unreachable ..."
  rather than being mistaken for "remote quiet".
- --json mode: emit one line per transition + every 30 s during a phase, not
  every heartbeat tick (m2).
- No percentages. No rescaling.

Usage:

    reporter = RichReporter.create(PHASES)
    with reporter:
        with reporter.phase("submit", feed=lambda: f"elapsed {reporter.elapsed():.0f}s"):
            ...
        with reporter.phase("queued", feed=queue_detail, mode="slow"):
            ...

`feed` is a callable returning a short string, called at the backoff rate.
Raising ``TransportError`` from the feed renders the warning and keeps the
heartbeat alive for the next tick.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Protocol

from mcode.launch.models import Phase, PhaseStatus


class TransportError(Exception):
    """Raised by a feed when the underlying transport (e.g. SSH) is unreachable.

    The reporter renders this as ⚠ and keeps the phase active. A subsequent
    feed call that succeeds clears the warning.
    """


# Adaptive backoff schedule: (elapsed_s, interval_s). First row that matches
# wins. Queue-like phases use SLOW_SCHEDULE from the start.
_FAST_SCHEDULE: tuple[tuple[float, float], ...] = (
    (30.0, 0.5),  # 2 Hz for the first 30 s
    (120.0, 2.0),  # 0.5 Hz from 30 s - 2 min
    (float("inf"), 10.0),  # 0.1 Hz after 2 min
)
_SLOW_SCHEDULE: tuple[tuple[float, float], ...] = (
    (float("inf"), 10.0),  # 0.1 Hz throughout
)

_STALL_WARNING_S = 30.0  # append "(no output for Ns)" after this
_JSON_HEARTBEAT_PERIOD_S = 30  # emit JSON ticks no more often than this


def _interval_for(elapsed_s: float, schedule) -> float:
    for threshold, interval in schedule:
        if elapsed_s < threshold:
            return interval
    return schedule[-1][1]


def _now() -> float:
    return time.monotonic()


@dataclass
class _PhaseState:
    phase: Phase
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    detail: str = ""
    last_detail_change: float | None = None
    last_feed_ok: float | None = None  # last successful feed poll (even if value unchanged)
    transport_warning: str | None = None

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else (now or _now())
        return max(0.0, end - self.started_at)


class Reporter(Protocol):
    def add_phases(self, phases: list[Phase]) -> None: ...
    def start(self, key: str) -> None: ...
    def set_detail(self, text: str) -> None: ...
    def transport_warning(self, text: str | None) -> None: ...
    def finish(self, status: PhaseStatus = PhaseStatus.DONE, detail: str = "") -> None: ...
    def close(self) -> None: ...


class _ReporterBase:
    """Shared state machine + heartbeat plumbing."""

    def __init__(self, schedule: tuple[tuple[float, float], ...] = _FAST_SCHEDULE) -> None:
        self._lock = threading.RLock()
        self._phases: list[_PhaseState] = []
        self._by_key: dict[str, _PhaseState] = {}
        self._active: _PhaseState | None = None
        self._schedule = schedule
        self._current_schedule = schedule
        self._feed: Callable[[], str] | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop: threading.Event | None = None
        # _closed guards all state mutation + output paths after shutdown so
        # a late heartbeat tick (e.g. feed stuck in SSH I/O) can't lie.
        self._closed = False

    # --- public API ---------------------------------------------------------
    def add_phases(self, phases: list[Phase]) -> None:
        with self._lock:
            self._phases = [_PhaseState(phase=p) for p in phases]
            self._by_key = {p.phase.key: p for p in self._phases}
        self._render()

    def start(self, key: str, feed: Callable[[], str] | None = None, *, mode: str = "fast") -> None:
        with self._lock:
            ps = self._by_key[key]
            if self._active is not None and self._active is not ps:
                # Auto-finish the previously-active phase if the caller forgot.
                self._active.status = PhaseStatus.DONE
                self._active.finished_at = _now()
            ps.status = PhaseStatus.ACTIVE
            ps.started_at = _now()
            ps.last_detail_change = ps.started_at
            self._active = ps
            self._current_schedule = _SLOW_SCHEDULE if mode == "slow" else self._schedule
            self._feed = feed
        self._start_heartbeat()
        self._on_transition(ps)
        self._render()

    def set_detail(self, text: str) -> None:
        with self._lock:
            if self._closed or self._active is None:
                return
            if text != self._active.detail:
                self._active.detail = text
                self._active.last_detail_change = _now()
        self._render()

    def transport_warning(self, text: str | None) -> None:
        with self._lock:
            if self._closed or self._active is None:
                return
            self._active.transport_warning = text
        self._render()

    def _note_feed_ok(self) -> None:
        with self._lock:
            if self._closed or self._active is None:
                return
            self._active.last_feed_ok = _now()

    def finish(self, status: PhaseStatus = PhaseStatus.DONE, detail: str = "") -> None:
        with self._lock:
            if self._active is None:
                return
            self._active.status = status
            self._active.finished_at = _now()
            if detail:
                self._active.detail = detail
            self._active.transport_warning = None
            ps = self._active
            self._active = None
            self._feed = None
        self._stop_heartbeat()
        self._on_transition(ps)
        self._render()

    def close(self) -> None:
        self._stop_heartbeat()
        orphan: _PhaseState | None = None
        with self._lock:
            if self._active is not None:
                # Caller forgot to finish(); mark failed for safety so the
                # output doesn't lie. Capture before clearing so we can emit
                # the terminal transition below (regression).
                self._active.status = PhaseStatus.FAILED
                self._active.finished_at = _now()
                orphan = self._active
                self._active = None
            self._feed = None
            self._closed = True
        if orphan is not None:
            # Emit the synthetic failure via the same path as finish(), so
            # Plain/Json reporters see a terminal transition, not a truncated
            # run. _on_transition is responsible for ignoring output if the
            # subclass decides to — but it's given the chance.
            try:
                self._on_transition(orphan)
            except Exception:
                pass
        self._on_close()
        self._render(final=True)

    def __enter__(self) -> _ReporterBase:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- phase context helper ----------------------------------------------
    @contextmanager
    def phase(
        self, key: str, feed: Callable[[], str] | None = None, *, mode: str = "fast"
    ) -> Iterator[None]:
        self.start(key, feed=feed, mode=mode)
        try:
            yield
        except Exception:
            self.finish(PhaseStatus.FAILED)
            raise
        else:
            self.finish(PhaseStatus.DONE)

    # --- heartbeat ---------------------------------------------------------
    def _start_heartbeat(self) -> None:
        self._stop_heartbeat()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="mcode-launch-heartbeat",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        t = self._heartbeat_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        # Keep the thread ref if it didn't actually exit — releasing it would
        # just orphan a still-running thread that can still mutate state. The
        # _closed flag (set by close()) prevents it from doing damage, and the
        # daemon=True means it won't block process exit.
        if t is None or not t.is_alive():
            self._heartbeat_thread = None
            self._heartbeat_stop = None

    def _heartbeat_loop(self) -> None:
        assert self._heartbeat_stop is not None
        stop = self._heartbeat_stop
        while not stop.is_set():
            with self._lock:
                if self._closed:
                    return
                active = self._active
                feed = self._feed
                schedule = self._current_schedule
            if active is None:
                return
            elapsed = active.elapsed()
            interval = _interval_for(elapsed, schedule)
            if feed is not None:
                try:
                    text = feed()
                except TransportError as e:
                    self.transport_warning(str(e) or "ssh unreachable")
                except Exception as e:
                    # Feed errors shouldn't kill the UI; render them as
                    # warnings and keep going.
                    self.transport_warning(f"feed error: {e}")
                else:
                    # Successful poll: clear transport warning and mark feed
                    # alive — even if text is unchanged. This prevents the
                    # stall warning from triggering on a steady-state feed.
                    self.transport_warning(None)
                    self._note_feed_ok()
                    if text:
                        self.set_detail(text)
            self._render()
            if stop.wait(interval):
                return

    # --- subclass hooks ----------------------------------------------------
    def _render(self, final: bool = False) -> None:
        raise NotImplementedError

    def _on_transition(self, ps: _PhaseState) -> None:
        pass

    def _on_close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Rich (interactive TTY) reporter
# ---------------------------------------------------------------------------
class RichReporter(_ReporterBase):
    """Renders phase list via rich.live.Live. Falls back to PlainReporter if
    rich is unavailable or stdout isn't a TTY."""

    _SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, stream: IO[str] | None = None) -> None:
        super().__init__()
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=stream or sys.stderr, force_terminal=True)
        self._live = Live(
            self._render_table(), console=self._console, refresh_per_second=10, transient=False
        )
        self._live.start()
        self._tick = 0

    @classmethod
    def create(cls, phases: list[Phase], *, stream: IO[str] | None = None) -> RichReporter:
        r = cls(stream=stream)
        r.add_phases(phases)
        return r

    def _render(self, final: bool = False) -> None:
        self._tick += 1
        try:
            self._live.update(self._render_table())
        except Exception:
            pass
        if final:
            try:
                self._live.stop()
            except Exception:
                pass

    def _render_table(self):
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 1))
        table.add_column()  # icon
        table.add_column()  # label
        table.add_column()  # detail
        with self._lock:
            width = shutil.get_terminal_size((120, 20)).columns
            detail_budget = max(20, width - 40)
            spinner = self._SPINNER[self._tick % len(self._SPINNER)]
            for ps in self._phases:
                icon = Text(self._icon(ps, spinner), style=self._icon_style(ps))
                label = Text(ps.phase.label)
                detail = Text(self._detail_for(ps, detail_budget), style=self._detail_style(ps))
                table.add_row(icon, label, detail)
        return table

    @staticmethod
    def _icon(ps: _PhaseState, spinner: str) -> str:
        if ps.status == PhaseStatus.ACTIVE:
            return spinner
        if ps.status == PhaseStatus.DONE:
            return "✓"
        if ps.status == PhaseStatus.FAILED:
            return "✗"
        return "·"

    @staticmethod
    def _icon_style(ps: _PhaseState) -> str:
        return {
            PhaseStatus.ACTIVE: "cyan",
            PhaseStatus.DONE: "green",
            PhaseStatus.FAILED: "red",
            PhaseStatus.PENDING: "dim",
        }[ps.status]

    @staticmethod
    def _detail_style(ps: _PhaseState) -> str:
        if ps.transport_warning:
            return "yellow"
        if ps.status == PhaseStatus.FAILED:
            return "red"
        if ps.status == PhaseStatus.DONE:
            return "green"
        return "dim"

    def _detail_for(self, ps: _PhaseState, budget: int) -> str:
        if ps.status == PhaseStatus.PENDING:
            return ""
        elapsed = ps.elapsed()
        if ps.status == PhaseStatus.ACTIVE:
            if ps.transport_warning:
                body = f"⚠ {ps.transport_warning}"
            else:
                body = ps.detail or "…"
                # Stall = no successful FEED POLL for N seconds — not "same
                # text repeated", which is a healthy steady-state signal.
                # last_feed_ok may be None if no feed was supplied for this
                # phase; fall back to last_detail_change in that case.
                anchor = ps.last_feed_ok if ps.last_feed_ok is not None else ps.last_detail_change
                if anchor is not None:
                    stale = _now() - anchor
                    if stale >= _STALL_WARNING_S and body and body != "…":
                        body = f"{body}  (no output for {int(stale)}s)"
            text = f"{elapsed:5.1f}s  {body}"
        else:
            text = f"{elapsed:5.1f}s  {ps.detail}"
        if len(text) > budget:
            text = text[: budget - 1] + "…"
        return text


# ---------------------------------------------------------------------------
# Plain (no-TTY) and JSON reporters
# ---------------------------------------------------------------------------
class PlainReporter(_ReporterBase):
    """Prints phase transitions + periodic one-line updates. For when stdout
    isn't a TTY or Rich is unavailable."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        super().__init__()
        self._stream = stream or sys.stderr
        self._last_print: dict[str, tuple[str, float]] = {}

    @classmethod
    def create(cls, phases: list[Phase], *, stream: IO[str] | None = None) -> PlainReporter:
        r = cls(stream=stream)
        r.add_phases(phases)
        return r

    def _render(self, final: bool = False) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
            key = active.phase.key
            last = self._last_print.get(key, ("", 0.0))
            line = self._line(active)
            now = _now()
            if line != last[0] or (now - last[1]) >= 2.0:
                print(line, file=self._stream, flush=True)
                self._last_print[key] = (line, now)

    def _on_transition(self, ps: _PhaseState) -> None:
        marker = {
            PhaseStatus.ACTIVE: "▶",
            PhaseStatus.DONE: "✓",
            PhaseStatus.FAILED: "✗",
            PhaseStatus.PENDING: "·",
        }[ps.status]
        print(f"{marker} {ps.phase.label} ({ps.elapsed():.1f}s)", file=self._stream, flush=True)

    def _line(self, ps: _PhaseState) -> str:
        body = f"⚠ {ps.transport_warning}" if ps.transport_warning else (ps.detail or "…")
        return f"  [{ps.phase.key} {ps.elapsed():.1f}s] {body}"


class JsonReporter(_ReporterBase):
    """Emits one JSON object per phase transition, plus one every 30 s during
    long phases (m2 cadence)."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        super().__init__()
        self._stream = stream or sys.stdout
        self._last_emit: dict[str, float] = {}

    @classmethod
    def create(cls, phases: list[Phase], *, stream: IO[str] | None = None) -> JsonReporter:
        r = cls(stream=stream)
        r.add_phases(phases)
        return r

    def _render(self, final: bool = False) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
        self._maybe_emit(active, force=False)

    def _on_transition(self, ps: _PhaseState) -> None:
        self._maybe_emit(ps, force=True)

    def _maybe_emit(self, ps: _PhaseState, *, force: bool) -> None:
        now = _now()
        last = self._last_emit.get(ps.phase.key, 0.0)
        if not force and (now - last) < _JSON_HEARTBEAT_PERIOD_S:
            return
        self._last_emit[ps.phase.key] = now
        payload = {
            "phase": ps.phase.key,
            "label": ps.phase.label,
            "status": ps.status.value,
            "elapsed_s": round(ps.elapsed(now), 2),
            "detail": ps.detail or "",
            "transport_warning": ps.transport_warning,
            "ts": time.time(),
        }
        print(json.dumps(payload), file=self._stream, flush=True)


class NullReporter(_ReporterBase):
    """For tests and --no-progress mode. Swallows everything."""

    @classmethod
    def create(cls, phases: list[Phase]) -> NullReporter:
        r = cls()
        r.add_phases(phases)
        return r

    def _render(self, final: bool = False) -> None:
        pass


def choose(
    phases: list[Phase], *, json_mode: bool = False, stream: IO[str] | None = None
) -> _ReporterBase:
    """Select a reporter. JSON mode wins; otherwise Rich if stderr is a TTY,
    else Plain."""
    if json_mode:
        return JsonReporter.create(phases, stream=stream)
    out = stream or sys.stderr
    is_tty = bool(getattr(out, "isatty", lambda: False)())
    if is_tty:
        try:
            return RichReporter.create(phases, stream=stream)
        except Exception:
            pass
    return PlainReporter.create(phases, stream=stream)
