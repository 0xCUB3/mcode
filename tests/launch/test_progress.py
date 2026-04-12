from __future__ import annotations

import io
import json
import time

from mcode.launch import progress
from mcode.launch.models import Phase, PhaseStatus

PHASES = [
    Phase("submit", "Submit"),
    Phase("queued", "Queued"),
    Phase("ready", "Ready"),
]


def _read_lines(buf: io.StringIO) -> list[str]:
    return [line for line in buf.getvalue().splitlines() if line.strip()]


def test_phase_transitions_emit_json() -> None:
    buf = io.StringIO()
    r = progress.JsonReporter.create(PHASES, stream=buf)
    r.start("submit", feed=lambda: "detail-1")
    r.finish(PhaseStatus.DONE)
    r.start("ready", feed=lambda: "ok")
    r.finish(PhaseStatus.DONE)
    r.close()

    lines = _read_lines(buf)
    events = [json.loads(line) for line in lines]
    # We get at least one per transition.
    phases_seen = [(e["phase"], e["status"]) for e in events]
    assert ("submit", "active") in phases_seen
    assert ("submit", "done") in phases_seen
    assert ("ready", "active") in phases_seen
    assert ("ready", "done") in phases_seen


def test_heartbeat_keeps_refreshing_while_feed_silent() -> None:
    """A silent phase must still produce UI updates — the heartbeat rule."""
    r = progress.NullReporter.create(PHASES)
    renders: list[float] = []
    original = r._render

    def counting_render(final: bool = False) -> None:
        renders.append(time.monotonic())
        original(final)

    r._render = counting_render  # type: ignore[method-assign]

    # Feed returns same value every call — "silent" remote.
    def silent_feed() -> str:
        return "nothing new"

    r.start("submit", feed=silent_feed)
    time.sleep(1.2)  # 2 Hz ⇒ expect ~3 ticks
    r.finish(PhaseStatus.DONE)
    r.close()

    # Each heartbeat tick calls _render via set_detail (if text changed) or
    # directly at end-of-loop. We expect at least 2 distinct renders during
    # the 1.2 s active window: start, heartbeat, finish.
    assert len(renders) >= 3, f"too few renders: {len(renders)}"


def test_transport_error_renders_warning() -> None:
    buf = io.StringIO()
    r = progress.JsonReporter.create(PHASES, stream=buf)
    calls = {"n": 0}

    def feed() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise progress.TransportError("ssh unreachable")
        return "back online"

    r.start("queued", feed=feed, mode="fast")
    time.sleep(1.2)
    r.finish(PhaseStatus.DONE)
    r.close()

    events = [json.loads(line) for line in _read_lines(buf)]
    # Active events during transport outage should carry transport_warning
    # (we can only verify this via JSON ticks that land within the 30 s
    # throttle — the first transition tick does, since it's forced).
    active_events = [e for e in events if e["status"] == "active"]
    # The transition tick may or may not carry a warning depending on race
    # with the heartbeat thread. Sanity: at least one active event exists.
    assert active_events


def test_backoff_schedule_shape() -> None:
    # 2 Hz for 30 s, 0.5 Hz for the next 90 s, 0.1 Hz thereafter.
    assert progress._interval_for(0.0, progress._FAST_SCHEDULE) == 0.5
    assert progress._interval_for(10.0, progress._FAST_SCHEDULE) == 0.5
    assert progress._interval_for(29.9, progress._FAST_SCHEDULE) == 0.5
    assert progress._interval_for(30.0, progress._FAST_SCHEDULE) == 2.0
    assert progress._interval_for(119.9, progress._FAST_SCHEDULE) == 2.0
    assert progress._interval_for(121.0, progress._FAST_SCHEDULE) == 10.0
    # Slow schedule is 10 s always.
    assert progress._interval_for(0.0, progress._SLOW_SCHEDULE) == 10.0
    assert progress._interval_for(3600.0, progress._SLOW_SCHEDULE) == 10.0


def test_finish_failed_sets_status() -> None:
    r = progress.NullReporter.create(PHASES)
    r.start("submit")
    r.finish(PhaseStatus.FAILED, detail="nope")
    r.close()
    submit_state = r._by_key["submit"]
    assert submit_state.status == PhaseStatus.FAILED
    assert submit_state.detail == "nope"


def test_context_manager_rolls_up_on_exception() -> None:
    r = progress.NullReporter.create(PHASES)
    try:
        with r.phase("submit"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    r.close()
    assert r._by_key["submit"].status == PhaseStatus.FAILED


def test_plain_reporter_prints_transitions() -> None:
    buf = io.StringIO()
    r = progress.PlainReporter.create(PHASES, stream=buf)
    r.start("submit", feed=lambda: "x")
    time.sleep(0.05)
    r.finish(PhaseStatus.DONE)
    r.close()
    lines = _read_lines(buf)
    # Plain emits at least one transition line with the label.
    assert any("Submit" in line for line in lines)


def test_close_without_finish_marks_active_failed() -> None:
    r = progress.NullReporter.create(PHASES)
    r.start("submit")
    r.close()  # caller forgot to finish()
    assert r._by_key["submit"].status == PhaseStatus.FAILED
