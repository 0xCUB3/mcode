"""Phase-list progress UI with heartbeat and adaptive backoff.

Contract (see plan: "Phase-based progress UI with heartbeat"):

- Each target declares a list[Phase] up front.
- Exactly one phase is active at a time; others are pending, done, or failed.
- The active phase has a background heartbeat task that refreshes its detail
  line at 2 Hz for the first 30 s, 0.5 Hz from 30 s to 2 min, and 0.1 Hz
  afterward. Transport failures render distinctly, not as "remote quiet".
- No percentages. No rescaling. Phase transitions come from explicit signals
  only; log markers decorate the detail line but never drive transitions.
- --json mode emits one line per transition and every 30 s during a phase,
  not every heartbeat tick.

Public API (TBD):

    class Reporter(Protocol):
        def start_phase(key: str) -> None: ...
        def update(detail: str) -> None: ...
        def finish_phase(status: PhaseStatus, detail: str = "") -> None: ...

    class RichReporter(Reporter): ...   # the default terminal UI
    class JsonReporter(Reporter): ...   # for --json
    class NullReporter(Reporter): ...   # for tests

    class HeartbeatLoop:
        '''Calls a feed fn on the adaptive backoff schedule.'''
"""

from __future__ import annotations
