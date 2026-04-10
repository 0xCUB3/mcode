from __future__ import annotations

from mcode.launch.progress import ProgressReporter


class _CollectingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.completed = 0
        self.updates: list[tuple[int, str]] = []

    def set(self, completed: int, description: str) -> None:
        self.completed = max(self.completed, completed)
        self.updates.append((self.completed, description))


def test_child_progress_stays_monotonic_across_phases() -> None:
    reporter = _CollectingReporter()

    sync = reporter.child(2, 20)
    sync.set(5, "Planning workspace sync")
    sync.set(90, "Remote workspace already current")
    reporter.set(20, "Starting or reusing Blue Vela server")
    server = reporter.child(20, 85)
    server.set(65, "Waiting for server health on host")

    # sync child: 5% of (2..20) = 2+0.9 = 2, 90% of (2..20) = 2+16.2 = 18
    # server child: 65% of (20..85) = 20+42.25 = 62
    assert [completed for completed, _ in reporter.updates] == [2, 18, 20, 62]
    assert reporter.completed == 62
