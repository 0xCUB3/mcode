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

    sync = reporter.child(5, 35)
    sync.set(5, "Planning workspace sync")
    sync.set(90, "Remote workspace already current")
    reporter.set(35, "Starting or reusing Blue Vela server")
    server = reporter.child(35, 70)
    server.set(65, "Waiting for server health on host")

    assert [completed for completed, _ in reporter.updates] == [6, 32, 35, 57]
    assert reporter.completed == 57
