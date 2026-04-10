from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressReporter:
    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def set(self, completed: int, description: str) -> None:
        pass

    def finish(self, description: str) -> None:
        self.set(100, description)

    def child(self, start: int, end: int) -> ProgressReporter:
        return MappedProgressReporter(parent=self, start=start, end=end)


class NullProgressReporter(ProgressReporter):
    pass


@dataclass
class RichProgressReporter(ProgressReporter):
    console: Console
    title: str

    def __post_init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self._task_id: int | None = None
        self._completed = 0

    def __enter__(self) -> RichProgressReporter:
        self._progress.__enter__()
        self._task_id = self._progress.add_task(self.title, total=100, completed=0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def set(self, completed: int, description: str) -> None:
        assert self._task_id is not None
        self._completed = max(self._completed, completed)
        self._progress.update(
            self._task_id,
            completed=self._completed,
            description=description,
        )
        self._progress.refresh()


@dataclass
class MappedProgressReporter(ProgressReporter):
    parent: ProgressReporter
    start: int
    end: int

    def __post_init__(self) -> None:
        self._completed = 0

    def set(self, completed: int, description: str) -> None:
        self._completed = max(self._completed, completed)
        span = self.end - self.start
        mapped = self.start + int((self._completed / 100) * span)
        self.parent.set(mapped, description)

    def child(self, start: int, end: int) -> ProgressReporter:
        base_start = self.start + int(((max(0, min(100, start))) / 100) * (self.end - self.start))
        base_end = self.start + int(((max(0, min(100, end))) / 100) * (self.end - self.start))
        return MappedProgressReporter(parent=self.parent, start=base_start, end=base_end)
