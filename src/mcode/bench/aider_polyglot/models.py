from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class AiderPolyglotTask:
    benchmark: str
    task_id: str
    language: str
    exercise: str
    source_dir: Path

    @property
    def repo(self) -> str:
        return f"aider-polyglot/{self.language}/{self.exercise}"


@dataclass(frozen=True)
class CommandOutcome:
    passed: bool
    output: str
    exit_code: int | None
    timed_out: bool


@dataclass(frozen=True)
class PreparedPolyglotTask:
    task: AiderPolyglotTask
    work_dir: Path
    stub_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    test_commands: tuple[str, ...]
    timeout_s: int
    tempdir: TemporaryDirectory[str] = field(repr=False, compare=False)

    def build_first_prompt(self) -> str:
        from .prompts import build_first_prompt

        return build_first_prompt(self)

    def build_retry_prompt(self, test_output: str) -> str:
        from .prompts import build_retry_prompt

        return build_retry_prompt(self, test_output)
