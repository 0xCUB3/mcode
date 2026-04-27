"""Shared pytest fixtures for the mcode test suite.

Wave 1 introduces these because every command test needs the same scaffolding:
a Typer CliRunner with deterministic NO_COLOR output, an ANSI-stripping helper,
a redirect for the persistent state file path, and a frozen-time helper.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """Typer CliRunner with NO_COLOR forced on and ANSI disabled."""
    monkeypatch.setenv("NO_COLOR", "1")
    return CliRunner(mix_stderr=False)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def strip_ansi():
    def _strip(s: str) -> str:
        return _ANSI_RE.sub("", s)

    return _strip


@pytest.fixture
def tmp_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the launch state file into tmp_path so tests don't poison the
    user's real state and can inspect it cleanly."""
    state_path = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    return state_path


@pytest.fixture
def freeze_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Replace time.time() with a controllable counter. Yields a list whose
    [0] entry is the current 'now'; tests can mutate it to advance time."""
    now = [1_700_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    yield now
