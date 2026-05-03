from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
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
    state_path = tmp_path / "launch-state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    return state_path


@pytest.fixture
def freeze_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    now = [1_700_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    yield now
