from __future__ import annotations

from pathlib import Path

from mcode.launch.state import LauncherState, load_state


def test_load_state_treats_empty_file_as_empty_state(tmp_path: Path) -> None:
    state_path = tmp_path / "launch-state.json"
    state_path.write_text("")

    state = load_state(state_path)

    assert state == LauncherState()
