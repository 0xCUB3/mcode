from __future__ import annotations

from typer.main import get_command
from typer.testing import CliRunner

from mcode.cli import app


def _command_option_names(*args: str) -> set[str]:
    command = get_command(app)
    current = command
    for name in args:
        current = current.get_command(None, name)
        assert current is not None
    return {param.name for param in current.params}


def test_cli_launch_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "--help"], color=False)
    assert res.exit_code == 0
    assert "doctor" in res.stdout
    assert "status" in res.stdout
    assert "sync" in res.stdout


def test_cli_launch_has_core_options() -> None:
    option_names = _command_option_names("launch")
    assert "target" in option_names
    assert "model" in option_names
    assert "benchmark" in option_names
    assert "reuse" in option_names
    assert "sync_mode" in option_names
    assert "json_mode" in option_names
    assert "gpu_memory_utilization" in option_names


def test_cli_launch_sync_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "sync", "--help"], color=False)
    assert res.exit_code == 0
    assert "--check" in res.stdout
    assert "--apply" in res.stdout
