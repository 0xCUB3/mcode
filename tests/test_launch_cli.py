from __future__ import annotations

import json
import re

from typer.main import get_command
from typer.testing import CliRunner

from mcode.cli import app
from mcode.launch.models import CommandResult


def _command_option_names(*args: str) -> set[str]:
    command = get_command(app)
    current = command
    for name in args:
        current = current.get_command(None, name)
        assert current is not None
    return {param.name for param in current.params}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


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
    assert "follow" in option_names
    assert "detach" not in option_names


def test_cli_launch_sync_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "sync", "--help"], color=False)
    assert res.exit_code == 0
    stdout = _strip_ansi(res.stdout)
    assert "--check" in stdout
    assert "--apply" in stdout


def test_cli_launch_json_emits_valid_json(monkeypatch) -> None:
    from mcode import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "launch_run",
        lambda spec, repo_root, reporter=None: CommandResult(
            ok=True,
            message="line one\nline two",
            data={"run_id": "run-1"},
        ),
    )
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "launch",
            "--target",
            "openai-compatible",
            "--openai-base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "test-model",
            "--benchmark",
            "swebench-lite",
            "--json",
        ],
        color=False,
    )
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["message"] == "line one\nline two"
    assert payload["data"]["run_id"] == "run-1"


def test_cli_launch_non_json_prints_run_and_server_ids_first(monkeypatch) -> None:
    from mcode import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "launch_run",
        lambda spec, repo_root, reporter=None: CommandResult(
            ok=True,
            message="submitted",
            data={"run_id": "run-1", "server_id": "server-1"},
        ),
    )
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "launch",
            "--target",
            "openai-compatible",
            "--openai-base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "test-model",
            "--benchmark",
            "swebench-lite",
        ],
        color=False,
    )
    assert res.exit_code == 0
    stdout = _strip_ansi(res.stdout)
    assert stdout.lstrip().startswith("run_id: run-1\nserver_id: server-1\nsubmitted")


def test_cli_launch_status_json_emits_valid_json(monkeypatch) -> None:
    from mcode import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "launch_status",
        lambda: {
            "runs": [
                {
                    "id": "run-1",
                    "metadata": {
                        "commands": ["line one\nline two"],
                    },
                }
            ],
            "servers": [],
            "workspaces": [],
        },
    )
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "status", "--json"], color=False)
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["runs"][0]["metadata"]["commands"] == ["line one\nline two"]


def test_cli_launch_stop_all_prompts_and_calls_service(monkeypatch) -> None:
    from mcode import cli as cli_module

    called: dict[str, object] = {}
    monkeypatch.setattr(cli_module.typer, "confirm", lambda message, abort=False: True)
    monkeypatch.setattr(
        cli_module,
        "launch_stop_all",
        lambda *,
        target=None,
        state_path=None,
        bluevela_login=None,
        bluevela_workspace_root=None,
        reporter=None: (
            called.update(
                {
                    "target": target,
                    "state_path": state_path,
                    "bluevela_login": bluevela_login,
                    "bluevela_workspace_root": bluevela_workspace_root,
                }
            )
            or CommandResult(ok=True, message="Stopped all", data={"runs_stopped": 1})
        ),
    )
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "stop", "--all", "--target", "bluevela"], color=False)
    assert res.exit_code == 0
    assert "Stopped all" in _strip_ansi(res.stdout)
    assert called["target"] == "bluevela"
    assert called["bluevela_login"]
    assert called["bluevela_workspace_root"]


def test_cli_launch_stop_all_json_requires_yes() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["launch", "stop", "--all", "--json"], color=False)
    assert res.exit_code == 1
    assert "--all with --json requires --yes" in _strip_ansi(res.stdout)
