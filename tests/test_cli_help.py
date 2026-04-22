from __future__ import annotations

from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from mcode.cli import app


def _invoke_help(*args: str) -> str:
    runner = CliRunner()
    res = runner.invoke(
        app,
        list(args),
        color=False,
        env={"COLUMNS": "120", "TERM": "xterm-256color"},
    )
    assert res.exit_code == 0
    return res.stdout


def _command_option_names(*args: str) -> set[str]:
    command = get_command(app)
    current = command
    for name in args:
        current = current.get_command(None, name)
        assert current is not None
    return {param.name for param in current.params}


def test_cli_help() -> None:
    _invoke_help("--help")


def test_cli_bench_swebench_help() -> None:
    _invoke_help("bench", "swebench-lite", "--help")
    option_names = _command_option_names("bench", "swebench-lite")
    assert "shards" in option_names
    assert "n_samples" in option_names
    assert "sampling" in option_names
    assert "sampling_budget" in option_names
    assert "on" in option_names
    assert "fetch_db" in option_names


def test_cli_report_help() -> None:
    _invoke_help("report", "--help")


def test_cli_bench_swebench_live_help() -> None:
    option_names = _command_option_names("bench", "swebench-live")
    assert "shards" in option_names
    assert "n_samples" in option_names
    assert "sampling" in option_names
    assert "sampling_budget" in option_names
    assert "on" in option_names
    assert "fetch_db" in option_names


def test_cli_bench_smoke_help() -> None:
    option_names = _command_option_names("bench", "smoke")
    assert "shards" in option_names


def test_cli_compare_help() -> None:
    _invoke_help("compare", "--help")


def test_cli_rejects_sampling_budget_without_sampling() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bench",
            "swebench-lite",
            "--model",
            "test-model",
            "--sampling-budget",
            "2",
        ],
        color=False,
    )

    assert res.exit_code != 0
    assert "--sampling-budget requires --sampling != none" in res.output


def test_cli_deps_sync_help() -> None:
    _invoke_help("deps", "sync", "--help")


def test_cli_deps_sync_defaults_to_dev_extra(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_sync(
        project_root: Path,
        *,
        env=None,
        sync_args=None,
        run_command=None,
    ):
        captured["project_root"] = project_root
        captured["sync_args"] = sync_args

        class Selection:
            source = "github"
            local_path = None

        return Selection()

    monkeypatch.setattr("mcode.uv_setup.sync_uv_environment", fake_sync)

    res = runner.invoke(app, ["deps", "sync"])

    assert res.exit_code == 0
    assert captured["project_root"] == Path.cwd()
    assert captured["sync_args"] == ["--extra", "dev"]


def test_cli_deps_sync_allows_disabling_dev_and_adding_extras(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_sync(
        project_root: Path,
        *,
        env=None,
        sync_args=None,
        run_command=None,
    ):
        captured["project_root"] = project_root
        captured["sync_args"] = sync_args

        class Selection:
            source = "github"
            local_path = None

        return Selection()

    monkeypatch.setattr("mcode.uv_setup.sync_uv_environment", fake_sync)

    res = runner.invoke(
        app,
        ["deps", "sync", "--no-dev", "--extra", "swebench", "--extra", "datasets"],
    )

    assert res.exit_code == 0
    assert captured["project_root"] == Path.cwd()
    assert captured["sync_args"] == ["--extra", "swebench", "--extra", "datasets"]
