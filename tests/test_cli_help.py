from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mcode.cli import app


def _invoke_help(*args: str) -> str:
    runner = CliRunner()
    res = runner.invoke(app, list(args), color=False)
    assert res.exit_code == 0
    return res.stdout


def test_cli_help() -> None:
    _invoke_help("--help")


def test_cli_bench_swebench_help() -> None:
    help_text = _invoke_help("bench", "swebench-lite", "--help")
    assert "--n-samples" in help_text
    assert "Sampling strategy: repair," in help_text
    assert "sofai, or raw" in help_text


def test_cli_bench_humaneval_plus_help() -> None:
    _invoke_help("bench", "humaneval+", "--help")


def test_cli_bench_mbpp_plus_help() -> None:
    _invoke_help("bench", "mbpp+", "--help")


def test_cli_report_help() -> None:
    _invoke_help("report", "--help")


def test_cli_bench_livecodebench_help() -> None:
    _invoke_help("bench", "livecodebench", "--help")


def test_cli_bench_bigcodebench_complete_help() -> None:
    _invoke_help("bench", "bigcodebench-complete", "--help")


def test_cli_bench_bigcodebench_instruct_help() -> None:
    _invoke_help("bench", "bigcodebench-instruct", "--help")


def test_cli_bench_swebench_live_help() -> None:
    help_text = _invoke_help("bench", "swebench-live", "--help")
    assert "--n-samples" in help_text


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
