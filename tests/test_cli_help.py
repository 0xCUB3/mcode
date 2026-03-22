from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mcode.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0


def test_cli_bench_swebench_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "swebench-lite", "--help"])
    assert res.exit_code == 0
    assert "--n-samples" in res.stdout
    assert "Sampling strategy: repair," in res.stdout
    assert "sofai, or raw" in res.stdout


def test_cli_bench_humaneval_plus_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "humaneval+", "--help"])
    assert res.exit_code == 0


def test_cli_bench_mbpp_plus_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "mbpp+", "--help"])
    assert res.exit_code == 0


def test_cli_report_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["report", "--help"])
    assert res.exit_code == 0


def test_cli_bench_livecodebench_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "livecodebench", "--help"])
    assert res.exit_code == 0


def test_cli_bench_bigcodebench_complete_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "bigcodebench-complete", "--help"])
    assert res.exit_code == 0


def test_cli_bench_bigcodebench_instruct_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "bigcodebench-instruct", "--help"])
    assert res.exit_code == 0


def test_cli_bench_swebench_live_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "swebench-live", "--help"])
    assert res.exit_code == 0
    assert "--n-samples" in res.stdout


def test_cli_deps_sync_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["deps", "sync", "--help"])
    assert res.exit_code == 0


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
