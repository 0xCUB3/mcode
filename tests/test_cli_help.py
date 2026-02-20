from __future__ import annotations

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
