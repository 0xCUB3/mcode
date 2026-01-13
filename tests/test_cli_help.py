from __future__ import annotations

from typer.testing import CliRunner

from mcode.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0

