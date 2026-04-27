"""Flag helpers should be Annotated typer.Option and round-trip through Typer."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from mcode.ui.flags import JsonFlag, NoColorFlag, QuietFlag


def test_flags_compose_into_a_typer_command():
    app = typer.Typer()

    @app.command()
    def show(json_mode: JsonFlag = False, quiet: QuietFlag = False, no_color: NoColorFlag = False):
        typer.echo(f"json={json_mode} quiet={quiet} no_color={no_color}")

    runner = CliRunner()
    res = runner.invoke(app, ["--json", "--quiet", "--no-color"])
    assert res.exit_code == 0, res.output
    assert "json=True quiet=True no_color=True" in res.output


def test_flags_default_off():
    app = typer.Typer()

    @app.command()
    def show(json_mode: JsonFlag = False):
        typer.echo(f"json={json_mode}")

    runner = CliRunner()
    res = runner.invoke(app, [])
    assert res.exit_code == 0
    assert "json=False" in res.output
