"""Entrypoint-level test: `mcode launch ...` (via mcode.cli:app) actually
reaches the launcher Typer app. Regression guard for the pre-merge finding
that `app.add_typer(launch_app, name='launch')` is wired in `mcode.cli`."""

from __future__ import annotations

from typer.testing import CliRunner


def test_launch_subcommand_reachable_from_mcode_cli() -> None:
    from mcode.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["launch", "--help"])
    assert result.exit_code == 0, result.output
    # Each of our target subcommands should be discoverable via the help tree.
    for sub in ("bluevela", "local-vllm", "local-ollama", "status", "doctor"):
        assert sub in result.output


def test_launch_status_reachable_from_mcode_cli(tmp_path, monkeypatch) -> None:
    """Not just help — the command actually runs."""
    from mcode.cli import app

    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(tmp_path / "state.json"))
    runner = CliRunner()
    result = runner.invoke(app, ["launch", "status"])
    assert result.exit_code == 0
