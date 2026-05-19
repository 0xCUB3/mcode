from __future__ import annotations

from typing import Annotated

import typer

from mcode.bench.cli import bench_app
from mcode.cli_shared import _parse_task_ids  # noqa: F401 - re-exported for compatibility/tests
from mcode.commands.deps import deps_app
from mcode.commands.doctor import register_doctor_command
from mcode.commands.results import register_results_commands
from mcode.ui.console import configure_logging as _configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run mCode benchmarks, launch model servers, and inspect results.",
)


def _version_callback(value: bool) -> None:
    if value:
        try:
            from importlib.metadata import PackageNotFoundError, version

            v = version("mcode")
        except (ImportError, PackageNotFoundError):
            v = "unknown"
        print(f"mcode {v}")
        raise typer.Exit(0)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show Mellea INFO logs")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print mcode version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """mCode benchmarking harness."""
    _configure_logging(verbose=verbose)


@app.command("watch")
def watch_cmd() -> None:
    """Live dashboard combining `mcode launch status` + `mcode bench list`.

    Refreshes every 2s. Quits cleanly on Ctrl+C. Recovers automatically from
    transient state-file read failures (partial writes, lock contention)."""
    from mcode.watch import watch

    raise typer.Exit(watch())


register_doctor_command(app)
register_results_commands(app)
app.add_typer(bench_app, name="bench", help="Run benchmarks and manage bench run records.")
app.add_typer(deps_app, name="deps", help="Sync dependencies and benchmark toolchains.")

from mcode.launch.cli import app as launch_app  # noqa: E402

app.add_typer(launch_app, name="launch")
