from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from mcode.ui.console import console
from mcode.ui.flags import JsonFlag

deps_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Install project dependencies and benchmark toolchains.",
)


@deps_app.command("sync")
def deps_sync(
    extra: Annotated[
        list[str] | None,
        typer.Option("--extra", help="Optional dependency extra to install (repeatable)."),
    ] = None,
    no_dev: Annotated[
        bool,
        typer.Option("--no-dev", help="Do not install the default dev extra."),
    ] = False,
) -> None:
    """Sync the uv environment for local development or benchmark runs."""
    from mcode.uv_setup import sync_uv_environment

    extras = list(extra or [])
    if not no_dev:
        extras.insert(0, "dev")
    sync_args: list[str] = []
    for name in extras:
        sync_args.extend(["--extra", name])

    selection = sync_uv_environment(Path.cwd(), sync_args=sync_args)
    if selection.source == "local":
        console.print(f"Using local mellea override at {selection.local_path}")
    else:
        console.print("Using upstream mellea package")


@deps_app.command("toolchains")
def deps_toolchains(
    benchmark: Annotated[
        str,
        typer.Option("--benchmark", help="Toolchain group to check or install"),
    ] = "aider-polyglot",
    language: Annotated[
        list[str] | None,
        typer.Option("--language", help="Aider Polyglot language to check (repeatable, or all)"),
    ] = None,
    install: Annotated[
        bool,
        typer.Option(
            "--install",
            help="Install missing runtimes with the local platform package manager",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Check or install the language runtimes used by Aider Polyglot."""
    if benchmark not in {"aider-polyglot", "polyglot"}:
        raise typer.BadParameter("only --benchmark aider-polyglot is supported")

    from mcode.bench.toolchains import (
        check_polyglot_toolchains,
        install_hint,
        install_polyglot_toolchains,
        normalize_polyglot_languages,
    )

    languages = normalize_polyglot_languages(language or "all")
    if install:
        install_polyglot_toolchains(languages)
    checks = check_polyglot_toolchains(languages)
    rows = [
        {
            "language": check.language,
            "name": check.name,
            "ok": check.ok,
            "detail": check.detail,
            "next": check.next,
        }
        for check in checks
    ]
    if json_mode:
        console.print_json(data=rows)
    else:
        table = Table(title="Aider Polyglot toolchains")
        table.add_column("language")
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail")
        table.add_column("next")
        for row in rows:
            table.add_row(
                str(row["language"]),
                str(row["name"]),
                "ok" if row["ok"] else "missing",
                str(row["detail"]),
                str(row["next"] or "-"),
            )
        console.print(table)
        missing_languages = sorted({str(row["language"]) for row in rows if not row["ok"]})
        hint = install_hint(missing_languages)
        if hint:
            console.print(f"install: {hint}")
    if any(not row["ok"] for row in rows):
        raise typer.Exit(1)
