"""Shared Typer Option helpers for cross-command flags.

Codex review note: Typer/Click root callback flags do NOT auto-propagate to
subcommands. Each subcommand opts in by adding the relevant Annotated alias
to its signature. There is no magic inheritance — the helpers below only
exist so the help text / option name / default stays consistent everywhere
the flag appears.
"""

from __future__ import annotations

from typing import Annotated

import typer

JsonFlag = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Emit machine-readable JSON instead of human-formatted output.",
    ),
]

QuietFlag = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress non-error output.",
    ),
]

NoColorFlag = Annotated[
    bool,
    typer.Option(
        "--no-color",
        help="Disable ANSI color (also honored via NO_COLOR env var).",
    ),
]

LogLevelFlag = Annotated[
    str,
    typer.Option(
        "--log-level",
        help="Set the log level: debug, info, warning, error.",
    ),
]


__all__ = ["JsonFlag", "LogLevelFlag", "NoColorFlag", "QuietFlag"]
