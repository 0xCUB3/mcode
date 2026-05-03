"""Rich Console singleton + unified logging configuration.

The singleton is used everywhere mcode wants Rich-rendered output (tables,
progress, formatted help). Honors NO_COLOR and MCODE_NO_COLOR env vars and
the isatty status of stderr.

configure_logging() drives both the stdlib `mcode` logger and Mellea's
FancyLogger from one entry point so --verbose / --log-level affect both
consistently.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

from rich.console import Console

from mcode.ui.styles import color_enabled


def _make_console() -> Console:
    no_color = not color_enabled(stream=sys.stderr)
    return Console(
        stderr=False,
        no_color=no_color,
        force_terminal=False if no_color else None,
    )


console: Console = _make_console()


LogLevel = Literal["debug", "info", "warn", "warning", "error"]
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: LogLevel | str | int = "warning", *, verbose: bool = False) -> None:
    """Configure mcode's stdlib logger and Mellea's FancyLogger together.

    `verbose=True` overrides `level` to INFO for back-compat with the
    pre-existing `mcode --verbose` flag at cli.py:191-197.
    """
    if verbose:
        resolved = logging.INFO
    elif isinstance(level, int):
        resolved = level
    else:
        resolved = _LEVEL_MAP.get(level.lower(), logging.WARNING)

    mcode_logger = logging.getLogger("mcode")
    mcode_logger.setLevel(resolved)

    # Mellea logging is optional in some test and install modes. Keep this
    # best-effort so mcode logging still works without the Mellea helper.
    try:
        from mellea.helpers.fancy_logger import FancyLogger

        mellea_logger = FancyLogger.get_logger()
        mellea_logger.setLevel(resolved)
        for h in mellea_logger.handlers:
            h.setLevel(resolved)
    except Exception:
        return


__all__ = ["LogLevel", "configure_logging", "console"]


if os.environ.get("MCODE_DEBUG"):
    # Make tracebacks visible during dev. Rich already formats nicely when
    # asked, but we don't install the global handler — error rendering still
    # goes through ui.errors.print_error.
    logging.basicConfig(level=logging.DEBUG)
