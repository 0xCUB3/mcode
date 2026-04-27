"""Centralized symbols and color tokens used by mcode UI.

Hand-coded ANSI lives only here so the rest of the codebase doesn't sprinkle
escape sequences. NO_COLOR / --no-color is honored at the rendering layer
(Rich Console handles it for the rich path; ui.errors checks isatty for the
plain stderr path).
"""

from __future__ import annotations

import os
import sys
from enum import Enum


class Symbol(str, Enum):
    SUCCESS = "✓"
    FAIL = "✗"
    PENDING = "·"
    RUNNING = "▶"
    WARN = "⚠"
    INFO = "i"
    ARROW = "→"
    RESUME = "↻"


# ANSI codes for the plain-stderr error path. Kept in lockstep with the
# pre-existing launch/cli.py:_print_error formatter so byte output is
# unchanged.
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"


def color_enabled(stream=None) -> bool:
    """Decide whether to emit ANSI colors for the given stream.

    Honors the NO_COLOR env var (https://no-color.org), MCODE_NO_COLOR for
    explicit override, and isatty() of the stream. Stream defaults to stderr,
    matching the only user of this helper today (ui.errors.print_error).
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("MCODE_NO_COLOR"):
        return False
    s = stream if stream is not None else sys.stderr
    return bool(getattr(s, "isatty", lambda: False)())
