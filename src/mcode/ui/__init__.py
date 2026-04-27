"""Shared UI primitives for mcode commands.

Console singleton, error formatting, styles, flag helpers, and a fresh task
reporter for bench-style progress (parallel to launch.progress, which keeps
its phase-list reporter unchanged).
"""

from __future__ import annotations

from mcode.ui.console import configure_logging, console
from mcode.ui.errors import (
    BenchError,
    ExitCode,
    InfraError,
    MCodeError,
    handle_errors,
    print_error,
)
from mcode.ui.styles import Symbol

__all__ = [
    "BenchError",
    "ExitCode",
    "InfraError",
    "MCodeError",
    "Symbol",
    "configure_logging",
    "console",
    "handle_errors",
    "print_error",
]
