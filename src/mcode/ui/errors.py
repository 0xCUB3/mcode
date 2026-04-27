"""Unified error model + formatter for mcode.

`MCodeError` is the parent of every domain-specific error in mcode (`LaunchError`,
`BenchError`, `InfraError`). It carries `what / why / next / logs` like the
pre-existing `LaunchError` and `print_error` formats it identically:

    ✗ {what}
      why:  {why}
      next: {next}
      logs: {logs}

The leading ✗ is wrapped in ANSI red when stderr is a TTY and color is allowed
by ui.styles.color_enabled. Bytes are unchanged from the pre-existing
launch/cli.py:_print_error so existing automation that greps for "✗ " keeps
working.

@handle_errors decorator wraps a Typer command callable so any MCodeError
raised inside is formatted and turns into typer.Exit(1). MCODE_DEBUG=1
bypasses the formatter and re-raises for the dev traceback.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from enum import IntEnum
from typing import TypeVar

from mcode.ui.styles import ANSI_RED, ANSI_RESET, color_enabled


class ExitCode(IntEnum):
    SUCCESS = 0
    USER_ERROR = 1
    USAGE = 2
    NOT_CANCELLABLE = 2
    INFRA_RETRYABLE = 86
    INTERRUPTED = 130


class MCodeError(Exception):
    """Base class for user-facing mcode errors with actionable remediation."""

    exit_code: int = ExitCode.USER_ERROR

    def __init__(self, what: str, why: str = "", next: str = "", logs: str = "") -> None:
        super().__init__(what)
        self.what = what
        self.why = why
        self.next = next
        self.logs = logs


class BenchError(MCodeError):
    """A benchmark command failed in a user-actionable way."""


class InfraError(MCodeError):
    """A retryable infrastructure failure (podman pulls, transient SSH, etc.)."""

    exit_code: int = ExitCode.INFRA_RETRYABLE


def print_error(e: MCodeError, *, stream=None) -> None:
    """Format an MCodeError to stderr.

    Output bytes are unchanged from launch/cli.py:_print_error so existing
    test fixtures and any user automation greps remain valid.
    """
    out = stream if stream is not None else sys.stderr
    if color_enabled(stream=out):
        prefix = f"{ANSI_RED}✗{ANSI_RESET}"
    else:
        prefix = "✗"
    print(f"{prefix} {e.what}", file=out)
    if e.why:
        print(f"  why:  {e.why}", file=out)
    if e.next:
        print(f"  next: {e.next}", file=out)
    if e.logs:
        print(f"  logs: {e.logs}", file=out)


F = TypeVar("F", bound=Callable[..., object])


def handle_errors(fn: F) -> F:
    """Decorator: catch MCodeError, render it, exit with the right code.

    MCODE_DEBUG=1 disables the formatter so devs get a full traceback. Other
    exception types propagate unchanged — Typer renders its own usage errors,
    KeyboardInterrupt should still bubble.
    """
    import typer

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MCodeError as e:
            if os.environ.get("MCODE_DEBUG"):
                raise
            print_error(e)
            raise typer.Exit(e.exit_code) from e

    return wrapper  # type: ignore[return-value]


__all__ = [
    "BenchError",
    "ExitCode",
    "InfraError",
    "MCodeError",
    "handle_errors",
    "print_error",
]
