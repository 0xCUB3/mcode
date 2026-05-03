"""Errors module: MCodeError hierarchy, print_error formatter, handle_errors."""

from __future__ import annotations

import io

import pytest
import typer

from mcode.launch.models import LaunchError
from mcode.ui.errors import (
    BenchError,
    ExitCode,
    InfraError,
    MCodeError,
    handle_errors,
    print_error,
)


def test_launch_error_is_mcode_error():
    e = LaunchError(what="x", why="y", next="z", logs="l")
    assert isinstance(e, MCodeError)
    assert e.what == "x"
    assert e.why == "y"
    assert e.next == "z"
    assert e.logs == "l"


def test_print_error_formats_with_what_why_next_logs(tmp_path):
    log_path = tmp_path / "x.log"
    err = MCodeError(what="boom", why="bad config", next="run init", logs=str(log_path))
    buf = io.StringIO()
    print_error(err, stream=buf)
    output = buf.getvalue()
    assert "boom" in output
    assert "  why:  bad config" in output
    assert "  next: run init" in output
    assert f"  logs: {log_path}" in output


def test_print_error_omits_empty_optional_fields():
    err = MCodeError(what="boom")
    buf = io.StringIO()
    print_error(err, stream=buf)
    output = buf.getvalue()
    assert "boom" in output
    assert "why:" not in output
    assert "next:" not in output
    assert "logs:" not in output


def test_print_error_no_ansi_when_not_a_tty(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    err = MCodeError(what="boom")
    buf = io.StringIO()
    print_error(err, stream=buf)
    assert "\x1b[" not in buf.getvalue()


def test_handle_errors_decorator_renders_and_exits():
    @handle_errors
    def cmd():
        raise LaunchError(what="bad", why="reason")

    with pytest.raises(typer.Exit) as exc:
        cmd()
    assert exc.value.exit_code == ExitCode.USER_ERROR


def test_handle_errors_propagates_non_mcode_errors():
    @handle_errors
    def cmd():
        raise RuntimeError("not us")

    with pytest.raises(RuntimeError):
        cmd()


def test_infra_error_uses_retryable_exit_code():
    @handle_errors
    def cmd():
        raise InfraError(what="podman flake")

    with pytest.raises(typer.Exit) as exc:
        cmd()
    assert exc.value.exit_code == ExitCode.INFRA_RETRYABLE


def test_mcode_debug_env_disables_formatter(monkeypatch):
    monkeypatch.setenv("MCODE_DEBUG", "1")

    @handle_errors
    def cmd():
        raise BenchError(what="x")

    with pytest.raises(BenchError):
        cmd()
