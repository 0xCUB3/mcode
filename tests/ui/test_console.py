"""Console + configure_logging."""

from __future__ import annotations

import logging

from mcode.ui.console import configure_logging, console


def test_console_singleton_exists():
    assert console is not None


def test_configure_logging_sets_mcode_logger_level():
    configure_logging("debug")
    assert logging.getLogger("mcode").level == logging.DEBUG
    configure_logging("warning")
    assert logging.getLogger("mcode").level == logging.WARNING


def test_configure_logging_verbose_overrides_level():
    configure_logging(level="error", verbose=True)
    assert logging.getLogger("mcode").level == logging.INFO


def test_configure_logging_handles_unknown_level():
    configure_logging("totally-bogus")
    # Falls back to WARNING.
    assert logging.getLogger("mcode").level == logging.WARNING


def test_configure_logging_sets_mellea_logger_level():
    from mellea.core.utils import FancyLogger

    configure_logging("error")
    logger = FancyLogger.get_logger()
    assert logger.level == logging.ERROR
    assert all(handler.level == logging.ERROR for handler in logger.handlers)

    configure_logging("warning")


def test_configure_logging_quiets_dependency_loggers():
    configure_logging("warning")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("cpex").level == logging.WARNING

    configure_logging(verbose=True)
    assert logging.getLogger("httpx").level == logging.INFO
    assert logging.getLogger("cpex").level == logging.INFO

    configure_logging("warning")
