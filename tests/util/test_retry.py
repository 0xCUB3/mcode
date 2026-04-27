"""with_backoff exponential-backoff helper."""

from __future__ import annotations

import pytest

from mcode.util.retry import with_backoff


def test_returns_first_success_immediately():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    out = with_backoff(fn, is_retryable=lambda e: True)
    assert out == "ok"
    assert len(calls) == 1


def test_retries_until_success(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("mcode.util.retry.time.sleep", lambda s: sleeps.append(s))
    attempts = [0]

    def fn():
        attempts[0] += 1
        if attempts[0] < 3:
            raise RuntimeError("transient")
        return "yes"

    out = with_backoff(
        fn,
        is_retryable=lambda e: True,
        max_attempts=5,
        base_sleep_s=1.0,
        max_sleep_s=30.0,
    )
    assert out == "yes"
    assert attempts[0] == 3
    # sleeps after attempt 1 and attempt 2: 1.0s, 2.0s
    assert sleeps == [1.0, 2.0]


def test_non_retryable_exceptions_propagate_immediately(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("mcode.util.retry.time.sleep", lambda s: sleeps.append(s))
    attempts = [0]

    def fn():
        attempts[0] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        with_backoff(fn, is_retryable=lambda e: isinstance(e, RuntimeError))
    assert attempts[0] == 1
    assert sleeps == []


def test_raises_last_error_after_max_attempts(monkeypatch):
    monkeypatch.setattr("mcode.util.retry.time.sleep", lambda s: None)
    attempts = [0]

    def fn():
        attempts[0] += 1
        raise RuntimeError(f"try {attempts[0]}")

    with pytest.raises(RuntimeError, match="try 3"):
        with_backoff(fn, is_retryable=lambda e: True, max_attempts=3)
    assert attempts[0] == 3


def test_on_attempt_callback_fires_with_history(monkeypatch):
    monkeypatch.setattr("mcode.util.retry.time.sleep", lambda s: None)
    history: list[tuple[int, type | None]] = []

    def on_attempt(n, err):
        history.append((n, type(err) if err is not None else None))

    attempts = [0]

    def fn():
        attempts[0] += 1
        if attempts[0] < 3:
            raise RuntimeError("transient")
        return "ok"

    with_backoff(fn, is_retryable=lambda e: True, on_attempt=on_attempt)
    assert history == [(1, None), (2, RuntimeError), (3, RuntimeError)]


def test_sleep_schedule_caps_at_max(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("mcode.util.retry.time.sleep", lambda s: sleeps.append(s))

    def fn():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        with_backoff(
            fn,
            is_retryable=lambda e: True,
            max_attempts=8,
            base_sleep_s=1.0,
            max_sleep_s=10.0,
        )
    # 1, 2, 4, 8, 10, 10, 10 — 7 sleeps for 8 attempts
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0]
