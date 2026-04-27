"""Exponential-backoff retry helper.

Replaces three near-identical loops in the codebase:
- launch/bluevela.py queued-phase SSH fail streak
- launch/bluevela.py starting-phase SSH fail streak (basically the same loop)
- execution/swebench.py podman pull retry

Sleep between attempts follows `min(base_sleep_s * 2 ** (attempt - 1), max_sleep_s)`.
Note this is NOT identical to the launcher's pre-existing
`min(2 ** ssh_fail_streak, 30)` schedule: that started at 2 (streak=1), this
starts at 1 (attempt=1). Callers that want the original schedule should pass
`base_sleep_s=2.0`. The launcher continues to use its own inline streak loop
inside `_absorb_ssh_blip` to preserve byte-identical sleep cadence; this
helper is for new callers and the swebench podman pull retry.

`is_retryable` decides whether to keep retrying after an exception. Non-retryable
exceptions propagate immediately. After `max_attempts` retries, the last
exception is re-raised wrapped or re-raised as-is depending on caller.

`on_attempt(attempt_number, last_error)` runs before each non-first attempt and
is the place to feed progress UI ("retrying after transient SSH error...").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_backoff(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_attempts: int = 5,
    base_sleep_s: float = 1.0,
    max_sleep_s: float = 30.0,
    on_attempt: Callable[[int, BaseException | None], None] | None = None,
) -> T:
    """Call `fn()` with exponential-backoff on retryable failures.

    Returns the first successful result. Raises the last retryable exception
    after `max_attempts` exhausted, or the first non-retryable exception
    immediately. `attempt` numbering is 1-based for `on_attempt`.
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt, last_error)
        try:
            return fn()
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep = min(base_sleep_s * (2 ** (attempt - 1)), max_sleep_s)
            time.sleep(sleep)
    assert last_error is not None  # max_attempts >= 1 invariant
    raise last_error


__all__ = ["with_backoff"]
