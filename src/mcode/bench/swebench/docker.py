from __future__ import annotations

import os
import time

import docker
from docker.errors import DockerException


class DockerUnavailableError(RuntimeError):
    pass


def _docker_retry_settings() -> tuple[int, float]:
    retries_raw = os.environ.get("MCODE_DOCKER_CONNECT_RETRIES", "3")
    delay_raw = os.environ.get("MCODE_DOCKER_RETRY_DELAY", "1")
    try:
        retries = int(retries_raw)
    except ValueError:
        raise ValueError(f"MCODE_DOCKER_CONNECT_RETRIES must be an int (got {retries_raw!r})")
    try:
        delay = float(delay_raw)
    except ValueError:
        raise ValueError(f"MCODE_DOCKER_RETRY_DELAY must be a float (got {delay_raw!r})")
    return max(1, retries), max(0.0, delay)


def _close_docker_client(client: object | None) -> None:
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    except Exception:
        pass


def _docker_error_message(scope: str) -> str:
    return (
        f"Docker is required for {scope}, but the daemon/socket is not reachable. "
        "Start Docker Desktop (or configure DOCKER_HOST) and retry."
    )


def is_docker_unavailable_error(exc: BaseException) -> bool:
    if isinstance(exc, DockerUnavailableError):
        return True
    if isinstance(exc, DockerException):
        return True

    current: BaseException | None = exc
    seen: set[int] = set()
    parts: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    text = " | ".join(parts).lower()
    markers = (
        "dockerexception",
        "connection aborted",
        "connection refused",
        "broken pipe",
        "read timed out",
        "readtimeout",
        "unixhttpconnectionpool",
        "podman.sock",
        "docker.sock",
        "unix://",
    )
    return any(marker in text for marker in markers)


def ensure_docker_client(
    client: object | None,
    *,
    scope: str,
    from_env_kwargs: dict | None = None,
):
    kwargs = dict(from_env_kwargs or {})
    message = _docker_error_message(scope)
    retries, delay = _docker_retry_settings()
    last_error: BaseException | None = None

    if client is not None:
        try:
            client.ping()
            return client
        except Exception as exc:
            last_error = exc
            _close_docker_client(client)

    for attempt in range(retries):
        fresh_client = None
        try:
            fresh_client = docker.from_env(**kwargs)
            fresh_client.ping()
            return fresh_client
        except Exception as exc:
            last_error = exc
            _close_docker_client(fresh_client)
            if attempt + 1 < retries and delay > 0:
                time.sleep(delay)

    raise DockerUnavailableError(message) from last_error


def reraise_docker_unavailable(exc: BaseException, *, scope: str) -> None:
    if is_docker_unavailable_error(exc):
        raise DockerUnavailableError(_docker_error_message(scope)) from exc
