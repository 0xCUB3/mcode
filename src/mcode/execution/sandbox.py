from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import DockerException, ImageNotFound


@dataclass(frozen=True)
class SandboxRun:
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None


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


class DockerSandbox:
    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        mem_limit: str = "1g",
        pids_limit: int = 256,
    ):
        self.image = image
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self._client: docker.DockerClient | None = None
        self._ensured_image: str | None = None

    def _get_client(self) -> docker.DockerClient:
        self._client = ensure_docker_client(self._client, scope="sandboxed execution")
        return self._client

    def check_available(self) -> None:
        self._get_client()

    def ensure_image(self) -> None:
        if self._ensured_image == self.image:
            return
        client = self._get_client()
        try:
            client.images.get(self.image)
        except ImageNotFound:
            client.images.pull(self.image)
        self._ensured_image = self.image

    def run_python(self, code: str, *, timeout_s: int = 60) -> SandboxRun:
        self.ensure_image()
        client = self._get_client()
        with tempfile.TemporaryDirectory(prefix="mcode-sandbox-") as td:
            host_dir = Path(td)
            host_dir.chmod(0o755)
            script = host_dir / "main.py"
            script.write_text(code, encoding="utf-8")
            script.chmod(0o644)

            container = None
            timed_out = False
            try:
                container = client.containers.run(
                    self.image,
                    command=["python", "-I", "-B", "/work/main.py"],
                    working_dir="/work",
                    detach=True,
                    network_disabled=True,
                    mem_limit=self.mem_limit,
                    pids_limit=self.pids_limit,
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    read_only=True,
                    tmpfs={"/tmp": ""},
                    user="65534:65534",
                    volumes={str(host_dir): {"bind": "/work", "mode": "ro"}},
                    environment={
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLBACKEND": "Agg",
                    },
                )
                try:
                    result = container.wait(timeout=timeout_s)
                except Exception:
                    timed_out = True
                    container.kill()
                    result = {"StatusCode": None}

                # docker-py log demux support varies by version; keep compatibility.
                try:
                    stdout_b = container.logs(stdout=True, stderr=False)
                    stderr_b = container.logs(stdout=False, stderr=True)
                except TypeError:
                    combined = container.logs(stdout=True, stderr=True)
                    stdout_b, stderr_b = combined, b""

                stdout = (stdout_b or b"").decode("utf-8", errors="replace")
                stderr = (stderr_b or b"").decode("utf-8", errors="replace")
                exit_code = result.get("StatusCode")
                success = (exit_code == 0) and not timed_out
                return SandboxRun(
                    success=success,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=timed_out,
                    error=None
                    if success
                    else ("Timed out" if timed_out else (stderr.strip() or "Execution failed")),
                )
            finally:
                if container is not None:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
