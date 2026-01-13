from __future__ import annotations

import tempfile
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
        if self._client is not None:
            return self._client
        try:
            self._client = docker.from_env()
            return self._client
        except DockerException as e:  # pragma: no cover
            raise RuntimeError(
                "Docker is required for sandboxed execution, but the daemon is not reachable. "
                "Start Docker Desktop (or configure DOCKER_HOST) and retry."
            ) from e

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
