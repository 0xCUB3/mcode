from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from mcode.execution.sandbox import SandboxRun


class ProcessSandbox:
    """Run untrusted code directly as a local subprocess.

    This is less isolated than `DockerSandbox` (no network isolation, no filesystem sandboxing),
    but it works in environments where Docker isn't available (e.g. inside Kubernetes pods).
    """

    def check_available(self) -> None:
        return

    def ensure_image(self) -> None:
        return

    def run_python(self, code: str, *, timeout_s: int = 60) -> SandboxRun:
        with tempfile.TemporaryDirectory(prefix="mcode-process-sandbox-") as td:
            host_dir = Path(td)
            script = host_dir / "main.py"
            script.write_text(code, encoding="utf-8", errors="replace")

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", str(script)],
                    cwd=str(host_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    encoding="utf-8",
                    errors="replace",
                    stdin=subprocess.DEVNULL,
                    env={
                        "HOME": str(host_dir),
                        "LANG": "C.UTF-8",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
            except subprocess.TimeoutExpired as e:
                stdout = (
                    (e.stdout or "").decode("utf-8", errors="replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
                stderr = (
                    (e.stderr or "").decode("utf-8", errors="replace")
                    if isinstance(e.stderr, bytes)
                    else (e.stderr or "")
                )
                return SandboxRun(
                    success=False,
                    exit_code=None,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                    error="Timed out",
                )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = int(proc.returncode)
            success = exit_code == 0
            return SandboxRun(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                error=None if success else (stderr.strip() or "Execution failed"),
            )
