from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcode.execution.sandbox import SandboxRun


def _kill_pg(proc: subprocess.Popen) -> None:
    """Kill the process and its entire process group (children, grandchildren)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


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
        max_output_bytes_raw = os.environ.get("MCODE_SANDBOX_MAX_OUTPUT_BYTES", "1048576")
        try:
            max_output_bytes = int(max_output_bytes_raw)
        except ValueError:
            raise ValueError(
                f"MCODE_SANDBOX_MAX_OUTPUT_BYTES must be an int (got {max_output_bytes_raw!r})"
            )
        if max_output_bytes < 1024:
            raise ValueError("MCODE_SANDBOX_MAX_OUTPUT_BYTES must be >= 1024")

        with tempfile.TemporaryDirectory(prefix="mcode-process-sandbox-") as td:
            host_dir = Path(td)
            script = host_dir / "main.py"
            script.write_text(code, encoding="utf-8", errors="replace")

            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", str(script)],
                cwd=str(host_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={
                    "HOME": str(host_dir),
                    "LANG": "C.UTF-8",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "MPLBACKEND": "Agg",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )

            start = time.monotonic()
            timed_out = False
            output_exceeded = False

            stdout_b = bytearray()
            stderr_b = bytearray()

            try:
                assert proc.stdout is not None
                assert proc.stderr is not None

                import selectors

                sel = selectors.DefaultSelector()
                sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
                sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

                while sel.get_map():
                    elapsed = time.monotonic() - start
                    remaining = timeout_s - elapsed
                    if remaining <= 0:
                        timed_out = True
                        _kill_pg(proc)
                        break

                    for key, _ in sel.select(timeout=min(1.0, remaining)):
                        stream = key.data
                        chunk = key.fileobj.read(8192)
                        if not chunk:
                            try:
                                sel.unregister(key.fileobj)
                            except Exception:
                                pass
                            continue

                        if stream == "stdout":
                            stdout_b += chunk
                        else:
                            stderr_b += chunk

                        if (len(stdout_b) + len(stderr_b)) > max_output_bytes:
                            output_exceeded = True
                            _kill_pg(proc)
                            break

                    if output_exceeded:
                        break

                    if proc.poll() is not None:
                        # Drain remaining buffered output.
                        continue

            finally:
                try:
                    proc.wait(timeout=1)
                except Exception:
                    _kill_pg(proc)
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass

            stdout = bytes(stdout_b[:max_output_bytes]).decode("utf-8", errors="replace")
            stderr = bytes(stderr_b[:max_output_bytes]).decode("utf-8", errors="replace")

            if output_exceeded:
                trunc_note = "\n\n[mcode] Output exceeded limit; process killed.\n"
                if len(stdout) < max_output_bytes:
                    stdout = (stdout + trunc_note)[:max_output_bytes]
                else:
                    stdout = stdout[: max_output_bytes - len(trunc_note)] + trunc_note

            exit_code = proc.returncode
            success = (exit_code == 0) and not timed_out and not output_exceeded
            if timed_out:
                error = "Timed out"
            elif output_exceeded:
                error = "Output exceeded limit"
            else:
                error = None if success else (stderr.strip() or "Execution failed")

            return SandboxRun(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                error=error,
            )
