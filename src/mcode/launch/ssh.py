"""Thin SSH wrapper for Blue Vela.

The launcher needs to tell apart a remote command failure from a transport
failure. OpenSSH uses exit code 255 for the latter, so `SshResult` keeps that
case visible to callers.

Callers pass a complete command string. For config-heavy commands, they upload
`env.json` and let the shell script read it with `jq @sh`; Python does not
build long `export KEY=VALUE` strings.

Public API:

    SshClient(login: str, ssh_options: list[str] = ...)
    client.run(cmd: str, *, timeout: float) -> SshResult
    client.upload(src: Path, dst: str, *, timeout: float) -> None
    client.download(src: str, dst: Path, *, timeout: float) -> None
    client.download_tree(src: str, dst: Path, *, timeout: float) -> None

Integration-test hook: set `MCODE_TEST_SSH_LOGIN=user@host` to enable
real-cluster round-trip tests (gated in tests/launch/test_ssh.py).
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from mcode.launch.progress import TransportError

# Default SSH client options applied to every invocation. Chosen for:
# - BatchMode: fail fast when keys/agent aren't set (don't prompt).
# - ConnectTimeout: transport failure surfaces within 10 s, not minutes.
# - ServerAliveInterval/Countmax: detect dead connections during long commands.
# - ControlPath OFF: concurrent launches should not share SSH state.
DEFAULT_SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "ControlPath=none",
    "-o",
    "StrictHostKeyChecking=accept-new",
)

# Stderr patterns that unambiguously mean the transport failed (not a remote
# command error). Used to raise TransportError from run()/stream() so the
# progress UI can render ⚠ instead of a normal failure.
_TRANSPORT_FAIL_MARKERS = (
    "Connection refused",
    "Connection timed out",
    "Connection closed",
    "Connection reset by peer",
    "No route to host",
    "Network is unreachable",
    "Could not resolve hostname",
    "Permission denied (publickey",
    "Host key verification failed",
    "kex_exchange_identification",
    "ssh_exchange_identification",
    "port 22: Connection refused",
)


@dataclass
class SshResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    cmd: str = field(default="")

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> SshResult:
        if not self.ok:
            raise SshError(
                f"ssh exit {self.returncode}: {self.stderr.strip() or self.stdout.strip()[:200]}",
                result=self,
            )
        return self


class SshError(Exception):
    """Remote command returned non-zero. Transport was healthy."""

    def __init__(self, msg: str, *, result: SshResult | None = None) -> None:
        super().__init__(msg)
        self.result = result


def _is_transport_failure(result: subprocess.CompletedProcess | subprocess.TimeoutExpired) -> bool:
    if isinstance(result, subprocess.TimeoutExpired):
        return True  # treat hung connection as transport issue
    # OpenSSH returns 255 when the SSH client itself fails (connect, auth,
    # network). BUT remote commands can also legitimately exit 255 — only
    # classify as transport if stderr carries a known OpenSSH client marker.
    # Otherwise it's just a remote non-zero the caller can handle.
    if result.returncode != 255:
        return False
    stderr = result.stderr or ""
    return any(marker in stderr for marker in _TRANSPORT_FAIL_MARKERS)


def _transport_message(result: subprocess.CompletedProcess | subprocess.TimeoutExpired) -> str:
    if isinstance(result, subprocess.TimeoutExpired):
        return f"ssh timeout after {result.timeout:.0f}s"
    stderr = (result.stderr or "").strip()
    for marker in _TRANSPORT_FAIL_MARKERS:
        if marker in stderr:
            return marker
    return stderr.splitlines()[-1] if stderr else "ssh transport failure"


class SshClient:
    def __init__(
        self,
        login: str,
        *,
        ssh_options: tuple[str, ...] = DEFAULT_SSH_OPTIONS,
        ssh_binary: str = "ssh",
        scp_binary: str = "scp",
    ) -> None:
        self.login = login
        self._ssh_options = tuple(ssh_options)
        self._ssh_binary = ssh_binary
        self._scp_binary = scp_binary

    # --- running commands --------------------------------------------------
    def run(self, cmd: str, *, timeout: float = 60.0) -> SshResult:
        """Run `cmd` on the remote shell. Raises TransportError on transport
        failure; otherwise returns SshResult (caller can inspect `ok`/stderr
        or call `raise_for_status()`)."""
        argv = [self._ssh_binary, *self._ssh_options, self.login, cmd]
        import time

        start = time.monotonic()
        try:
            r = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as te:
            # time and transport error go together — raise.
            raise TransportError(_transport_message(te)) from te
        duration = time.monotonic() - start
        if _is_transport_failure(r):
            raise TransportError(_transport_message(r))
        return SshResult(
            returncode=r.returncode,
            stdout=r.stdout,
            stderr=r.stderr,
            duration_s=duration,
            cmd=shlex.join(argv),
        )

    # --- file transfer -----------------------------------------------------
    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        argv = [
            self._scp_binary,
            *self._ssh_options,
            str(src),
            f"{self.login}:{dst}",
        ]
        self._run_xfer(argv, timeout=timeout)

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            self._scp_binary,
            *self._ssh_options,
            f"{self.login}:{src}",
            str(dst),
        ]
        self._run_xfer(argv, timeout=timeout)

    def download_tree(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            self._scp_binary,
            "-r",
            *self._ssh_options,
            f"{self.login}:{src}",
            str(dst),
        ]
        self._run_xfer(argv, timeout=timeout)

    def _run_xfer(self, argv: list[str], *, timeout: float) -> None:
        try:
            r = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as te:
            raise TransportError(_transport_message(te)) from te
        if _is_transport_failure(r):
            raise TransportError(_transport_message(r))
        if r.returncode != 0:
            raise SshError(
                f"scp exit {r.returncode}: {r.stderr.strip() or r.stdout.strip()[:200]}",
            )
