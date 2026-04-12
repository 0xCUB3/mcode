"""Thin SSH wrapper for Blue Vela.

Contract:

    run_ssh(login: str, command: str, *, timeout: float) -> SshResult
    run_ssh_stream(login: str, command: str, *, timeout: float) -> Iterator[str]
    upload(login: str, src: Path, dst: str) -> None
    download(login: str, src: str, dst: Path) -> None

Every call carries a timeout. Transport failures (exit code 255, ssh_exchange
errors, no route to host) are distinguished from remote-command non-zero exits
via a typed SshResult — the progress UI uses this to render ⚠ ssh unreachable
rather than misclassifying as "remote quiet".

No shell quoting is done in Python: callers pass a pre-formed command string.
Arguments that need interpolation go through env.json + jq on the remote side.
"""

from __future__ import annotations
