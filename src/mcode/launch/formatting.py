"""Formatters for `mcode launch status`.

Extracted from `cmd_status` so the CLI handler is a thin shim and the
output shape is unit-testable. JSON output bytes are unchanged from the
pre-extraction implementation — `tests/launch/test_cli.py` still passes
without modification.
"""

from __future__ import annotations

from typing import Any

from mcode.launch.state import State

_SERVER_MARKERS = {"healthy": "✓", "pending": "·", "failed": "✗", "stopped": "—"}


def format_status_json(s: State, *, raw: bool) -> dict[str, Any]:
    """Build the dict that `mcode launch status --json` serializes.

    Wave 1 added `shard_pids` to `RunRecord`; the `shards` count falls back
    to len(shard_pids) when shard_job_ids is empty so local sharded bench
    runs report a sensible count.
    """
    return {
        "servers": [
            {
                "id": srv.id,
                "target": srv.target.value,
                "endpoint": srv.endpoint,
                "model": srv.model,
                "status": srv.status,
                "job_id": srv.job_id,
                **({"lsf_state": srv.metadata.get("lsf_state")} if raw else {}),
            }
            for srv in s.servers
        ],
        "runs": [
            {
                "id": r.id,
                "target": r.target.value,
                "status": r.status.value,
                "benchmark": r.benchmark,
                "server_id": r.server_id,
                "shards": len(r.shard_job_ids) or len(r.shard_pids),
            }
            for r in s.runs
        ],
    }


def format_status_lines(s: State) -> list[str]:
    """Build the human-readable line list for non-JSON `mcode launch status`.

    Returns lines without trailing newlines; caller prints them. Empty list
    when there are no servers and no runs (caller prints the "nothing
    recorded" message itself for parity with the pre-extraction shape).
    """
    if not s.servers and not s.runs:
        return []
    lines: list[str] = []
    if s.servers:
        lines.append("servers:")
        for srv in s.servers:
            marker = _SERVER_MARKERS.get(srv.status, "?")
            lines.append(
                f"  {marker} {srv.id}  [{srv.target.value}]  {srv.model}"
                f"  {srv.endpoint or '(no endpoint yet)'}  ({srv.status})"
            )
    if s.runs:
        lines.append("runs:")
        for r in s.runs:
            lines.append(f"  - {r.id}  [{r.target.value}]  {r.benchmark}  ({r.status.value})")
    return lines


__all__ = ["format_status_json", "format_status_lines"]
