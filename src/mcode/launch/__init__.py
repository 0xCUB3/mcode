"""mcode launcher: submit vLLM + benchmark jobs to Blue Vela LSF or run locally.

Public surface (imported by mcode.cli):

    launch_bluevela(spec, reporter) -> RunRecord
    launch_local_vllm(spec, reporter) -> RunRecord
    launch_local_ollama(spec, reporter) -> RunRecord

    doctor(target) -> list[Check]
    stop(record_id) -> bool
    fetch(record_id, dest) -> Path
    refresh(record) -> RunRecord   # re-query LSF/process and update state in place

Each target module is self-contained. There is no unified router — cli.py
dispatches per target. See /Users/skula/.claude/plans/fancy-jingling-bumblebee.md
for the authoritative design.
"""

from __future__ import annotations

from mcode.launch.models import (
    LaunchError,
    LaunchSpec,
    Phase,
    RunRecord,
    ServerRecord,
    ServingProfile,
)

__all__ = [
    "LaunchError",
    "LaunchSpec",
    "Phase",
    "RunRecord",
    "ServerRecord",
    "ServingProfile",
]
