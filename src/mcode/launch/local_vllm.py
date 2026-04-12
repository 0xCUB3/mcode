"""Local vLLM target.

Simplest end-to-end path: subprocess `vllm serve` on the current host, poll
/v1/models for readiness, hand the endpoint to the benchmark runner.

Public surface:

    launch(spec: LaunchSpec, reporter: Reporter) -> RunRecord
    doctor(cfg: LocalVllmConfig) -> list[Check]
    stop(record_id: str) -> bool
    refresh(record: RunRecord) -> RunRecord

Phases: submit -> starting -> ready -> running -> done.
"""

from __future__ import annotations
