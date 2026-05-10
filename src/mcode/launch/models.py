"""Dataclasses for the launcher.

Deliberately slim. Fields are only what the state file, CLI, and progress UI
actually consume — no abstraction for its own sake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mcode.ui.errors import MCodeError


class Target(StrEnum):
    BLUEVELA = "bluevela"
    LOCAL_VLLM = "local-vllm"
    LOCAL_OLLAMA = "local-ollama"


class RunStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


class PhaseStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class Phase:
    key: str
    label: str


@dataclass
class ServingProfile:
    """Per-model knowledge: flags, sizing, env, template, image, min vLLM.

    This is the single source of truth for how a given model is served.
    The launcher never hardcodes model specifics outside this dataclass.
    """

    name: str
    flags: list[str]
    tensor_parallel: int
    max_model_len: int
    extra_env: dict[str, str] = field(default_factory=dict)
    chat_template: str | None = (
        None  # bundled resource filename, e.g. "tool_chat_template_gemma4.jinja"
    )
    min_vllm: str | None = None  # semver or "nightly@<sha>"
    image: str | None = None  # container image override


@dataclass
class LaunchSpec:
    target: Target
    model: str
    profile: ServingProfile
    benchmark: str = "swebench-live"
    shards: int = 1
    limit: int | None = None
    task_ids: list[str] | None = None
    follow: bool = False
    json_mode: bool = False
    reuse_server: str | None = None  # opt-in only
    allow_old_vllm: bool = False
    offline: bool = False  # skip refresh() on read commands


@dataclass
class ServerRecord:
    id: str
    target: Target
    endpoint: str
    model: str
    config_hash: str  # full hash of spec; reuse requires exact match
    job_id: str | None = None
    log_path: str | None = None
    started_at: str | None = None
    refs: list[str] = field(default_factory=list)  # run ids currently using this server
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    id: str
    target: Target
    benchmark: str
    status: RunStatus = RunStatus.SUBMITTED
    server_id: str | None = None
    shard_job_ids: list[str] = field(default_factory=list)
    shard_states: dict[str, str] = field(default_factory=dict)  # job id -> raw LSF state
    lsf_state: str | None = None  # raw LSF state for the coordinating job (optional)
    log_paths: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # additive bench-run lifecycle fields. Old state files load fine because
    # _filter_fields drops unknown keys and these all have defaults.
    started_at: float | None = None
    ended_at: float | None = None
    db_path: str | None = None
    shard_pids: list[int] = field(default_factory=list)
    remote: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, int] = field(default_factory=dict)


class LaunchError(MCodeError):
    """User-facing launcher error with actionable remediation.

    Subclass of mcode.ui.errors.MCodeError so the unified print_error
    formatter handles it. Fields and constructor signature unchanged.

    The CLI formats this as:

        ✗ {what}
          why:  {why}
          next: {next}
          logs: {logs}
    """


@dataclass
class Check:
    """One row in `mcode doctor` output."""

    name: str
    ok: bool
    detail: str = ""
    next: str = ""  # remediation hint if not ok


def default_state_path() -> Path:
    return Path.home() / ".config" / "mcode" / "launch-state.json"


def default_config_path() -> Path:
    return Path.home() / ".config" / "mcode" / "launch.toml"
