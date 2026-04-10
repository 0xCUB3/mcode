from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class TargetKind(StrEnum):
    BLUEVELA = "bluevela"
    LOCAL_VLLM = "local-vllm"
    LOCAL_OLLAMA = "local-ollama"
    OPENAI_COMPATIBLE = "openai-compatible"


class SyncMode(StrEnum):
    GIT_OVERLAY = "git-overlay"
    GIT_REF = "git-ref"
    WORKING_TREE = "working-tree"


class ReuseMode(StrEnum):
    PREFER = "prefer"
    FORCE_NEW = "force-new"
    STOP_AND_REPLACE = "stop-and-replace"


@dataclass
class ServingProfile:
    name: str = "default"
    flags: list[str] = field(default_factory=list)


@dataclass
class BenchSpec:
    benchmark: str = "swebench-live"
    backend: str = "openai"
    split: str = "verified"
    dataset: str | None = None
    loop_budget: int = 15
    timeout: int = 1800
    parallelism: int = 1
    limit: int | None = None
    task_ids: str | None = None
    mem_limit: str = "4g"
    pids_limit: int = 512
    n_samples: int = 1


@dataclass
class SyncSpec:
    mode: SyncMode = SyncMode.GIT_OVERLAY
    ref: str = "HEAD"
    check: bool = False
    apply: bool = False
    bootstrap_key: str = "uv-sync:swebench,datasets"


@dataclass
class ServingSpec:
    engine: str
    port: int
    tensor_parallel: int = 1
    data_parallel: int = 1
    api_server_count: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    image: str | None = None
    profile: ServingProfile = field(default_factory=ServingProfile)
    keep_alive: str | None = None
    ollama_num_parallel: int | None = None
    ollama_max_queue: int | None = None


@dataclass
class BaseTargetSpec:
    kind: TargetKind


@dataclass
class BlueVelaTargetSpec(BaseTargetSpec):
    login: str
    workspace_root: str
    queue: str
    group: str
    shared_root: str
    hf_env: str
    podman_graphroot: str | None = None
    podman_runroot: str | None = None
    results_root: str | None = None


@dataclass
class LocalVllmTargetSpec(BaseTargetSpec):
    host: str = "127.0.0.1"


@dataclass
class LocalOllamaTargetSpec(BaseTargetSpec):
    host: str = "127.0.0.1"


@dataclass
class OpenAICompatibleTargetSpec(BaseTargetSpec):
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class LaunchSpec:
    target: BaseTargetSpec
    model: str
    benchmark: BenchSpec
    serving: ServingSpec
    sync: SyncSpec
    reuse: ReuseMode = ReuseMode.PREFER
    json_mode: bool = False
    yes: bool = False
    follow: bool = False


@dataclass
class WorkspaceHandle:
    signature: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerHandle:
    id: str
    target: str
    reuse_key: str
    endpoint: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None


@dataclass
class RunHandle:
    id: str
    target: str
    benchmark: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None


@dataclass
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


DEFAULT_BOOTSTRAP_KEY = SyncSpec.bootstrap_key


def default_state_path() -> Path:
    return Path.home() / ".config" / "mcode" / "launch-state.json"


def default_config_path() -> Path:
    return Path.home() / ".config" / "mcode" / "launch.toml"
