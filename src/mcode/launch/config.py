from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mcode.launch.models import default_config_path


@dataclass
class BlueVelaConfig:
    login: str = f"{os.environ.get('USER', 'user')}@login3.bluevela.rmf.ibm.com"
    workspace_root: str = f"/u/{os.environ.get('USER', 'user')}/mcode-launch"
    queue: str = "normal"
    group: str = "grp_runtime"
    shared_root: str = f"/proj/dmfexp/{os.environ.get('USER', 'user')}"
    hf_env: str = f"/u/{os.environ.get('USER', 'user')}/.config/mcode/hf-env.sh"
    podman_graphroot: str | None = None
    podman_runroot: str | None = None
    results_root: str | None = None

    def __post_init__(self) -> None:
        if self.podman_graphroot is None:
            self.podman_graphroot = f"{self.shared_root.rstrip('/')}/podman/graphroot"
        if self.podman_runroot is None:
            self.podman_runroot = f"{self.shared_root.rstrip('/')}/podman/runroot"


@dataclass
class LocalVllmConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class LocalOllamaConfig:
    host: str = "127.0.0.1"
    port: int = 11434
    keep_alive: str = "-1"
    num_parallel: int = 1
    max_queue: int = 512


@dataclass
class LaunchConfig:
    bluevela: BlueVelaConfig = field(default_factory=BlueVelaConfig)
    local_vllm: LocalVllmConfig = field(default_factory=LocalVllmConfig)
    local_ollama: LocalOllamaConfig = field(default_factory=LocalOllamaConfig)


def load_launch_config(path: Path | None = None) -> LaunchConfig:
    config_path = path or Path(os.environ.get("MCODE_LAUNCH_CONFIG", default_config_path()))
    if not config_path.exists():
        return LaunchConfig()

    data = tomllib.loads(config_path.read_text())
    bluevela = data.get("bluevela", {})
    local_vllm = data.get("local_vllm", {})
    local_ollama = data.get("local_ollama", {})
    shared_root = bluevela.get("shared_root", BlueVelaConfig.shared_root)
    return LaunchConfig(
        bluevela=BlueVelaConfig(
            login=bluevela.get("login", BlueVelaConfig.login),
            workspace_root=bluevela.get("workspace_root", BlueVelaConfig.workspace_root),
            queue=bluevela.get("queue", BlueVelaConfig.queue),
            group=bluevela.get("group", BlueVelaConfig.group),
            shared_root=shared_root,
            hf_env=bluevela.get("hf_env", BlueVelaConfig.hf_env),
            podman_graphroot=bluevela.get(
                "podman_graphroot", f"{shared_root.rstrip('/')}/podman/graphroot"
            ),
            podman_runroot=bluevela.get(
                "podman_runroot", f"{shared_root.rstrip('/')}/podman/runroot"
            ),
            results_root=bluevela.get("results_root"),
        ),
        local_vllm=LocalVllmConfig(
            host=local_vllm.get("host", LocalVllmConfig.host),
            port=local_vllm.get("port", LocalVllmConfig.port),
        ),
        local_ollama=LocalOllamaConfig(
            host=local_ollama.get("host", LocalOllamaConfig.host),
            port=local_ollama.get("port", LocalOllamaConfig.port),
            keep_alive=local_ollama.get("keep_alive", LocalOllamaConfig.keep_alive),
            num_parallel=local_ollama.get("num_parallel", LocalOllamaConfig.num_parallel),
            max_queue=local_ollama.get("max_queue", LocalOllamaConfig.max_queue),
        ),
    )
