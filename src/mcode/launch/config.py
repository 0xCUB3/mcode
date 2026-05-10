"""Launcher config (TOML at ~/.config/mcode/launch.toml).

Schema:

    [bluevela]
    login = "<user>@<login-host>"
    workspace_root = "<$HOME>/mcode-launch"
    shared_root = "<$HOME>/mcode-shared"
    queue_order = ["normal"]       # populated by doctor --init from real bqueues
    group = ""                     # must be set (by doctor --init or manually)
    gpu_mode = "exclusive_process"
    hf_env = "<$HOME>/.config/mcode/hf-env.sh"

    [bluevela.podman]
    # graphroot_base = "..."      # optional; shell derives sensible default
    # runroot_base  = "..."

    [local_vllm]
    port = 8000

    [local_ollama]
    host = "127.0.0.1"
    port = 11434

All paths in the file are raw strings (no Python-side interpolation of
$HOME/$USER). The remote shell expands them. Local paths expand via
os.path.expanduser at read time.

Zero developer-specific defaults here. Values that depend on a Blue Vela
account come from doctor --init probing that account.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mcode.launch.models import LaunchError, default_config_path


@dataclass
class BluevelaPodmanConfig:
    graphroot_base: str | None = None
    runroot_base: str | None = None


@dataclass
class BluevelaConfig:
    login: str = ""
    workspace_root: str = ""
    shared_root: str = ""
    queue_order: list[str] = field(default_factory=lambda: ["normal"])
    group: str = ""
    gpu_mode: str = "exclusive_process"
    hf_env: str = ""
    podman: BluevelaPodmanConfig = field(default_factory=BluevelaPodmanConfig)


@dataclass
class LocalVllmConfig:
    port: int = 8000


@dataclass
class LocalOllamaConfig:
    host: str = "127.0.0.1"
    port: int = 11434


@dataclass
class LaunchConfig:
    bluevela: BluevelaConfig = field(default_factory=BluevelaConfig)
    local_vllm: LocalVllmConfig = field(default_factory=LocalVllmConfig)
    local_ollama: LocalOllamaConfig = field(default_factory=LocalOllamaConfig)
    source: Path | None = None  # where this config was loaded from


def _path(value: str | None) -> str:
    if not value:
        return ""
    return os.path.expanduser(value)


def _parse_bluevela(raw: dict) -> BluevelaConfig:
    podman_raw = raw.get("podman") or {}
    if not isinstance(podman_raw, dict):
        raise LaunchError(
            what="invalid [bluevela.podman] section",
            why="expected a TOML table",
            next="see docs/bluevela.md for the schema",
        )
    return BluevelaConfig(
        login=str(raw.get("login", "") or ""),
        workspace_root=_path(raw.get("workspace_root")),
        shared_root=_path(raw.get("shared_root")),
        queue_order=list(raw.get("queue_order", ["normal"]) or ["normal"]),
        group=str(raw.get("group", "") or ""),
        gpu_mode=str(raw.get("gpu_mode", "exclusive_process") or "exclusive_process"),
        hf_env=_path(raw.get("hf_env")),
        podman=BluevelaPodmanConfig(
            graphroot_base=podman_raw.get("graphroot_base"),
            runroot_base=podman_raw.get("runroot_base"),
        ),
    )


def _parse_local_vllm(raw: dict) -> LocalVllmConfig:
    return LocalVllmConfig(port=int(raw.get("port", 8000)))


def _parse_local_ollama(raw: dict) -> LocalOllamaConfig:
    return LocalOllamaConfig(
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 11434)),
    )


def load(path: Path | None = None) -> LaunchConfig:
    """Load config. Returns all-defaults if the file doesn't exist.

    Missing file is intentional — lets `doctor --init` run on a brand-new
    account without an error. Malformed file is a hard error.
    """
    cfg_path = path or Path(os.environ.get("MCODE_LAUNCH_CONFIG", default_config_path()))
    if not cfg_path.exists():
        return LaunchConfig(source=None)
    try:
        with open(cfg_path, "rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise LaunchError(
            what=f"config at {cfg_path} is not valid TOML",
            why=str(e),
            next=(
                f"fix the TOML syntax, or `mv {cfg_path} {cfg_path}.bak` "
                "and run `mcode launch doctor --init`"
            ),
        ) from e
    return LaunchConfig(
        bluevela=_parse_bluevela(raw.get("bluevela") or {}),
        local_vllm=_parse_local_vllm(raw.get("local_vllm") or {}),
        local_ollama=_parse_local_ollama(raw.get("local_ollama") or {}),
        source=cfg_path,
    )


def save(cfg: LaunchConfig, path: Path | None = None) -> Path:
    """Write cfg back as TOML. Creates parent dirs as needed."""
    dst = path or cfg.source or default_config_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    bv = cfg.bluevela
    lines.append("[bluevela]")
    lines.append(f'login = "{bv.login}"')
    lines.append(f'workspace_root = "{bv.workspace_root}"')
    lines.append(f'shared_root = "{bv.shared_root}"')
    lines.append(f"queue_order = [{', '.join(_toml_str(q) for q in bv.queue_order)}]")
    lines.append(f'group = "{bv.group}"')
    lines.append(f'gpu_mode = "{bv.gpu_mode}"')
    lines.append(f'hf_env = "{bv.hf_env}"')
    if bv.podman.graphroot_base or bv.podman.runroot_base:
        lines.append("")
        lines.append("[bluevela.podman]")
        if bv.podman.graphroot_base:
            lines.append(f'graphroot_base = "{bv.podman.graphroot_base}"')
        if bv.podman.runroot_base:
            lines.append(f'runroot_base = "{bv.podman.runroot_base}"')
    lines.append("")
    lines.append("[local_vllm]")
    lines.append(f"port = {cfg.local_vllm.port}")
    lines.append("")
    lines.append("[local_ollama]")
    lines.append(f'host = "{cfg.local_ollama.host}"')
    lines.append(f"port = {cfg.local_ollama.port}")
    lines.append("")
    dst.write_text("\n".join(lines))
    return dst


def _toml_str(s: str) -> str:
    # Minimal TOML string quoting: we don't allow embedded quotes in queue
    # names (they'd be nonsensical on LSF anyway). If they sneak in, hard-fail.
    if '"' in s or "\n" in s:
        raise LaunchError(
            what=f"invalid queue name: {s!r}",
            why="queue names may not contain double-quotes or newlines",
            next="fix queue_order in your launch.toml",
        )
    return f'"{s}"'


def validate_for_bluevela(cfg: LaunchConfig) -> list[str]:
    """Return a list of human-readable error strings; empty list = ok.

    Called before any bsub to catch misconfiguration early with actionable
    messages (see LaunchError's what/why/next formatting).
    """
    bv = cfg.bluevela
    errs: list[str] = []
    if not bv.login:
        errs.append("bluevela.login not set — run `mcode launch doctor bluevela --init`")
    if "@" not in bv.login:
        errs.append(f"bluevela.login {bv.login!r} is not in user@host form")
    if not bv.group:
        errs.append("bluevela.group not set — `-G <group>` is required on every bsub")
    if not bv.queue_order:
        errs.append("bluevela.queue_order is empty")
    if bv.gpu_mode not in ("shared", "exclusive_process"):
        errs.append(f"bluevela.gpu_mode {bv.gpu_mode!r} must be 'shared' or 'exclusive_process'")
    if not bv.workspace_root:
        errs.append("bluevela.workspace_root not set")
    if not bv.shared_root:
        errs.append("bluevela.shared_root not set")
    return errs
