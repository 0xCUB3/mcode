"""Launcher config (TOML at ~/.config/mcode/launch.toml).

Single flat schema; three tables. No skula-specific defaults here — values
that depend on the user's Blue Vela account come from doctor --init, which
probes the user's own account. See the plan's "Portability" section.

    [bluevela]
    login = "<user>@<login-host>"
    workspace_root = "<$HOME>/mcode-launch"
    queue_order = ["normal"]     # confirmed via Phase 0.5 probe (login3, grp_runtime)
    group = ""                   # must be set by doctor --init
    shared_root = "<$HOME>/mcode-shared"
    hf_env = "<$HOME>/.config/mcode/hf-env.sh"
    gpu_mode = "exclusive_process"  # Phase 0.5 probe showed shared is deprecated on this cluster

    [bluevela.podman]
    # graphroot_base = "..."     # optional; shell derives sensible default
    # runroot_base = "..."

    [local_vllm]
    port = 8000

    [local_ollama]
    host = "127.0.0.1"
"""

from __future__ import annotations
