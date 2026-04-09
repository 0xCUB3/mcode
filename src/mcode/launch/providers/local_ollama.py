from __future__ import annotations

import json
import shlex

from mcode.launch.models import CommandResult, LaunchSpec, LocalOllamaTargetSpec


def build_ollama_serve_command(spec: LaunchSpec) -> str:
    target = spec.target
    assert isinstance(target, LocalOllamaTargetSpec)
    num_parallel = spec.serving.ollama_num_parallel or 1
    max_queue = spec.serving.ollama_max_queue or 512
    return (
        f"OLLAMA_HOST={shlex.quote(target.host)}:{spec.serving.port} "
        f"OLLAMA_NUM_PARALLEL={num_parallel} "
        f"OLLAMA_MAX_QUEUE={max_queue} "
        f"OLLAMA_KEEP_ALIVE={shlex.quote(spec.serving.keep_alive or '-1')} "
        "ollama serve"
    )


def build_ollama_warmup_command(spec: LaunchSpec) -> str:
    target = spec.target
    assert isinstance(target, LocalOllamaTargetSpec)
    keep_alive = spec.serving.keep_alive or "-1"
    payload = json.dumps({"model": spec.model, "keep_alive": int(keep_alive)})
    return (
        f"curl -s http://{shlex.quote(target.host)}:{spec.serving.port}/api/generate -d '{payload}'"
    )


def local_ollama_doctor_result(target: LocalOllamaTargetSpec) -> CommandResult:
    return CommandResult(
        ok=True,
        message="Local Ollama doctor checks pending live command execution.",
        data={"host": target.host},
    )
