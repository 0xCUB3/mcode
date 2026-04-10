from __future__ import annotations

import shlex

from mcode.launch.models import CommandResult, LaunchSpec, LocalVllmTargetSpec


def build_local_vllm_reuse_key(spec: LaunchSpec) -> str:
    return "|".join(
        [
            "local-vllm",
            spec.model,
            f"tp={spec.serving.tensor_parallel}",
            f"dp={spec.serving.data_parallel}",
            f"api={spec.serving.api_server_count}",
            f"port={spec.serving.port}",
            f"mem={spec.serving.gpu_memory_utilization}",
            f"profile={spec.serving.profile.name}",
        ]
    )


def build_local_vllm_command(spec: LaunchSpec) -> str:
    target = spec.target
    assert isinstance(target, LocalVllmTargetSpec)
    flags = " ".join(shlex.quote(flag) for flag in spec.serving.profile.flags)
    return (
        f"vllm serve {shlex.quote(spec.model)} "
        f"--port {spec.serving.port} "
        f"--max-model-len {spec.serving.max_model_len} "
        f"--gpu-memory-utilization {spec.serving.gpu_memory_utilization} "
        f"--tensor-parallel-size {spec.serving.tensor_parallel} "
        f"--api-server-count {spec.serving.api_server_count} "
        f"{flags}"
    ).strip()


def local_vllm_doctor_result(target: LocalVllmTargetSpec) -> CommandResult:
    return CommandResult(
        ok=True,
        message="Local vLLM doctor checks pending live command execution.",
        data={"host": target.host},
    )
