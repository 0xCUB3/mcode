from __future__ import annotations

from pathlib import Path

from mcode.launch import service as service_module
from mcode.launch.models import (
    BenchSpec,
    LaunchSpec,
    LocalOllamaTargetSpec,
    LocalVllmTargetSpec,
    ReuseMode,
    ServingSpec,
    SyncSpec,
    TargetKind,
)
from mcode.launch.profiles import resolve_serving_profile
from mcode.launch.providers.local_ollama import (
    build_ollama_serve_command,
    build_ollama_warmup_command,
)
from mcode.launch.providers.local_vllm import build_local_vllm_command, build_local_vllm_reuse_key
from mcode.launch.state import LauncherState


def test_local_vllm_command_uses_profile_flags() -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-live", backend="openai"),
        serving=ServingSpec(
            engine="vllm",
            port=8000,
            tensor_parallel=2,
            data_parallel=1,
            api_server_count=1,
            max_model_len=32768,
            profile=resolve_serving_profile("Qwen/Qwen3.5-27B"),
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
        detach=False,
    )

    command = build_local_vllm_command(spec)

    assert "vllm serve" in command
    assert "qwen3_coder" in command
    assert "--port 8000" in command
    assert "--gpu-memory-utilization 0.9" in command


def test_local_vllm_reuse_key_changes_with_tp() -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-live", backend="openai"),
        serving=ServingSpec(
            engine="vllm",
            port=8000,
            tensor_parallel=1,
            data_parallel=1,
            api_server_count=1,
            max_model_len=32768,
            profile=resolve_serving_profile("Qwen/Qwen3.5-27B"),
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
        detach=False,
    )
    left = build_local_vllm_reuse_key(spec)
    spec.serving.tensor_parallel = 2
    right = build_local_vllm_reuse_key(spec)

    assert left != right


def test_local_vllm_reuse_key_changes_with_gpu_memory_utilization() -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-live", backend="openai"),
        serving=ServingSpec(
            engine="vllm",
            port=8000,
            tensor_parallel=1,
            data_parallel=1,
            api_server_count=1,
            max_model_len=32768,
            profile=resolve_serving_profile("Qwen/Qwen3.5-27B"),
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
        detach=False,
    )
    left = build_local_vllm_reuse_key(spec)
    spec.serving.gpu_memory_utilization = 0.4
    right = build_local_vllm_reuse_key(spec)

    assert left != right


def test_local_ollama_commands_include_keep_alive_and_parallelism() -> None:
    spec = LaunchSpec(
        target=LocalOllamaTargetSpec(kind=TargetKind.LOCAL_OLLAMA, host="127.0.0.1"),
        model="llama3.2",
        benchmark=BenchSpec(benchmark="swebench-live", backend="ollama"),
        serving=ServingSpec(
            engine="ollama",
            port=11434,
            keep_alive="-1",
            ollama_num_parallel=4,
            ollama_max_queue=512,
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
        detach=False,
    )

    serve = build_ollama_serve_command(spec)
    warm = build_ollama_warmup_command(spec)

    assert "OLLAMA_NUM_PARALLEL=4" in serve
    assert "OLLAMA_MAX_QUEUE=512" in serve
    assert '"keep_alive": -1' in warm


def test_local_ollama_attaches_to_existing_healthy_endpoint(tmp_path: Path) -> None:
    spec = LaunchSpec(
        target=LocalOllamaTargetSpec(kind=TargetKind.LOCAL_OLLAMA, host="127.0.0.1"),
        model="granite3.3:8b",
        benchmark=BenchSpec(benchmark="swebench-live", backend="ollama", limit=1),
        serving=ServingSpec(
            engine="ollama",
            port=11434,
            keep_alive="-1",
            ollama_num_parallel=1,
            ollama_max_queue=512,
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=False,
        follow=False,
        detach=False,
    )
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_health = service_module._endpoint_is_healthy
    original_launch_local = service_module._launch_local_benchmark
    try:
        service_module._endpoint_is_healthy = lambda base_url: base_url == "http://127.0.0.1:11434"
        service_module._launch_local_benchmark = (
            lambda *args, **kwargs: service_module.CommandResult(
                ok=True, message="bench", data={"run_id": "run-1"}
            )
        )
        result = service_module._launch_local_ollama(spec, state=state, state_path=state_path)
    finally:
        service_module._endpoint_is_healthy = original_health
        service_module._launch_local_benchmark = original_launch_local

    assert result.ok is True
    assert result.data["server_id"].startswith("server-")
    assert state.servers[0].status == "healthy"
    assert state.servers[0].metadata["discovered"] is True


def test_local_vllm_attaches_to_existing_healthy_endpoint(tmp_path: Path) -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-live", backend="openai", limit=1),
        serving=ServingSpec(
            engine="vllm",
            port=8000,
            tensor_parallel=1,
            data_parallel=1,
            api_server_count=1,
            max_model_len=32768,
            profile=resolve_serving_profile("Qwen/Qwen3.5-27B"),
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=False,
        follow=False,
        detach=False,
    )
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_health = service_module._endpoint_is_healthy
    original_launch_local = service_module._launch_local_benchmark
    try:
        service_module._endpoint_is_healthy = (
            lambda base_url: base_url == "http://127.0.0.1:8000/v1"
        )
        service_module._launch_local_benchmark = (
            lambda *args, **kwargs: service_module.CommandResult(
                ok=True, message="bench", data={"run_id": "run-1"}
            )
        )
        result = service_module._launch_local_vllm(spec, state=state, state_path=state_path)
    finally:
        service_module._endpoint_is_healthy = original_health
        service_module._launch_local_benchmark = original_launch_local

    assert result.ok is True
    assert result.data["server_id"].startswith("server-")
    assert state.servers[0].status == "healthy"
    assert state.servers[0].metadata["discovered"] is True
