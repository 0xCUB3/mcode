from __future__ import annotations

import signal
from pathlib import Path
from subprocess import CompletedProcess

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
from mcode.launch.state import LauncherState, ServerHandle, update_state


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
    )
    left = build_local_vllm_reuse_key(spec)
    spec.serving.gpu_memory_utilization = 0.4
    right = build_local_vllm_reuse_key(spec)

    assert left != right


def test_local_launch_parallelism_spawns_one_shard_per_process(tmp_path: Path) -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-lite", backend="openai", parallelism=3, limit=1),
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
    )
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    launched: list[str] = []
    original_health = service_module._endpoint_is_healthy
    original_wait = service_module._wait_for_endpoint
    original_popen = service_module.subprocess.Popen

    class _Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def fake_popen(command, *args, **kwargs):
        del args, kwargs
        launched.append(command)
        return _Process(1000 + len(launched))

    try:
        service_module._endpoint_is_healthy = (
            lambda base_url: base_url == "http://127.0.0.1:8000/v1"
        )
        service_module._wait_for_endpoint = lambda base_url: None
        service_module.subprocess.Popen = fake_popen
        result = service_module._launch_local_vllm(spec, state=state, state_path=state_path)
    finally:
        service_module._endpoint_is_healthy = original_health
        service_module._wait_for_endpoint = original_wait
        service_module.subprocess.Popen = original_popen

    assert result.ok is True
    run = state.runs[0]
    assert run.metadata["pids"] == [1001, 1002, 1003]
    assert len(run.metadata["commands"]) == 3
    assert len(run.metadata["db_paths"]) == 3
    assert len(run.metadata["log_paths"]) == 3
    assert "--dataset SWE-bench/SWE-bench_Lite" in run.metadata["commands"][0]
    assert "--shard-count 3 --shard-index 0" in run.metadata["commands"][0]
    assert "--shard-count 3 --shard-index 1" in run.metadata["commands"][1]
    assert "--shard-count 3 --shard-index 2" in run.metadata["commands"][2]
    assert result.data["server_id"].startswith("server-")


def test_stop_run_kills_all_local_shard_pids(tmp_path: Path) -> None:
    state = LauncherState(
        runs=[
            service_module.RunHandle(
                id="run-1",
                target=TargetKind.LOCAL_VLLM.value,
                benchmark="swebench-live",
                status="running",
                metadata={"pids": [111, 222, 333]},
                log_path=str(tmp_path / "benchmark.log"),
            )
        ]
    )
    state_path = tmp_path / "launch-state.json"
    original_kill = service_module.os.kill
    killed: list[tuple[int, int]] = []
    try:
        service_module.os.kill = lambda pid, sig: killed.append((pid, sig))
        result = service_module._stop_run(state.runs[0], state_path, state=state)
    finally:
        service_module.os.kill = original_kill

    assert result.ok is True
    assert [pid for pid, _ in killed] == [111, 222, 333]


def test_stop_run_ignores_missing_local_shard_pids(tmp_path: Path) -> None:
    state = LauncherState(
        runs=[
            service_module.RunHandle(
                id="run-1",
                target=TargetKind.LOCAL_VLLM.value,
                benchmark="swebench-live",
                status="running",
                metadata={"pids": [111, 222]},
                log_path=str(tmp_path / "benchmark.log"),
            )
        ]
    )
    state_path = tmp_path / "launch-state.json"
    original_kill = service_module.os.kill
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        del sig
        killed.append(pid)
        if pid == 111:
            raise ProcessLookupError

    try:
        service_module.os.kill = fake_kill
        result = service_module._stop_run(state.runs[0], state_path, state=state)
    finally:
        service_module.os.kill = original_kill

    assert result.ok is True
    assert killed == [111, 222]


def test_local_stop_and_replace_uses_sigterm(tmp_path: Path) -> None:
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
        reuse=ReuseMode.STOP_AND_REPLACE,
        json_mode=False,
        yes=False,
        follow=False,
    )
    state = LauncherState(
        servers=[
            ServerHandle(
                id="server-1",
                target=TargetKind.LOCAL_VLLM.value,
                reuse_key=build_local_vllm_reuse_key(spec),
                endpoint="http://127.0.0.1:8000/v1",
                status="healthy",
                metadata={"pid": 4321},
                log_path=None,
            )
        ]
    )
    state_path = tmp_path / "launch-state.json"
    original_kill = service_module.os.kill
    original_which = service_module.shutil.which
    killed: list[tuple[int, int]] = []
    try:
        service_module.os.kill = lambda pid, sig: killed.append((pid, sig))
        service_module.shutil.which = lambda executable: None
        service_module._resolve_local_server(
            spec,
            state=state,
            state_path=state_path,
            reuse_key=build_local_vllm_reuse_key(spec),
            endpoint="http://127.0.0.1:8000/v1",
            target=TargetKind.LOCAL_VLLM,
            executable="vllm",
            command="vllm serve",
        )
    finally:
        service_module.os.kill = original_kill
        service_module.shutil.which = original_which

    assert killed == [(4321, signal.SIGTERM)]


def test_launch_attach_json_returns_metadata_for_local_run(tmp_path: Path) -> None:
    state = LauncherState(
        runs=[
            service_module.RunHandle(
                id="run-1",
                target=TargetKind.LOCAL_VLLM.value,
                benchmark="swebench-live",
                status="running",
                metadata={"log_paths": [str(tmp_path / "a.log"), str(tmp_path / "b.log")]},
                log_path=str(tmp_path / "a.log"),
            )
        ]
    )
    state_path = tmp_path / "launch-state.json"
    update_state(state_path, lambda current: setattr(current, "runs", state.runs))

    result = service_module.launch_attach("run-1", follow=False, state_path=state_path)

    assert result.ok is True
    assert result.message == str(tmp_path / "a.log")


def test_launch_attach_follows_all_local_logs(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.LOCAL_VLLM.value,
        benchmark="swebench-live",
        status="running",
        metadata={"log_paths": [str(tmp_path / "a.log"), str(tmp_path / "b.log")]},
        log_path=str(tmp_path / "a.log"),
    )
    state = LauncherState(runs=[run])
    state_path = tmp_path / "launch-state.json"
    update_state(state_path, lambda current: setattr(current, "runs", state.runs))
    original_run = service_module.subprocess.run
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    try:
        service_module.subprocess.run = fake_run
        result = service_module.launch_attach("run-1", state_path=state_path)
    finally:
        service_module.subprocess.run = original_run

    assert result.ok is True
    assert commands[0][:3] == ["tail", "-n", "20"]
    assert str(tmp_path / "a.log") in commands[0]
    assert str(tmp_path / "b.log") in commands[0]


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
    )

    serve = build_ollama_serve_command(spec)
    warm = build_ollama_warmup_command(spec)

    assert "OLLAMA_NUM_PARALLEL=4" in serve
    assert "OLLAMA_MAX_QUEUE=512" in serve
    assert '"keep_alive": -1' in warm


def test_local_ollama_warmup_accepts_duration_keep_alive() -> None:
    spec = LaunchSpec(
        target=LocalOllamaTargetSpec(kind=TargetKind.LOCAL_OLLAMA, host="127.0.0.1"),
        model="llama3.2",
        benchmark=BenchSpec(benchmark="swebench-live", backend="ollama"),
        serving=ServingSpec(
            engine="ollama",
            port=11434,
            keep_alive="5m",
            ollama_num_parallel=1,
            ollama_max_queue=512,
        ),
        sync=SyncSpec(),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
    )

    warm = build_ollama_warmup_command(spec)

    assert '"keep_alive": "5m"' in warm


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


def test_local_vllm_waits_for_pending_server_instead_of_spawning(tmp_path: Path) -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-lite", backend="openai", parallelism=1, limit=1),
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
    )
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    pending = ServerHandle(
        id="server-pending",
        target=TargetKind.LOCAL_VLLM.value,
        reuse_key=build_local_vllm_reuse_key(spec),
        endpoint="http://127.0.0.1:8000/v1",
        status="pending",
        metadata={"command": "vllm serve", "pid": 4321, "created_at": 1.0},
        log_path=str(tmp_path / "pending.log"),
    )
    update_state(state_path, lambda current: current.servers.append(pending))
    original_health = service_module._endpoint_is_healthy
    original_pid_is_alive = service_module._pid_is_alive
    original_wait = service_module._wait_for_endpoint
    original_popen = service_module.subprocess.Popen
    original_launch_local = service_module._launch_local_benchmark
    original_which = service_module.shutil.which
    checks = {"count": 0}

    def fake_health(base_url: str) -> bool:
        assert base_url == "http://127.0.0.1:8000/v1"
        checks["count"] += 1
        return checks["count"] >= 2

    try:
        service_module._endpoint_is_healthy = fake_health
        service_module._pid_is_alive = lambda pid: pid == 4321
        service_module._wait_for_endpoint = lambda base_url: None
        service_module._launch_local_benchmark = (
            lambda *args, **kwargs: service_module.CommandResult(
                ok=True, message="bench", data={"run_id": "run-1"}
            )
        )
        service_module.shutil.which = lambda executable: executable
        service_module.subprocess.Popen = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not spawn a second local server")
        )
        result = service_module._launch_local_vllm(spec, state=state, state_path=state_path)
    finally:
        service_module._endpoint_is_healthy = original_health
        service_module._pid_is_alive = original_pid_is_alive
        service_module._wait_for_endpoint = original_wait
        service_module._launch_local_benchmark = original_launch_local
        service_module.shutil.which = original_which
        service_module.subprocess.Popen = original_popen

    assert result.ok is True
    assert result.data["server_id"] == "server-pending"
    assert state.servers[0].id == "server-pending"
    assert state.servers[0].status == "healthy"


def test_local_vllm_replaces_stale_pending_server(tmp_path: Path) -> None:
    spec = LaunchSpec(
        target=LocalVllmTargetSpec(kind=TargetKind.LOCAL_VLLM, host="127.0.0.1"),
        model="Qwen/Qwen3.5-27B",
        benchmark=BenchSpec(benchmark="swebench-lite", backend="openai", parallelism=1, limit=1),
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
    )
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    stale = ServerHandle(
        id="server-stale",
        target=TargetKind.LOCAL_VLLM.value,
        reuse_key=build_local_vllm_reuse_key(spec),
        endpoint="http://127.0.0.1:8000/v1",
        status="pending",
        metadata={"command": "vllm serve", "created_at": 0.0},
        log_path=str(tmp_path / "stale.log"),
    )
    update_state(state_path, lambda current: current.servers.append(stale))
    original_health = service_module._endpoint_is_healthy
    original_pid_is_alive = service_module._pid_is_alive
    original_wait = service_module._wait_for_endpoint
    original_popen = service_module.subprocess.Popen
    original_launch_local = service_module._launch_local_benchmark
    original_which = service_module.shutil.which
    original_time = service_module.time.time
    launched: list[str] = []

    class _Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    try:
        service_module._endpoint_is_healthy = lambda base_url: False
        service_module._pid_is_alive = lambda pid: False
        service_module._wait_for_endpoint = lambda base_url: None
        service_module._launch_local_benchmark = (
            lambda *args, **kwargs: service_module.CommandResult(
                ok=True, message="bench", data={"run_id": "run-1"}
            )
        )
        service_module.shutil.which = lambda executable: executable
        service_module.time.time = lambda: 100.0
        service_module.subprocess.Popen = lambda command, *args, **kwargs: launched.append(
            command
        ) or _Process(9999)
        result = service_module._launch_local_vllm(spec, state=state, state_path=state_path)
    finally:
        service_module._endpoint_is_healthy = original_health
        service_module._pid_is_alive = original_pid_is_alive
        service_module._wait_for_endpoint = original_wait
        service_module._launch_local_benchmark = original_launch_local
        service_module.shutil.which = original_which
        service_module.subprocess.Popen = original_popen
        service_module.time.time = original_time

    assert result.ok is True
    assert len(launched) == 1
    assert state.servers[0].id != "server-stale"
    assert state.servers[0].status == "healthy"
    assert state.servers[0].metadata["pid"] == 9999
