from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess

from mcode.launch import service as service_module
from mcode.launch.models import (
    BenchSpec,
    BlueVelaTargetSpec,
    CommandResult,
    LaunchSpec,
    ReuseMode,
    ServerHandle,
    ServingSpec,
    SyncMode,
    SyncSpec,
    TargetKind,
)
from mcode.launch.profiles import resolve_serving_profile
from mcode.launch.providers.bluevela import (
    build_bluevela_benchmark_command,
    build_bluevela_lock_path,
    build_bluevela_server_registry_path,
    build_bluevela_server_reuse_key,
    build_bluevela_vllm_command,
    build_remote_workspace_prepare_command,
)
from mcode.launch.state import LauncherState, update_state


def _spec() -> LaunchSpec:
    return LaunchSpec(
        target=BlueVelaTargetSpec(
            kind=TargetKind.BLUEVELA,
            login="user@login3.example.com",
            workspace_root="/u/user/mcode-launch",
            queue="normal",
            group="grp_runtime",
            shared_root="/proj/shared/user",
            hf_env="/u/user/.config/mcode/hf-env.sh",
            podman_graphroot="/proj/shared/user/podman/graphroot",
            podman_runroot="/proj/shared/user/podman/runroot",
        ),
        model="google/gemma-4-31B-it",
        benchmark=BenchSpec(
            benchmark="swebench-lite",
            backend="openai",
            split="test",
            loop_budget=15,
            timeout=300,
            parallelism=4,
            task_ids="tasks.txt",
        ),
        serving=ServingSpec(
            engine="vllm",
            port=8331,
            tensor_parallel=2,
            data_parallel=1,
            api_server_count=1,
            max_model_len=32768,
            profile=resolve_serving_profile("google/gemma-4-31B-it"),
        ),
        sync=SyncSpec(mode=SyncMode.GIT_OVERLAY, ref="HEAD"),
        reuse=ReuseMode.PREFER,
        json_mode=False,
        yes=True,
        follow=False,
    )


def test_bluevela_vllm_command_uses_resolved_profile() -> None:
    command = build_bluevela_vllm_command(_spec(), run_dir=Path("/u/user/mcode-launch/runs/run-1"))

    assert command.count("bash -lc") == 1
    assert "--tool-call-parser gemma4" in command
    assert "--reasoning-parser gemma4" in command
    assert "docker.io/vllm/vllm-openai:gemma4" in command
    assert "--tensor-parallel-size 2" in command
    assert "--gpu-memory-utilization 0.9" in command
    assert "--storage-opt ignore_chown_errors=true" in command
    assert "mode=exclusive_process" in command
    assert "export HF_HUB_OFFLINE=1;" in command
    assert "export TRANSFORMERS_OFFLINE=1;" in command
    assert "IMAGE_ARCHIVE=/proj/shared/user/podman-image-cache/" in command
    assert "podman --cgroup-manager=cgroupfs --storage-driver=overlay " in command
    assert 'image exists "$IMAGE" >/dev/null 2>&1' in command
    assert 'load -i "$IMAGE_ARCHIVE" >/dev/null' in command
    assert 'save -o "$IMAGE_ARCHIVE" "$IMAGE" >/dev/null' in command
    assert "-e HF_HUB_OFFLINE -e TRANSFORMERS_OFFLINE" in command
    assert '-e HF_HOME="/root/.cache/huggingface"' in command
    assert '-e HF_HUB_CACHE="/root/.cache/huggingface/hub"' in command
    assert "/proj/shared/user/hf-cache/hub/models--google--gemma-4-31B-it" in command
    assert 'GRAPHROOT="${GRAPHROOT_BASE}/${HOST_TAG}/graphroot"' in command
    assert 'RUNROOT="${RUNROOT_BASE}/${HOST_TAG}/runroot"' in command
    assert "GRAPHROOT_BASE=/proj/shared/user/podman" in command
    assert "RUNROOT_BASE=/proj/shared/user/podman" in command
    assert '--root="$GRAPHROOT"' in command
    assert '--runroot="$RUNROOT"' in command
    assert "--cgroup-manager=cgroupfs" in command


def test_bluevela_benchmark_command_uses_parallelism_and_openai_backend() -> None:
    command = build_bluevela_benchmark_command(
        _spec(),
        workspace_path=Path("/u/user/mcode-launch/workspaces/ws-1"),
        db_path=Path("/u/user/mcode-launch/runs/run-1/diagnostic.db"),
        shard_index=2,
        endpoint="http://host:8331/v1",
    )

    assert "uv run mcode bench swebench-lite" in command
    assert "bash -lc" not in command
    assert "--backend openai" in command
    assert "--shard-count 4" in command
    assert "--shard-index 2" in command
    assert "export OPENAI_BASE_URL=http://host:8331/v1" in command
    assert "podman --cgroup-manager=cgroupfs --storage-driver=overlay" in command
    assert "PODMAN_ROOT_BASE=/proj/shared/user/podman-bench" in command
    assert 'JOB_KEY="${LSB_JOBID:-0}"' in command
    assert 'GRAPHROOT="${PODMAN_ROOT_BASE}/${JOB_KEY}/graphroot"' in command
    assert 'RUNROOT="${PODMAN_ROOT_BASE}/${JOB_KEY}/runroot"' in command
    assert '--root="$GRAPHROOT"' in command
    assert '--runroot="$RUNROOT"' in command
    assert "GRAPHROOT_BASE=/proj/shared/user/podman" not in command
    assert "RUNROOT_BASE=/proj/shared/user/podman" not in command
    assert "HOST_TAG=$(hostname -s)" not in command
    assert 'rm -rf "$RUNROOT/networks/rootless-netns"' in command
    assert (
        "podman --cgroup-manager=cgroupfs --storage-driver=overlay "
        '--root="$GRAPHROOT" --runroot="$RUNROOT" rm -af >/dev/null 2>&1 || true;' in command
    )
    assert 'export DOCKER_HOST="unix://${SOCK}"' in command
    assert "client.ping()" in command


def test_bluevela_server_reuse_key_includes_profile() -> None:
    key = build_bluevela_server_reuse_key(_spec())

    assert "bluevela" in key
    assert "google/gemma-4-31B-it" in key
    assert "gemma4" in key
    assert "mem=0.9" in key


def test_bluevela_registry_and_lock_paths_are_deterministic() -> None:
    target = _spec().target
    assert isinstance(target, BlueVelaTargetSpec)

    registry_path = build_bluevela_server_registry_path(target, reuse_key="reuse-key")
    lock_path = build_bluevela_lock_path(target, kind="server", key="reuse-key")

    assert registry_path.startswith("/u/user/mcode-launch/state/servers/")
    assert registry_path.endswith(".json")
    assert lock_path.startswith("/u/user/mcode-launch/locks/server-")
    assert lock_path.endswith(".lock")


def test_remote_workspace_prepare_command_does_not_double_wrap_bash() -> None:
    target = _spec().target
    assert isinstance(target, BlueVelaTargetSpec)

    class _Plan:
        remote_path = "/u/user/mcode-launch/workspaces/ws-1"

    command = build_remote_workspace_prepare_command(target, _Plan())

    assert "bash -lc" not in command
    assert "mkdir -p" in command


def test_acquire_remote_lock_does_not_double_wrap_bash() -> None:
    commands: list[str] = []
    original_run_ssh_result = service_module._run_ssh_result
    try:
        service_module._run_ssh_result = lambda login, command, check=False: (
            commands.append(command)
            or CompletedProcess(args=[login, command], returncode=0, stdout="", stderr="")
        )
        service_module._acquire_remote_lock(
            "user@login3.example.com",
            "/u/user/mcode-launch/locks/server-reuse-key.lock",
        )
    finally:
        service_module._run_ssh_result = original_run_ssh_result

    assert len(commands) == 1
    assert "bash -lc" not in commands[0]


def test_run_ssh_result_uses_ssh_n_and_bash_lc() -> None:
    original_run = service_module.subprocess.run
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        captured.append(cmd)
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    try:
        service_module.subprocess.run = fake_run
        result = service_module._run_ssh_result("user@login3.example.com", "echo ok")
    finally:
        service_module.subprocess.run = original_run

    assert result.returncode == 0
    assert captured == [["ssh", "-n", "user@login3.example.com", "bash -lc 'echo ok'"]]


@dataclass
class _Plan:
    signature: str
    remote_path: str
    ref_sha: str
    diff_summary: str
    repo_url: str
    mode: SyncMode
    bootstrap_key: str
    is_noop: bool = False


def test_sync_bluevela_workspace_uses_remote_json_helper_for_manifest(tmp_path: Path) -> None:
    target = _spec().target
    assert isinstance(target, BlueVelaTargetSpec)
    plan = _Plan(
        signature="ws-1",
        remote_path="/u/user/mcode-launch/workspaces/ws-1",
        ref_sha="deadbeef",
        diff_summary="summary",
        repo_url="https://example.com/repo.git",
        mode=SyncMode.WORKING_TREE,
        bootstrap_key="uv-sync:swebench,datasets",
    )
    captured_payloads: list[tuple[str, str, dict[str, object]]] = []
    original_acquire = service_module._acquire_remote_lock
    original_release = service_module._release_remote_lock
    original_read_manifest = service_module._read_remote_workspace_manifest
    original_run_ssh = service_module._run_ssh
    original_write_remote_json = service_module._write_remote_json
    original_subprocess_run = service_module.subprocess.run
    try:
        service_module._acquire_remote_lock = lambda *args, **kwargs: None
        service_module._release_remote_lock = lambda *args, **kwargs: None
        service_module._read_remote_workspace_manifest = lambda *args, **kwargs: None
        service_module._run_ssh = lambda *args, **kwargs: ""
        service_module._write_remote_json = lambda login, path, payload: captured_payloads.append(
            (login, path, payload)
        )
        service_module.subprocess.run = lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout=b"tracked.py\0" if kwargs.get("capture_output") else "",
            stderr="",
        )
        service_module._sync_bluevela_workspace(target, tmp_path, plan)
    finally:
        service_module._acquire_remote_lock = original_acquire
        service_module._release_remote_lock = original_release
        service_module._read_remote_workspace_manifest = original_read_manifest
        service_module._run_ssh = original_run_ssh
        service_module._write_remote_json = original_write_remote_json
        service_module.subprocess.run = original_subprocess_run

    assert captured_payloads == [
        (
            "user@login3.example.com",
            "/u/user/mcode-launch/workspaces/ws-1/.mcode-launch-workspace.json",
            {
                "signature": "ws-1",
                "remote_path": "/u/user/mcode-launch/workspaces/ws-1",
                "ref_sha": "deadbeef",
                "repo_url": "https://example.com/repo.git",
            },
        )
    ]


def test_describe_bluevela_health_wait_reports_image_pull() -> None:
    description = service_module._describe_bluevela_health_wait(
        "p6-r05-n3.bluevela.rmf.ibm.com",
        "Trying to pull docker.io/vllm/vllm-openai:v0.17.0...\nCopying blob sha256:123",
    )

    assert description == "Pulling vLLM image on p6-r05-n3.bluevela.rmf.ibm.com"


def test_describe_bluevela_health_wait_reports_cuda_graph_capture() -> None:
    description = service_module._describe_bluevela_health_wait(
        "p2-r05-n2.bluevela.rmf.ibm.com",
        (
            "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  47%|████▋"
            "     | 24/51 [00:02<00:02, 11.08it/s]"
        ),
    )

    assert description == "Capturing CUDA graphs on p2-r05-n2.bluevela.rmf.ibm.com"


def test_bluevela_health_wait_progress_advances_during_weight_loading() -> None:
    progress = service_module._bluevela_health_wait_progress(
        "Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:45<00:45, 45.16s/it]"
    )

    assert progress == 40


def test_bluevela_health_wait_progress_advances_during_cuda_graph_capture() -> None:
    progress = service_module._bluevela_health_wait_progress(
        "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  47%|████▋"
        "     | 24/51 [00:02<00:02, 11.08it/s]"
    )

    assert progress == 80


def test_describe_bluevela_health_wait_prefers_later_stage_over_early_markers() -> None:
    log_tail = (
        "Non-default args: ...\n"
        "Starting vLLM API server\n"
        "Loading safetensors checkpoint shards:  50% Completed | 1/2\n"
    )
    description = service_module._describe_bluevela_health_wait("host.example.com", log_tail)
    assert description == "Loading model weights on host.example.com"


def test_bluevela_health_wait_progress_prefers_later_stage_over_early_markers() -> None:
    log_tail = "Non-default args: ...\nStarting vLLM API server\nGPU KV cache size: 4 GiB\n"
    progress = service_module._bluevela_health_wait_progress(log_tail)
    assert progress == 72


def test_bluevela_launch_preview_computes_sync_without_applying(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    calls: dict[str, bool] = {}
    original_launch_sync = service_module._launch_sync
    try:
        service_module._launch_sync = lambda spec, **kwargs: (
            calls.update({"apply": spec.sync.apply, "check": spec.sync.check})
            or CommandResult(
                ok=True,
                message="sync",
                data={"signature": "ws-1", "remote_path": "/u/user/mcode-launch/workspaces/ws-1"},
            )
        )
        result = service_module._launch_bluevela(
            replace(_spec(), yes=False),
            repo_root=tmp_path,
            state=state,
            state_path=state_path,
        )
    finally:
        service_module._launch_sync = original_launch_sync

    assert calls == {"apply": False, "check": True}
    assert result.ok is True
    assert "bsub" in result.message
    assert state.runs[0].log_path.endswith("/benchmark-shard-0.log")


def test_stop_run_skips_bkill_for_planned_bluevela_run(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-lite",
        status="planned",
        metadata={
            "login": "user@login3.example.com",
            "job_ids": [],
            "commands": ["bsub some command"],
        },
        log_path="/u/user/mcode-launch/runs/run-1/benchmark-shard-0.log",
    )
    state = LauncherState(runs=[run])
    state_path = tmp_path / "launch-state.json"
    original_run_ssh = service_module._run_ssh
    original_run_ssh_result = service_module._run_ssh_result
    calls: list[tuple[str, str]] = []
    try:
        service_module._run_ssh = lambda login, command: calls.append((login, command)) or ""
        service_module._run_ssh_result = lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )
        result = service_module._stop_run(run, state_path, state=state)
    finally:
        service_module._run_ssh = original_run_ssh
        service_module._run_ssh_result = original_run_ssh_result

    assert result.ok is True
    assert calls == []


def test_stop_run_normalizes_bluevela_submission_messages(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-lite",
        status="running",
        metadata={
            "login": "user@login3.example.com",
            "job_ids": [
                "Job <845767> is submitted to queue <normal>.",
                "Job <845768> is submitted to queue <normal>.",
            ],
        },
        log_path="/u/user/mcode-launch/runs/run-1/benchmark-shard-0.log",
    )
    state = LauncherState(runs=[run])
    state_path = tmp_path / "launch-state.json"
    original_run_ssh = service_module._run_ssh
    original_run_ssh_result = service_module._run_ssh_result
    calls: list[tuple[str, str]] = []
    try:
        service_module._run_ssh = lambda login, command: calls.append((login, command)) or ""
        service_module._run_ssh_result = lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )
        result = service_module._stop_run(run, state_path, state=state)
    finally:
        service_module._run_ssh = original_run_ssh
        service_module._run_ssh_result = original_run_ssh_result

    assert result.ok is True
    assert calls == []


def test_stop_server_normalizes_bluevela_submission_message(tmp_path: Path) -> None:
    server = ServerHandle(
        id="server-1",
        target=TargetKind.BLUEVELA.value,
        reuse_key="reuse-key",
        endpoint="http://host:8331/v1",
        status="healthy",
        metadata={
            "job_id": "Job <845766> is submitted to queue <normal>.",
            "login": "user@login3.example.com",
            "registry_path": "/u/user/mcode-launch/state/servers/reuse-key.json",
        },
        log_path="/u/user/mcode-launch/runs/server-1/vllm.log",
    )
    state = LauncherState(servers=[server])
    state_path = tmp_path / "launch-state.json"
    original_run_ssh = service_module._run_ssh
    original_run_ssh_result = service_module._run_ssh_result
    calls: list[tuple[str, str]] = []
    try:
        service_module._run_ssh = lambda login, command: calls.append((login, command)) or ""
        service_module._run_ssh_result = lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )
        result = service_module._stop_server(server, state_path, state=state)
    finally:
        service_module._run_ssh = original_run_ssh
        service_module._run_ssh_result = original_run_ssh_result

    assert result.ok is True
    assert calls == [
        ("user@login3.example.com", "rm -f /u/user/mcode-launch/state/servers/reuse-key.json"),
    ]


def test_launch_stop_all_bluevela_stops_tracked_entries_and_sweeps_cluster(tmp_path: Path) -> None:
    state_path = tmp_path / "launch-state.json"
    bluevela_run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-lite",
        status="running",
        metadata={"login": "user@login3.example.com", "job_ids": ["123"]},
        log_path="/u/user/run-1.log",
    )
    local_run = service_module.RunHandle(
        id="run-2",
        target=TargetKind.LOCAL_OLLAMA.value,
        benchmark="swebench-lite",
        status="running",
        metadata={"pid": 1234},
        log_path=str(tmp_path / "local.log"),
    )
    bluevela_server = ServerHandle(
        id="server-1",
        target=TargetKind.BLUEVELA.value,
        reuse_key="reuse-1",
        endpoint="http://host:8331/v1",
        status="healthy",
        metadata={"login": "user@login3.example.com", "job_id": "456"},
        log_path="/u/user/server.log",
    )
    update_state(
        state_path,
        lambda current: (
            setattr(current, "runs", [bluevela_run, local_run]),
            setattr(current, "servers", [bluevela_server]),
        ),
    )
    original_stop_run = service_module._stop_run
    original_stop_server = service_module._stop_server
    original_run_ssh = service_module._run_ssh
    stopped_runs: list[str] = []
    stopped_servers: list[str] = []
    commands: list[tuple[str, str]] = []
    try:
        service_module._stop_run = lambda run, path, state=None: (
            stopped_runs.append(run.id) or CommandResult(ok=True, message=f"Stopped {run.id}")
        )
        service_module._stop_server = lambda server, path, state=None: (
            stopped_servers.append(server.id)
            or CommandResult(ok=True, message=f"Stopped {server.id}")
        )
        service_module._run_ssh = lambda login, command: commands.append((login, command)) or (
            "jobs_killed=2\nregistries_removed=3\nlocks_removed=4"
        )
        result = service_module.launch_stop_all(
            target=TargetKind.BLUEVELA.value,
            state_path=state_path,
            bluevela_login="user@login3.example.com",
            bluevela_workspace_root="/u/user/mcode-launch",
        )
    finally:
        service_module._stop_run = original_stop_run
        service_module._stop_server = original_stop_server
        service_module._run_ssh = original_run_ssh

    assert result.ok is True
    assert stopped_runs == ["run-1"]
    assert stopped_servers == ["server-1"]
    assert result.data == {
        "jobs_killed": 2,
        "locks_removed": 4,
        "registries_removed": 3,
        "runs_stopped": 1,
        "servers_stopped": 1,
    }
    assert len(commands) == 1
    assert commands[0][0] == "user@login3.example.com"
    assert "awk '/mcode-vllm|mcode-bench/ {print $1}'" in commands[0][1]
    assert 'bkill "$job"' in commands[0][1]
    assert "/u/user/mcode-launch/state/servers/*.json" in commands[0][1]
    assert "/u/user/mcode-launch/locks/*" in commands[0][1]


def test_launch_stop_all_requires_bluevela_config_for_sweep(tmp_path: Path) -> None:
    result = service_module.launch_stop_all(
        target=TargetKind.BLUEVELA.value,
        state_path=tmp_path / "launch-state.json",
    )

    assert result.ok is False
    assert "requires Blue Vela login and workspace root" in result.message


def test_bkill_ignores_already_finished_job() -> None:
    original_run_ssh_result = service_module._run_ssh_result

    try:
        service_module._run_ssh_result = lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=255,
            stdout="",
            stderr="Job <845766>: Job has already finished\n",
        )
        service_module._maybe_bkill_bluevela_job("user@login3.example.com", "845766")
    finally:
        service_module._run_ssh_result = original_run_ssh_result


def test_wait_for_bluevela_endpoint_fails_when_job_exits(tmp_path: Path) -> None:
    original_remote_health = service_module._remote_endpoint_is_healthy
    original_run_ssh = service_module._run_ssh
    original_job_active = service_module._bluevela_job_is_active
    original_sleep = service_module.time.sleep

    try:
        service_module._remote_endpoint_is_healthy = lambda *args, **kwargs: False
        service_module._run_ssh = (
            lambda *args, **kwargs: "KeyError: invalid tool call parser: gemma4"
        )
        service_module._bluevela_job_is_active = lambda *args, **kwargs: False
        service_module.time.sleep = lambda *_args, **_kwargs: None
        try:
            service_module._wait_for_bluevela_endpoint(
                "user@login3.example.com",
                "http://host:8331/v1",
                log_path="/u/user/run/vllm.log",
                job_id="860088",
                timeout_s=1,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        service_module._remote_endpoint_is_healthy = original_remote_health
        service_module._run_ssh = original_run_ssh
        service_module._bluevela_job_is_active = original_job_active
        service_module.time.sleep = original_sleep

    assert "failed before endpoint was healthy" in message
    assert "invalid tool call parser: gemma4" in message


def test_wait_for_bluevela_endpoint_fails_immediately_on_invalid_tool_parser() -> None:
    original_remote_health = service_module._remote_endpoint_is_healthy
    original_run_ssh = service_module._run_ssh
    original_job_active = service_module._bluevela_job_is_active
    original_sleep = service_module.time.sleep

    try:
        service_module._remote_endpoint_is_healthy = lambda *args, **kwargs: False
        service_module._run_ssh = (
            lambda *args, **kwargs: "KeyError: invalid tool call parser: gemma4"
        )
        service_module._bluevela_job_is_active = lambda *args, **kwargs: True
        service_module.time.sleep = lambda *_args, **_kwargs: None
        try:
            service_module._wait_for_bluevela_endpoint(
                "user@login3.example.com",
                "http://host:8331/v1",
                log_path="/u/user/run/vllm.log",
                job_id="860088",
                timeout_s=1,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        service_module._remote_endpoint_is_healthy = original_remote_health
        service_module._run_ssh = original_run_ssh
        service_module._bluevela_job_is_active = original_job_active
        service_module.time.sleep = original_sleep

    assert "failed before endpoint was healthy" in message
    assert "invalid tool call parser: gemma4" in message


def test_wait_for_bluevela_endpoint_allows_active_log_progress() -> None:
    original_remote_health = service_module._remote_endpoint_is_healthy
    original_run_ssh = service_module._run_ssh
    original_job_active = service_module._bluevela_job_is_active
    original_sleep = service_module.time.sleep
    original_time = service_module.time.time

    health_checks = iter([False, False, False, True])
    log_tails = iter(
        [
            "Copying blob sha256:1",
            "Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:45<00:45, 45.16s/it]",
            (
                "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  47%|████▋"
                "     | 24/51 [00:02<00:02, 11.08it/s]"
            ),
        ]
    )
    now = {"value": 0.0}

    try:
        service_module._remote_endpoint_is_healthy = lambda *args, **kwargs: next(health_checks)
        service_module._run_ssh = lambda *args, **kwargs: next(log_tails)
        service_module._bluevela_job_is_active = lambda *args, **kwargs: True
        service_module.time.time = lambda: now["value"]
        service_module.time.sleep = lambda seconds: now.__setitem__("value", now["value"] + seconds)
        service_module._wait_for_bluevela_endpoint(
            "user@login3.example.com",
            "http://host:8331/v1",
            log_path="/u/user/run/vllm.log",
            job_id="860088",
            timeout_s=1,
            max_timeout_s=10,
        )
    finally:
        service_module._remote_endpoint_is_healthy = original_remote_health
        service_module._run_ssh = original_run_ssh
        service_module._bluevela_job_is_active = original_job_active
        service_module.time.sleep = original_sleep
        service_module.time.time = original_time


def test_wait_for_bluevela_endpoint_does_not_fail_while_job_is_running() -> None:
    original_remote_health = service_module._remote_endpoint_is_healthy
    original_run_ssh = service_module._run_ssh
    original_job_active = service_module._bluevela_job_is_active
    original_sleep = service_module.time.sleep

    health_checks = iter([False, False, True])
    log_tails = iter(
        [
            "Copying blob sha256:1",
            "Copying blob sha256:1",
        ]
    )

    try:
        service_module._remote_endpoint_is_healthy = lambda *args, **kwargs: next(health_checks)
        service_module._run_ssh = lambda *args, **kwargs: next(log_tails)
        service_module._bluevela_job_is_active = lambda *args, **kwargs: True
        service_module.time.sleep = lambda *_args, **_kwargs: None
        service_module._wait_for_bluevela_endpoint(
            "user@login3.example.com",
            "http://host:8331/v1",
            log_path="/u/user/run/vllm.log",
            job_id="860088",
            timeout_s=1,
            max_timeout_s=2,
        )
    finally:
        service_module._remote_endpoint_is_healthy = original_remote_health
        service_module._run_ssh = original_run_ssh
        service_module._bluevela_job_is_active = original_job_active
        service_module.time.sleep = original_sleep


def test_bluevela_server_resolution_reuses_remote_registry(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    remote = asdict(
        ServerHandle(
            id="server-remote",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://host:8331/v1",
            status="healthy",
            metadata={"job_id": "123", "login": "user@login3.example.com"},
            log_path="/u/user/mcode-launch/runs/r1/vllm.log",
        )
    )
    original_acquire = service_module._acquire_remote_lock
    original_release = service_module._release_remote_lock
    original_read = service_module._read_remote_json
    original_health = service_module._remote_endpoint_is_healthy
    try:
        service_module._acquire_remote_lock = lambda *args, **kwargs: None
        service_module._release_remote_lock = lambda *args, **kwargs: None
        service_module._read_remote_json = lambda *args, **kwargs: remote
        service_module._remote_endpoint_is_healthy = (
            lambda login, base_url: login == "user@login3.example.com"
            and base_url == "http://host:8331/v1"
        )
        server = service_module._resolve_bluevela_server(
            _spec(),
            state=state,
            state_path=state_path,
            reuse_key="reuse-key",
            workspace_signature="ws-1",
            existing_server=None,
        )
    finally:
        service_module._acquire_remote_lock = original_acquire
        service_module._release_remote_lock = original_release
        service_module._read_remote_json = original_read
        service_module._remote_endpoint_is_healthy = original_health

    assert server.id == "server-remote"
    assert state.servers[0].endpoint == "http://host:8331/v1"


def test_bluevela_server_resolution_waits_on_pending_remote_registry(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    remote = asdict(
        ServerHandle(
            id="server-pending",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://pending:8331/v1",
            status="pending",
            metadata={
                "job_id": "123",
                "login": "user@login3.example.com",
                "registry_path": "/u/user/mcode-launch/state/servers/reuse-key.json",
                "run_dir": "/u/user/mcode-launch/runs/r1",
                "workspace_signature": "ws-1",
            },
            log_path="/u/user/mcode-launch/runs/r1/vllm.log",
        )
    )
    writes: list[dict[str, object]] = []
    original_acquire = service_module._acquire_remote_lock
    original_release = service_module._release_remote_lock
    original_read = service_module._read_remote_json
    original_job_active = service_module._bluevela_job_is_active
    original_wait_file = service_module._wait_for_remote_file
    original_wait_endpoint = service_module._wait_for_bluevela_endpoint
    original_write = service_module._write_remote_json
    try:
        service_module._acquire_remote_lock = lambda *args, **kwargs: None
        service_module._release_remote_lock = lambda *args, **kwargs: None
        service_module._read_remote_json = lambda *args, **kwargs: remote
        service_module._bluevela_job_is_active = lambda *args, **kwargs: True
        service_module._wait_for_remote_file = lambda *args, **kwargs: "host"
        service_module._wait_for_bluevela_endpoint = lambda *args, **kwargs: None
        service_module._write_remote_json = lambda login, path, payload: writes.append(payload)
        server = service_module._resolve_bluevela_server(
            _spec(),
            state=state,
            state_path=state_path,
            reuse_key="reuse-key",
            workspace_signature="ws-1",
            existing_server=None,
        )
    finally:
        service_module._acquire_remote_lock = original_acquire
        service_module._release_remote_lock = original_release
        service_module._read_remote_json = original_read
        service_module._bluevela_job_is_active = original_job_active
        service_module._wait_for_remote_file = original_wait_file
        service_module._wait_for_bluevela_endpoint = original_wait_endpoint
        service_module._write_remote_json = original_write

    assert server.status == "healthy"
    assert server.endpoint == "http://host:8331/v1"
    assert writes[-1]["status"] == "healthy"


def test_bluevela_server_resolution_replaces_stale_pending_registry(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    remote = asdict(
        ServerHandle(
            id="server-pending",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://pending:8331/v1",
            status="pending",
            metadata={
                "job_id": "123",
                "login": "user@login3.example.com",
                "registry_path": "/u/user/mcode-launch/state/servers/reuse-key.json",
                "run_dir": "/u/user/mcode-launch/runs/r1",
                "workspace_signature": "ws-1",
            },
            log_path="/u/user/mcode-launch/runs/r1/vllm.log",
        )
    )
    writes: list[dict[str, object]] = []
    original_acquire = service_module._acquire_remote_lock
    original_release = service_module._release_remote_lock
    original_read = service_module._read_remote_json
    original_job_active = service_module._bluevela_job_is_active
    original_wait_file = service_module._wait_for_remote_file
    original_wait_endpoint = service_module._wait_for_bluevela_endpoint
    original_write = service_module._write_remote_json
    original_run_ssh = service_module._run_ssh
    original_build = service_module.build_bluevela_vllm_command
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._acquire_remote_lock = lambda *args, **kwargs: None
        service_module._release_remote_lock = lambda *args, **kwargs: None
        reads = {"count": 0}
        service_module._read_remote_json = (
            lambda *args, **kwargs: remote
            if (reads.__setitem__("count", reads["count"] + 1) or reads["count"]) == 1
            else writes[-1]
        )
        service_module._bluevela_job_is_active = lambda *args, **kwargs: False
        service_module._wait_for_remote_file = lambda *args, **kwargs: "host"
        service_module._wait_for_bluevela_endpoint = lambda *args, **kwargs: None
        service_module._write_remote_json = lambda login, path, payload: writes.append(payload)
        service_module._run_ssh = lambda login, command: (
            ""
            if command.startswith("mkdir -p ") or command.startswith("rm -f ")
            else "Job <456> is submitted to queue <normal>."
        )
        service_module.build_bluevela_vllm_command = lambda *args, **kwargs: "launch-vllm"
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        server = service_module._resolve_bluevela_server(
            _spec(),
            state=state,
            state_path=state_path,
            reuse_key="reuse-key",
            workspace_signature="ws-1",
            existing_server=None,
        )
    finally:
        service_module._acquire_remote_lock = original_acquire
        service_module._release_remote_lock = original_release
        service_module._read_remote_json = original_read
        service_module._bluevela_job_is_active = original_job_active
        service_module._wait_for_remote_file = original_wait_file
        service_module._wait_for_bluevela_endpoint = original_wait_endpoint
        service_module._write_remote_json = original_write
        service_module._run_ssh = original_run_ssh
        service_module.build_bluevela_vllm_command = original_build
        service_module.uuid.uuid4 = original_uuid4

    assert server.metadata["job_id"] == "456"
    assert [payload["status"] for payload in writes] == ["pending", "healthy"]


def test_bluevela_server_resolution_retries_failed_pending_server(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    remote = asdict(
        ServerHandle(
            id="server-pending",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://pending:8331/v1",
            status="pending",
            metadata={
                "job_id": "123",
                "login": "user@login3.example.com",
                "registry_path": "/u/user/mcode-launch/state/servers/reuse-key.json",
                "run_dir": "/u/user/mcode-launch/runs/r1",
                "workspace_signature": "ws-1",
            },
            log_path="/u/user/mcode-launch/runs/r1/vllm.log",
        )
    )
    writes: list[dict[str, object]] = []
    commands: list[str] = []
    original_acquire = service_module._acquire_remote_lock
    original_release = service_module._release_remote_lock
    original_read = service_module._read_remote_json
    original_job_active = service_module._bluevela_job_is_active
    original_wait_file = service_module._wait_for_remote_file
    original_wait_endpoint = service_module._wait_for_bluevela_endpoint
    original_write = service_module._write_remote_json
    original_run_ssh = service_module._run_ssh
    original_build = service_module.build_bluevela_vllm_command
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._acquire_remote_lock = lambda *args, **kwargs: None
        service_module._release_remote_lock = lambda *args, **kwargs: None
        reads = {"count": 0}

        def _read(*args, **kwargs):
            reads["count"] += 1
            if reads["count"] <= 2:
                return remote
            return writes[-1] if writes else None

        service_module._read_remote_json = _read
        job_checks = iter([True, False])
        service_module._bluevela_job_is_active = lambda *args, **kwargs: next(job_checks)
        service_module._wait_for_remote_file = lambda *args, **kwargs: "host"
        wait_calls = {"count": 0}

        def _wait_endpoint(*args, **kwargs):
            wait_calls["count"] += 1
            if wait_calls["count"] == 1:
                raise RuntimeError("Blue Vela vLLM job failed before endpoint was healthy")

        service_module._wait_for_bluevela_endpoint = _wait_endpoint
        service_module._write_remote_json = lambda login, path, payload: writes.append(payload)
        service_module._run_ssh = lambda login, command: (
            commands.append(command)
            or (
                ""
                if command.startswith("mkdir -p ") or command.startswith("rm -f ")
                else "Job <456> is submitted to queue <normal>."
            )
        )
        service_module.build_bluevela_vllm_command = lambda *args, **kwargs: "launch-vllm"
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        server = service_module._resolve_bluevela_server(
            _spec(),
            state=state,
            state_path=state_path,
            reuse_key="reuse-key",
            workspace_signature="ws-1",
            existing_server=None,
        )
    finally:
        service_module._acquire_remote_lock = original_acquire
        service_module._release_remote_lock = original_release
        service_module._read_remote_json = original_read
        service_module._bluevela_job_is_active = original_job_active
        service_module._wait_for_remote_file = original_wait_file
        service_module._wait_for_bluevela_endpoint = original_wait_endpoint
        service_module._write_remote_json = original_write
        service_module._run_ssh = original_run_ssh
        service_module.build_bluevela_vllm_command = original_build
        service_module.uuid.uuid4 = original_uuid4

    assert server.status == "healthy"
    assert server.metadata["job_id"] == "456"
    assert any(
        command.startswith("rm -f /u/user/mcode-launch/state/servers/")
        and command.endswith(".json")
        for command in commands
    )
    assert "launch-vllm" in commands
    assert wait_calls["count"] == 2
    assert [payload["status"] for payload in writes] == ["pending", "healthy"]


def test_bluevela_remote_lock_creates_parent_directory() -> None:
    commands: list[str] = []
    original = service_module._run_ssh_result

    def fake_run_ssh_result(*args, **kwargs) -> CompletedProcess[str]:
        del kwargs
        commands.append(args[1])
        return CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    try:
        service_module._run_ssh_result = fake_run_ssh_result
        service_module._acquire_remote_lock(
            "user@login3.example.com",
            "/u/user/mcode-launch/locks/server-123.lock",
            timeout_s=1,
        )
    finally:
        service_module._run_ssh_result = original

    assert "mkdir -p /u/user/mcode-launch/locks;" in commands[0]


def test_bluevela_resolves_host_scoped_podman_storage() -> None:
    target = _spec().target
    assert isinstance(target, BlueVelaTargetSpec)
    original_run = service_module._run_ssh_result
    original_ssh = service_module._run_ssh

    def fake_run_ssh_result(*args, **kwargs) -> CompletedProcess[str]:
        del args, kwargs
        return CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    try:
        service_module._run_ssh = lambda *args, **kwargs: "login3"
        service_module._run_ssh_result = fake_run_ssh_result
        graphroot, runroot = service_module._resolve_bluevela_podman_storage(target)
    finally:
        service_module._run_ssh = original_ssh
        service_module._run_ssh_result = original_run

    assert graphroot == "/proj/shared/user/podman/login3/graphroot"
    assert runroot == "/proj/shared/user/podman/login3/runroot"


def test_bluevela_wait_for_endpoint_raises_on_engine_failure() -> None:
    original_health = service_module._remote_endpoint_is_healthy
    original_run_ssh = service_module._run_ssh

    try:
        service_module._remote_endpoint_is_healthy = lambda login, base_url: False
        service_module._run_ssh = (
            lambda *args, **kwargs: "RuntimeError: Engine core initialization failed."
        )
        try:
            service_module._wait_for_bluevela_endpoint(
                "user@login3.example.com",
                "http://host:8331/v1",
                log_path="/tmp/vllm.log",
                timeout_s=1,
            )
        except RuntimeError as exc:
            assert "Engine core initialization failed" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        service_module._remote_endpoint_is_healthy = original_health
        service_module._run_ssh = original_run_ssh


def test_bluevela_wait_for_endpoint_uses_remote_health_probe() -> None:
    calls: list[tuple[str, str]] = []
    original_health = service_module._remote_endpoint_is_healthy

    try:
        service_module._remote_endpoint_is_healthy = (
            lambda login, base_url: calls.append((login, base_url)) or True
        )
        service_module._wait_for_bluevela_endpoint(
            "user@login3.example.com",
            "http://host:8331/v1",
            log_path="/tmp/vllm.log",
            timeout_s=1,
        )
    finally:
        service_module._remote_endpoint_is_healthy = original_health

    assert calls == [("user@login3.example.com", "http://host:8331/v1")]


def test_bluevela_bsub_benchmark_command_is_shell_parseable() -> None:
    command = service_module._build_bluevela_bsub_benchmark_command(
        _spec(),
        workspace_path=Path("/u/user/mcode-launch/workspaces/ws-1"),
        run_dir=Path("/u/user/mcode-launch/runs/run-1"),
        shard_index=0,
        endpoint="http://host:8331/v1",
    )

    result = subprocess.run(
        ["bash", "-n", "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_bluevela_launch_records_real_shard_log_paths(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_launch_sync = service_module._launch_sync
    original_find_server = service_module._find_existing_server
    original_resolve_server = service_module._resolve_bluevela_server
    original_resolve_podman = service_module._resolve_bluevela_podman_storage
    original_run_ssh = service_module._run_ssh
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._launch_sync = lambda spec, **kwargs: CommandResult(
            ok=True,
            message="sync",
            data={"signature": "ws-1", "remote_path": "/u/user/mcode-launch/workspaces/ws-1"},
        )
        service_module._resolve_bluevela_podman_storage = lambda target: ("graphroot", "runroot")
        service_module._find_existing_server = lambda *args, **kwargs: None
        service_module._resolve_bluevela_server = lambda *args, **kwargs: ServerHandle(
            id="server-1",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://host:8331/v1",
            status="healthy",
            metadata={"login": "user@login3.example.com", "job_id": "1"},
            log_path="/u/user/mcode-launch/runs/server/vllm.log",
        )
        service_module._run_ssh = lambda *args, **kwargs: "123"
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        result = service_module._launch_bluevela(
            _spec(),
            repo_root=tmp_path,
            state=state,
            state_path=state_path,
        )
    finally:
        service_module._launch_sync = original_launch_sync
        service_module._find_existing_server = original_find_server
        service_module._resolve_bluevela_server = original_resolve_server
        service_module._resolve_bluevela_podman_storage = original_resolve_podman
        service_module._run_ssh = original_run_ssh
        service_module.uuid.uuid4 = original_uuid4

    assert result.ok is True
    run = state.runs[0]
    assert run.log_path == "/u/user/mcode-launch/runs/run-aaaaaaaa/benchmark-shard-0.log"
    assert run.metadata["job_ids"] == ["123", "123", "123", "123"]
    assert run.metadata["log_paths"] == [
        "/u/user/mcode-launch/runs/run-aaaaaaaa/benchmark-shard-0.log",
        "/u/user/mcode-launch/runs/run-aaaaaaaa/benchmark-shard-1.log",
        "/u/user/mcode-launch/runs/run-aaaaaaaa/benchmark-shard-2.log",
        "/u/user/mcode-launch/runs/run-aaaaaaaa/benchmark-shard-3.log",
    ]


def test_bluevela_launch_normalizes_job_ids_from_bsub_output(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_launch_sync = service_module._launch_sync
    original_find_server = service_module._find_existing_server
    original_resolve_server = service_module._resolve_bluevela_server
    original_resolve_podman = service_module._resolve_bluevela_podman_storage
    original_run_ssh = service_module._run_ssh
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._launch_sync = lambda spec, **kwargs: CommandResult(
            ok=True,
            message="sync",
            data={"signature": "ws-1", "remote_path": "/u/user/mcode-launch/workspaces/ws-1"},
        )
        service_module._resolve_bluevela_podman_storage = lambda target: ("graphroot", "runroot")
        service_module._find_existing_server = lambda *args, **kwargs: None
        service_module._resolve_bluevela_server = lambda *args, **kwargs: ServerHandle(
            id="server-1",
            target=TargetKind.BLUEVELA.value,
            reuse_key="reuse-key",
            endpoint="http://host:8331/v1",
            status="healthy",
            metadata={"login": "user@login3.example.com", "job_id": "1"},
            log_path="/u/user/mcode-launch/runs/server/vllm.log",
        )
        service_module._run_ssh = (
            lambda *args, **kwargs: "Job <845767> is submitted to queue <normal>."
        )
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        result = service_module._launch_bluevela(
            _spec(),
            repo_root=tmp_path,
            state=state,
            state_path=state_path,
        )
    finally:
        service_module._launch_sync = original_launch_sync
        service_module._find_existing_server = original_find_server
        service_module._resolve_bluevela_server = original_resolve_server
        service_module._resolve_bluevela_podman_storage = original_resolve_podman
        service_module._run_ssh = original_run_ssh
        service_module.uuid.uuid4 = original_uuid4

    assert result.ok is True
    assert state.runs[0].metadata["job_ids"] == ["845767", "845767", "845767", "845767"]


def test_bluevela_launch_persists_pending_run_before_server_wait(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_launch_sync = service_module._launch_sync
    original_find_server = service_module._find_existing_server
    original_resolve_server = service_module._resolve_bluevela_server
    original_resolve_podman = service_module._resolve_bluevela_podman_storage
    original_run_ssh = service_module._run_ssh
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._launch_sync = lambda spec, **kwargs: CommandResult(
            ok=True,
            message="sync",
            data={"signature": "ws-1", "remote_path": "/u/user/mcode-launch/workspaces/ws-1"},
        )
        service_module._resolve_bluevela_podman_storage = lambda target: ("graphroot", "runroot")
        service_module._find_existing_server = lambda *args, **kwargs: None

        def fake_resolve(*args, on_pending_server=None, **kwargs):
            del args, kwargs
            pending = ServerHandle(
                id="server-1",
                target=TargetKind.BLUEVELA.value,
                reuse_key="reuse-key",
                endpoint="http://pending:8331/v1",
                status="pending",
                metadata={"login": "user@login3.example.com", "job_id": "1"},
                log_path="/u/user/mcode-launch/runs/server/vllm.log",
            )
            assert on_pending_server is not None
            on_pending_server(pending)
            snapshot = service_module.load_state(state_path)
            assert len(snapshot.runs) == 1
            assert snapshot.runs[0].id == "run-aaaaaaaa"
            assert snapshot.runs[0].status == "pending"
            assert snapshot.runs[0].log_path == "/u/user/mcode-launch/runs/server/vllm.log"
            return ServerHandle(
                id="server-1",
                target=TargetKind.BLUEVELA.value,
                reuse_key="reuse-key",
                endpoint="http://host:8331/v1",
                status="healthy",
                metadata={"login": "user@login3.example.com", "job_id": "1"},
                log_path="/u/user/mcode-launch/runs/server/vllm.log",
            )

        service_module._resolve_bluevela_server = fake_resolve
        service_module._run_ssh = lambda *args, **kwargs: "123"
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        result = service_module._launch_bluevela(
            _spec(),
            repo_root=tmp_path,
            state=state,
            state_path=state_path,
        )
    finally:
        service_module._launch_sync = original_launch_sync
        service_module._find_existing_server = original_find_server
        service_module._resolve_bluevela_server = original_resolve_server
        service_module._resolve_bluevela_podman_storage = original_resolve_podman
        service_module._run_ssh = original_run_ssh
        service_module.uuid.uuid4 = original_uuid4

    assert result.ok is True
    assert result.data["run_id"] == "run-aaaaaaaa"
    assert state.runs[0].status == "running"


def test_bluevela_launch_marks_pending_run_failed_when_server_startup_fails(tmp_path: Path) -> None:
    state = LauncherState()
    state_path = tmp_path / "launch-state.json"
    original_launch_sync = service_module._launch_sync
    original_find_server = service_module._find_existing_server
    original_resolve_server = service_module._resolve_bluevela_server
    original_resolve_podman = service_module._resolve_bluevela_podman_storage
    original_uuid4 = service_module.uuid.uuid4
    uuids = iter(["aaaaaaaaaaaaaaaa"])

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    try:
        service_module._launch_sync = lambda spec, **kwargs: CommandResult(
            ok=True,
            message="sync",
            data={"signature": "ws-1", "remote_path": "/u/user/mcode-launch/workspaces/ws-1"},
        )
        service_module._resolve_bluevela_podman_storage = lambda target: ("graphroot", "runroot")
        service_module._find_existing_server = lambda *args, **kwargs: None

        def fake_resolve(*args, on_pending_server=None, **kwargs):
            del args, kwargs
            pending = ServerHandle(
                id="server-1",
                target=TargetKind.BLUEVELA.value,
                reuse_key="reuse-key",
                endpoint="http://pending:8331/v1",
                status="pending",
                metadata={"login": "user@login3.example.com", "job_id": "1"},
                log_path="/u/user/mcode-launch/runs/server/vllm.log",
            )
            assert on_pending_server is not None
            on_pending_server(pending)
            raise RuntimeError("server failed")

        service_module._resolve_bluevela_server = fake_resolve
        service_module.uuid.uuid4 = lambda: _UUID(next(uuids))
        result = service_module._launch_bluevela(
            _spec(),
            repo_root=tmp_path,
            state=state,
            state_path=state_path,
        )
    finally:
        service_module._launch_sync = original_launch_sync
        service_module._find_existing_server = original_find_server
        service_module._resolve_bluevela_server = original_resolve_server
        service_module._resolve_bluevela_podman_storage = original_resolve_podman
        service_module.uuid.uuid4 = original_uuid4

    assert result.ok is False
    assert result.data["run_id"] == "run-aaaaaaaa"
    assert state.runs[0].status == "failed"
    assert state.runs[0].metadata["error"] == "server failed"


def test_launch_attach_follows_all_bluevela_logs(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-live",
        status="running",
        metadata={
            "login": "user@login3.example.com",
            "log_paths": ["/u/user/run/benchmark-shard-0.log", "/u/user/run/benchmark-shard-1.log"],
        },
        log_path="/u/user/run/benchmark-shard-0.log",
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
    assert commands[0][0:3] == ["ssh", "-n", "user@login3.example.com"]
    assert commands[0][3].startswith("bash -lc ")
    assert (
        "tail -n 20 -f /u/user/run/benchmark-shard-0.log /u/user/run/benchmark-shard-1.log"
    ) in commands[0][3]


def test_launch_attach_follows_pending_bluevela_startup_log(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-live",
        status="pending",
        metadata={
            "login": "user@login3.example.com",
            "server_id": "server-1",
            "startup_server_status": "pending",
        },
        log_path="/u/user/run/vllm.log",
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
    assert commands[0][0:3] == ["ssh", "-n", "user@login3.example.com"]
    assert "tail -n 20 -f /u/user/run/vllm.log" in commands[0][3]


def test_stop_pending_bluevela_run_stops_pending_server(tmp_path: Path) -> None:
    run = service_module.RunHandle(
        id="run-1",
        target=TargetKind.BLUEVELA.value,
        benchmark="swebench-live",
        status="pending",
        metadata={
            "login": "user@login3.example.com",
            "server_id": "server-1",
            "startup_server_status": "pending",
        },
        log_path="/u/user/run/vllm.log",
    )
    server = ServerHandle(
        id="server-1",
        target=TargetKind.BLUEVELA.value,
        reuse_key="reuse-key",
        endpoint="http://pending:8331/v1",
        status="pending",
        metadata={"login": "user@login3.example.com", "job_id": "123"},
        log_path="/u/user/run/vllm.log",
    )
    state_path = tmp_path / "launch-state.json"
    update_state(
        state_path,
        lambda current: (
            setattr(current, "runs", [run]),
            setattr(current, "servers", [server]),
        ),
    )
    original_stop_server = service_module._stop_server
    called: list[str] = []

    try:
        service_module._stop_server = lambda current, path, state=None: (
            called.append(current.id) or CommandResult(ok=True, message=f"Stopped {current.id}")
        )
        result = service_module.launch_stop("run-1", state_path=state_path)
    finally:
        service_module._stop_server = original_stop_server

    assert result.ok is True
    assert called == ["server-1"]
