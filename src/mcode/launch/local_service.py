from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcode.launch.models import (
    CommandResult,
    LaunchSpec,
    ReuseMode,
    RunHandle,
    ServerHandle,
    TargetKind,
)
from mcode.launch.providers.local_ollama import (
    build_ollama_serve_command,
    build_ollama_warmup_command,
)
from mcode.launch.providers.local_vllm import build_local_vllm_command, build_local_vllm_reuse_key
from mcode.launch.state import LauncherState


def launch_local_vllm(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    resolve_local_server: Callable[..., ServerHandle],
    launch_local_benchmark: Callable[..., CommandResult],
) -> CommandResult:
    reuse_key = build_local_vllm_reuse_key(spec)
    endpoint = f"http://127.0.0.1:{spec.serving.port}/v1"
    server = resolve_local_server(
        spec,
        state=state,
        state_path=state_path,
        reuse_key=reuse_key,
        endpoint=endpoint,
        target=TargetKind.LOCAL_VLLM,
        executable="vllm",
        command=build_local_vllm_command(spec),
    )
    run = launch_local_benchmark(
        spec,
        env={
            "OPENAI_BASE_URL": server.endpoint,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
        },
        state=state,
        state_path=state_path,
    )
    run.data["server_id"] = server.id
    return run


def launch_local_ollama(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    resolve_local_server: Callable[..., ServerHandle],
    launch_local_benchmark: Callable[..., CommandResult],
) -> CommandResult:
    endpoint = f"http://127.0.0.1:{spec.serving.port}"
    reuse_key = "|".join(
        [
            "local-ollama",
            spec.model,
            str(spec.serving.port),
            str(spec.serving.ollama_num_parallel or 1),
        ]
    )
    server = resolve_local_server(
        spec,
        state=state,
        state_path=state_path,
        reuse_key=reuse_key,
        endpoint=endpoint,
        target=TargetKind.LOCAL_OLLAMA,
        executable="ollama",
        command=build_ollama_serve_command(spec),
        warmup_command=build_ollama_warmup_command(spec),
    )
    run = launch_local_benchmark(
        spec,
        env={"OLLAMA_HOST": endpoint},
        state=state,
        state_path=state_path,
    )
    run.data["server_id"] = server.id
    return run


def launch_openai_compatible(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    launch_local_benchmark: Callable[..., CommandResult],
) -> CommandResult:
    env = {
        "OPENAI_BASE_URL": spec.target.base_url,
        "OPENAI_API_KEY": os.environ.get(
            spec.target.api_key_env, os.environ.get("OPENAI_API_KEY", "dummy")
        ),
    }
    return launch_local_benchmark(spec, env=env, state=state, state_path=state_path)


def resolve_local_server(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    reuse_key: str,
    endpoint: str,
    target: TargetKind,
    executable: str,
    command: str,
    endpoint_is_healthy: Callable[[str], bool],
    wait_for_endpoint: Callable[[str], None],
    save_state: Callable[[Path | None, LauncherState], None],
    which: Callable[[str], str | None],
    warmup_command: str | None = None,
) -> ServerHandle:
    existing = next(
        (
            server
            for server in state.servers
            if server.reuse_key == reuse_key and server.status == "healthy"
        ),
        None,
    )
    if existing and spec.reuse == ReuseMode.STOP_AND_REPLACE and existing.metadata.get("pid"):
        os.kill(int(existing.metadata["pid"]), 15)
    if existing and spec.reuse == ReuseMode.PREFER and endpoint_is_healthy(existing.endpoint):
        return existing
    if spec.reuse == ReuseMode.PREFER and endpoint_is_healthy(endpoint):
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=target.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status="healthy",
            metadata={"discovered": True},
            log_path=None,
        )
    else:
        log_path = str(Path.cwd() / f".mcode-{target.value}-{spec.serving.port}.log")
        metadata: dict[str, Any]
        if spec.yes and which(executable):
            with open(log_path, "a", encoding="utf-8") as handle:
                process = subprocess.Popen(command, shell=True, stdout=handle, stderr=handle)
            if warmup_command:
                subprocess.run(warmup_command, shell=True, check=False)
            wait_for_endpoint(endpoint)
            metadata = {"pid": process.pid}
            if warmup_command:
                metadata["warmup"] = warmup_command
            status = "healthy"
        else:
            metadata = {"command": command}
            if warmup_command:
                metadata["warmup"] = warmup_command
            status = "planned"
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=target.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status=status,
            metadata=metadata,
            log_path=log_path,
        )
    state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [server]
    save_state(state_path, state)
    return server


def launch_local_benchmark(
    spec: LaunchSpec,
    *,
    env: dict[str, str],
    state: LauncherState,
    state_path: Path | None,
    save_state: Callable[[Path | None, LauncherState], None],
) -> CommandResult:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    run_dir = Path("results") / "launch" / run_id
    log_paths = [
        str(run_dir / f"benchmark-shard-{shard_index}.log")
        for shard_index in range(spec.benchmark.parallelism)
    ]
    db_paths = [
        str(run_dir / f"diagnostic-shard-{shard_index}.db")
        for shard_index in range(spec.benchmark.parallelism)
    ]
    commands = [
        build_local_benchmark_command(
            spec,
            run_dir=run_dir,
            shard_count=spec.benchmark.parallelism,
            shard_index=shard_index,
        )
        for shard_index in range(spec.benchmark.parallelism)
    ]
    metadata: dict[str, Any] = {
        "commands": commands,
        "db_paths": db_paths,
        "log_paths": log_paths,
        "run_dir": str(run_dir),
        **env,
    }
    if spec.yes:
        merged_env = os.environ.copy()
        merged_env.update(env)
        run_dir.mkdir(parents=True, exist_ok=True)
        pids: list[int] = []
        for command, log_path in zip(commands, log_paths, strict=True):
            with open(log_path, "a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command, shell=True, stdout=handle, stderr=handle, env=merged_env
                )
            pids.append(process.pid)
        metadata["pids"] = pids
        metadata["pid"] = pids[0]
        status = "running"
    else:
        status = "planned"
    run = RunHandle(
        id=run_id,
        target=spec.target.kind.value,
        benchmark=spec.benchmark.benchmark,
        status=status,
        metadata=metadata,
        log_path=log_paths[0],
    )
    state.runs.append(run)
    save_state(state_path, state)
    return CommandResult(ok=True, message="\n".join(commands), data={"run_id": run.id})


def build_local_benchmark_command(
    spec: LaunchSpec,
    *,
    run_dir: Path,
    shard_count: int,
    shard_index: int,
) -> str:
    task_ids = f"--task-ids {spec.benchmark.task_ids}" if spec.benchmark.task_ids else ""
    limit = f"--limit {spec.benchmark.limit}" if spec.benchmark.limit is not None else ""
    db_path = run_dir / f"diagnostic-shard-{shard_index}.db"
    return (
        f"uv run mcode bench {spec.benchmark.benchmark} "
        f"--model {spec.model} "
        f"--backend {spec.benchmark.backend} "
        f"--loop-budget {spec.benchmark.loop_budget} "
        f"--timeout {spec.benchmark.timeout} "
        f"--split {spec.benchmark.split} "
        f"--mem-limit {spec.benchmark.mem_limit} "
        f"--pids-limit {spec.benchmark.pids_limit} "
        f"--shard-count {shard_count} "
        f"--shard-index {shard_index} "
        f"--n-samples {spec.benchmark.n_samples} "
        f"--db {db_path} "
        f"{task_ids} {limit}"
    ).strip()
