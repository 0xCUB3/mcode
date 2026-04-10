from __future__ import annotations

import os
import signal
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import replace
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
from mcode.launch.progress import NullProgressReporter, ProgressReporter
from mcode.launch.providers.local_ollama import (
    build_ollama_serve_command,
    build_ollama_warmup_command,
)
from mcode.launch.providers.local_vllm import build_local_vllm_command, build_local_vllm_reuse_key
from mcode.launch.state import LauncherState

LOCAL_PENDING_STARTUP_TIMEOUT_S = 30.0
LOCAL_PENDING_GRACE_S = 5.0


def launch_local_vllm(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    resolve_local_server: Callable[..., ServerHandle],
    launch_local_benchmark: Callable[..., CommandResult],
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    reporter.set(5, "Resolving local vLLM server")
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
    reporter.set(60, "Server ready, launching benchmark")
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
    reporter.set(90, "Benchmark launched")
    return run


def launch_local_ollama(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    resolve_local_server: Callable[..., ServerHandle],
    launch_local_benchmark: Callable[..., CommandResult],
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    reporter.set(5, "Resolving local Ollama server")
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
    reporter.set(60, "Server ready, launching benchmark")
    run = launch_local_benchmark(
        spec,
        env={"OLLAMA_HOST": endpoint},
        state=state,
        state_path=state_path,
    )
    run.data["server_id"] = server.id
    reporter.set(90, "Benchmark launched")
    return run


def launch_openai_compatible(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    launch_local_benchmark: Callable[..., CommandResult],
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    reporter.set(10, "Configuring endpoint")
    env = {
        "OPENAI_BASE_URL": spec.target.base_url,
        "OPENAI_API_KEY": os.environ.get(
            spec.target.api_key_env, os.environ.get("OPENAI_API_KEY", "dummy")
        ),
    }
    reporter.set(30, "Launching benchmark")
    result = launch_local_benchmark(spec, env=env, state=state, state_path=state_path)
    reporter.set(90, "Benchmark launched")
    return result


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
    load_state: Callable[[Path | None], LauncherState],
    merge_server: Callable[[Path | None, ServerHandle], ServerHandle],
    update_state: Callable[[Path | None, Callable[[LauncherState], object]], object],
    which: Callable[[str], str | None],
    pid_is_alive: Callable[[int], bool],
    sleep: Callable[[float], None],
    now: Callable[[], float],
    warmup_command: str | None = None,
) -> ServerHandle:
    existing = _find_server(state, reuse_key)
    if existing and spec.reuse == ReuseMode.STOP_AND_REPLACE and existing.metadata.get("pid"):
        os.kill(int(existing.metadata["pid"]), signal.SIGTERM)
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
        state.servers = [
            entry for entry in state.servers if entry.reuse_key != server.reuse_key
        ] + [server]
        return merge_server(state_path, server)
    if not (spec.yes and which(executable)):
        server = _build_local_server(
            target=target,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status="planned",
            metadata=_local_server_metadata(command, warmup_command=warmup_command),
            log_path=str(Path.cwd() / f".mcode-{target.value}-{spec.serving.port}.log"),
        )
        state.servers = [
            entry for entry in state.servers if entry.reuse_key != server.reuse_key
        ] + [server]
        return merge_server(state_path, server)

    for _ in range(2):
        reservation = _reserve_local_server(
            state_path=state_path,
            reuse_key=reuse_key,
            endpoint=endpoint,
            target=target,
            command=command,
            warmup_command=warmup_command,
            log_path=str(Path.cwd() / f".mcode-{target.value}-{spec.serving.port}.log"),
            endpoint_is_healthy=endpoint_is_healthy,
            merge_server=merge_server,
            update_state=update_state,
            pid_is_alive=pid_is_alive,
            now=now,
        )
        mode = reservation["mode"]
        server = reservation["server"]
        if mode == "reuse":
            state.servers = [
                entry for entry in state.servers if entry.reuse_key != server.reuse_key
            ] + [server]
            return server
        if mode == "wait":
            waited = _wait_for_local_server(
                state_path=state_path,
                reuse_key=reuse_key,
                endpoint=endpoint,
                endpoint_is_healthy=endpoint_is_healthy,
                merge_server=merge_server,
                load_state=load_state,
                pid_is_alive=pid_is_alive,
                sleep=sleep,
                now=now,
            )
            if waited is not None:
                state.servers = [
                    entry for entry in state.servers if entry.reuse_key != waited.reuse_key
                ] + [waited]
                return waited
            continue

        with open(server.log_path or os.devnull, "a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, shell=True, stdout=handle, stderr=handle)
        pending_server = replace(
            server,
            metadata={
                **server.metadata,
                "pid": process.pid,
            },
        )
        merge_server(state_path, pending_server)
        if warmup_command:
            subprocess.run(warmup_command, shell=True, check=False)
        try:
            wait_for_endpoint(endpoint)
        except Exception as exc:
            failed_server = replace(
                pending_server,
                status="failed",
                metadata={**pending_server.metadata, "error": str(exc)},
            )
            merge_server(state_path, failed_server)
            raise
        healthy_server = replace(pending_server, status="healthy")
        state.servers = [
            entry for entry in state.servers if entry.reuse_key != healthy_server.reuse_key
        ] + [healthy_server]
        return merge_server(state_path, healthy_server)

    raise RuntimeError(f"Timed out waiting for local server reservation: {reuse_key}")


def _build_local_server(
    *,
    target: TargetKind,
    reuse_key: str,
    endpoint: str,
    status: str,
    metadata: dict[str, Any],
    log_path: str | None,
) -> ServerHandle:
    return ServerHandle(
        id=f"server-{uuid.uuid4().hex[:8]}",
        target=target.value,
        reuse_key=reuse_key,
        endpoint=endpoint,
        status=status,
        metadata=metadata,
        log_path=log_path,
    )


def _local_server_metadata(command: str, *, warmup_command: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"command": command}
    if warmup_command:
        metadata["warmup"] = warmup_command
    return metadata


def _find_server(state: LauncherState, reuse_key: str) -> ServerHandle | None:
    return next((server for server in state.servers if server.reuse_key == reuse_key), None)


def _pending_server_is_waitable(
    server: ServerHandle,
    *,
    pid_is_alive: Callable[[int], bool],
    now: Callable[[], float],
) -> bool:
    pid = server.metadata.get("pid")
    if pid is not None:
        return pid_is_alive(int(pid))
    created_at = float(server.metadata.get("created_at", 0.0))
    return (now() - created_at) < LOCAL_PENDING_GRACE_S


def _reserve_local_server(
    *,
    state_path: Path | None,
    reuse_key: str,
    endpoint: str,
    target: TargetKind,
    command: str,
    warmup_command: str | None,
    log_path: str,
    endpoint_is_healthy: Callable[[str], bool],
    merge_server: Callable[[Path | None, ServerHandle], ServerHandle],
    update_state: Callable[[Path | None, Callable[[LauncherState], object]], object],
    pid_is_alive: Callable[[int], bool],
    now: Callable[[], float],
) -> dict[str, ServerHandle | str]:
    pending_server = _build_local_server(
        target=target,
        reuse_key=reuse_key,
        endpoint=endpoint,
        status="pending",
        metadata={
            **_local_server_metadata(command, warmup_command=warmup_command),
            "created_at": now(),
        },
        log_path=log_path,
    )
    decision: dict[str, ServerHandle | str] = {"mode": "start", "server": pending_server}

    def _update(current: LauncherState) -> None:
        existing = _find_server(current, reuse_key)
        if existing is None:
            current.servers = [
                entry for entry in current.servers if entry.reuse_key != reuse_key
            ] + [pending_server]
            return
        if existing.status == "healthy" and endpoint_is_healthy(existing.endpoint):
            decision["mode"] = "reuse"
            decision["server"] = existing
            return
        if existing.status == "pending" and _pending_server_is_waitable(
            existing,
            pid_is_alive=pid_is_alive,
            now=now,
        ):
            decision["mode"] = "wait"
            decision["server"] = existing
            return
        current.servers = [entry for entry in current.servers if entry.reuse_key != reuse_key] + [
            pending_server
        ]

    update_state(state_path, _update)
    if decision["mode"] == "start":
        merge_server(state_path, pending_server)
    return decision


def _wait_for_local_server(
    *,
    state_path: Path | None,
    reuse_key: str,
    endpoint: str,
    endpoint_is_healthy: Callable[[str], bool],
    merge_server: Callable[[Path | None, ServerHandle], ServerHandle],
    load_state: Callable[[Path | None], LauncherState],
    pid_is_alive: Callable[[int], bool],
    sleep: Callable[[float], None],
    now: Callable[[], float],
) -> ServerHandle | None:
    deadline = now() + LOCAL_PENDING_STARTUP_TIMEOUT_S
    while now() < deadline:
        current = _find_server(load_state(state_path), reuse_key)
        if current is None:
            return None
        if current.status == "healthy" and endpoint_is_healthy(current.endpoint):
            return current
        if current.status != "pending":
            return None
        if not _pending_server_is_waitable(current, pid_is_alive=pid_is_alive, now=now):
            return None
        if endpoint_is_healthy(endpoint):
            return merge_server(state_path, replace(current, status="healthy"))
        sleep(0.1)
    return None


def launch_local_benchmark(
    spec: LaunchSpec,
    *,
    env: dict[str, str],
    state: LauncherState,
    state_path: Path | None,
    merge_run: Callable[[Path | None, RunHandle], RunHandle],
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
    state.runs = [entry for entry in state.runs if entry.id != run.id] + [run]
    merge_run(state_path, run)
    return CommandResult(ok=True, message="\n".join(commands), data={"run_id": run.id})


def build_local_benchmark_command(
    spec: LaunchSpec,
    *,
    run_dir: Path,
    shard_count: int,
    shard_index: int,
) -> str:
    task_ids = f"--task-ids {spec.benchmark.task_ids}" if spec.benchmark.task_ids else ""
    dataset = (
        f"--dataset {spec.benchmark.dataset or 'SWE-bench/SWE-bench_Lite'}"
        if spec.benchmark.benchmark == "swebench-lite"
        else ""
    )
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
        f"{task_ids} {dataset} {limit}"
    ).strip()
