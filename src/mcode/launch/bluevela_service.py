from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from mcode.launch.models import (
    BlueVelaTargetSpec,
    CommandResult,
    LaunchSpec,
    ReuseMode,
    RunHandle,
    ServerHandle,
    TargetKind,
    WorkspaceHandle,
)
from mcode.launch.state import LauncherState


def launch_bluevela(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state: LauncherState,
    state_path: Path | None,
    launch_sync: Callable[..., CommandResult],
    resolve_podman_storage: Callable[[BlueVelaTargetSpec], tuple[str, str]],
    build_server_reuse_key: Callable[[LaunchSpec], str],
    find_existing_server: Callable[..., ServerHandle | None],
    resolve_bluevela_server: Callable[..., ServerHandle],
    run_ssh: Callable[[str, str], str],
    build_vllm_command: Callable[..., str],
    build_bsub_benchmark_command: Callable[..., str],
    save_state: Callable[[Path | None, LauncherState], None],
) -> CommandResult:
    if spec.yes:
        resolve_podman_storage(spec.target)
    sync_result = launch_sync(
        replace(
            spec,
            sync=replace(spec.sync, apply=spec.yes, check=not spec.yes),
        ),
        repo_root=repo_root,
        state_path=state_path,
    )
    if not sync_result.ok:
        return sync_result
    workspace_signature = sync_result.data["signature"]
    reuse_key = build_server_reuse_key(spec)
    existing_server = find_existing_server(state, reuse_key=reuse_key)
    if spec.yes:
        server = resolve_bluevela_server(
            spec,
            state=state,
            state_path=state_path,
            reuse_key=reuse_key,
            workspace_signature=workspace_signature,
            existing_server=existing_server,
        )
    elif existing_server and spec.reuse == ReuseMode.PREFER:
        server = existing_server
    else:
        run_dir = Path(spec.target.workspace_root) / "runs" / uuid.uuid4().hex[:12]
        command = build_vllm_command(spec, run_dir=run_dir)
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=TargetKind.BLUEVELA.value,
            reuse_key=reuse_key,
            endpoint=f"http://pending:{spec.serving.port}/v1",
            status="planned",
            metadata={
                "command": command,
                "login": spec.target.login,
                "run_dir": str(run_dir),
            },
            log_path=str(run_dir / "vllm.log"),
        )

    workspace = WorkspaceHandle(signature=workspace_signature, path=sync_result.data["remote_path"])
    if not any(entry.signature == workspace.signature for entry in state.workspaces):
        state.workspaces.append(workspace)

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    run_dir = Path(spec.target.workspace_root) / "runs" / run_id
    if spec.yes:
        run_ssh(spec.target.login, f"mkdir -p {run_dir}")
    log_paths = [
        str(run_dir / f"benchmark-shard-{shard_index}.log")
        for shard_index in range(spec.benchmark.parallelism)
    ]
    db_paths = [
        str(run_dir / f"diagnostic-shard-{shard_index}.db")
        for shard_index in range(spec.benchmark.parallelism)
    ]
    commands = [
        build_bsub_benchmark_command(
            spec,
            workspace_path=Path(workspace.path),
            run_dir=run_dir,
            shard_index=shard_index,
            endpoint=server.endpoint,
        )
        for shard_index in range(spec.benchmark.parallelism)
    ]
    job_ids = (
        [run_ssh(spec.target.login, command).strip() for command in commands] if spec.yes else []
    )
    run = RunHandle(
        id=run_id,
        target=TargetKind.BLUEVELA.value,
        benchmark=spec.benchmark.benchmark,
        status="running" if spec.yes else "planned",
        metadata={
            "job_ids": job_ids,
            "commands": commands,
            "db_paths": db_paths,
            "login": spec.target.login,
            "log_paths": log_paths,
            "run_dir": str(run_dir),
            "workspace_signature": workspace_signature,
        },
        log_path=log_paths[0],
    )
    state.runs.append(run)
    save_state(state_path, state)
    return CommandResult(
        ok=True,
        message="\n".join(job_ids) if spec.yes else "\n".join(commands),
        data={"run_id": run_id, "server_id": server.id, "workspace_signature": workspace_signature},
    )
