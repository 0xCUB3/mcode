from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from mcode.launch.lsf import extract_lsf_job_id
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
from mcode.launch.progress import NullProgressReporter, ProgressReporter
from mcode.launch.state import LauncherState, load_state


def submit_bluevela_benchmark_shards(
    run: RunHandle,
    *,
    spec: LaunchSpec,
    server: ServerHandle,
    workspace: WorkspaceHandle,
    run_ssh: Callable[[str, str], str],
    build_bsub_benchmark_command: Callable[..., str],
    record_run: Callable[[RunHandle], RunHandle],
    reporter: ProgressReporter | None = None,
) -> RunHandle:
    reporter = reporter or NullProgressReporter()
    run_dir = Path(str(run.metadata["run_dir"]))
    login = str(run.metadata["login"])
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
    job_ids: list[str] = []
    if spec.yes:
        run_ssh(login, f"mkdir -p {run_dir}")
        for index, command in enumerate(commands, start=1):
            output = run_ssh(login, command).strip()
            job_ids.append(extract_lsf_job_id(output) or output)
            reporter.set(int((index / len(commands)) * 100), "Submitting benchmark shards")
    updated_run = replace(
        run,
        status="running" if spec.yes else "planned",
        metadata={
            **run.metadata,
            "commands": commands,
            "db_paths": db_paths,
            "job_ids": job_ids,
            "log_paths": log_paths,
            "server_id": server.id,
            "startup_server_status": server.status,
        },
        log_path=log_paths[0],
    )
    return record_run(updated_run)


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
    merge_workspace: Callable[[Path | None, WorkspaceHandle], WorkspaceHandle],
    merge_run: Callable[[Path | None, RunHandle], RunHandle],
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    sync_reporter = reporter.child(2, 20)
    server_reporter = reporter.child(20, 85)
    shard_reporter = reporter.child(85, 95)
    if spec.yes:
        reporter.set(1, "Resolving Podman storage")
        resolve_podman_storage(spec.target)
    sync_result = launch_sync(
        replace(
            spec,
            sync=replace(spec.sync, apply=spec.yes, check=not spec.yes),
        ),
        repo_root=repo_root,
        state_path=state_path,
        state=state,
        reporter=sync_reporter,
    )
    if not sync_result.ok:
        return sync_result
    workspace_signature = sync_result.data["signature"]
    reuse_key = build_server_reuse_key(spec)
    existing_server = find_existing_server(state, reuse_key=reuse_key)

    workspace = WorkspaceHandle(signature=workspace_signature, path=sync_result.data["remote_path"])
    state.workspaces = [
        entry for entry in state.workspaces if entry.signature != workspace.signature
    ] + [workspace]
    merge_workspace(state_path, workspace)

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    run_dir = Path(spec.target.workspace_root) / "runs" / run_id
    pending_run = RunHandle(
        id=run_id,
        target=TargetKind.BLUEVELA.value,
        benchmark=spec.benchmark.benchmark,
        status="planned",
        metadata={
            "db_paths": [],
            "job_ids": [],
            "launch_spec": asdict(spec),
            "launcher_pid": os.getpid(),
            "login": spec.target.login,
            "log_paths": [],
            "run_dir": str(run_dir),
            "workspace_signature": workspace_signature,
        },
        log_path=None,
    )
    current_run = pending_run

    def _record_run(run: RunHandle) -> RunHandle:
        nonlocal current_run
        current_run = run
        state.runs = [entry for entry in state.runs if entry.id != run.id] + [run]
        return merge_run(state_path, run)

    def _record_pending_run(server: ServerHandle) -> None:
        log_paths = [server.log_path] if server.log_path else []
        run = replace(
            pending_run,
            status="pending" if spec.yes else "planned",
            metadata={
                **pending_run.metadata,
                "log_paths": log_paths,
                "server_id": server.id,
                "startup_server_status": server.status,
            },
            log_path=server.log_path,
        )
        _record_run(run)

    try:
        if spec.yes:
            server_reporter.set(0, "Starting or reusing Blue Vela server")
            server = resolve_bluevela_server(
                spec,
                state=state,
                state_path=state_path,
                reuse_key=reuse_key,
                workspace_signature=workspace_signature,
                existing_server=existing_server,
                on_pending_server=_record_pending_run,
                reporter=server_reporter,
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
            _record_pending_run(server)
    except Exception as exc:
        persisted_run = next(
            (entry for entry in load_state(state_path).runs if entry.id == run_id),
            None,
        )
        if persisted_run is not None and persisted_run.status == "stopped":
            state.runs = [entry for entry in state.runs if entry.id != persisted_run.id] + [
                persisted_run
            ]
            current_run = persisted_run
            return CommandResult(
                ok=False,
                message=str(exc),
                data={"run_id": run_id, "workspace_signature": workspace_signature},
            )
        failed_run = replace(
            current_run,
            status="failed",
            metadata={
                **current_run.metadata,
                "error": str(exc),
            },
        )
        _record_run(failed_run)
        return CommandResult(
            ok=False,
            message=str(exc),
            data={"run_id": run_id, "workspace_signature": workspace_signature},
        )

    run = submit_bluevela_benchmark_shards(
        pending_run,
        spec=spec,
        server=server,
        workspace=workspace,
        run_ssh=run_ssh,
        build_bsub_benchmark_command=build_bsub_benchmark_command,
        record_run=_record_run,
        reporter=shard_reporter,
    )
    reporter.set(95, "Finalizing launch metadata")
    return CommandResult(
        ok=True,
        message="\n".join(run.metadata["job_ids"])
        if spec.yes
        else "\n".join(run.metadata["commands"]),
        data={"run_id": run_id, "server_id": server.id, "workspace_signature": workspace_signature},
    )
