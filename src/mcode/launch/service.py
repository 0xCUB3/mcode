from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from warnings import warn

from mcode.launch.bluevela_service import launch_bluevela as launch_bluevela_impl
from mcode.launch.config import LaunchConfig
from mcode.launch.follow import follow_run_logs
from mcode.launch.local_service import (
    launch_local_benchmark as launch_local_benchmark_impl,
)
from mcode.launch.local_service import (
    launch_local_ollama as launch_local_ollama_impl,
)
from mcode.launch.local_service import (
    launch_local_vllm as launch_local_vllm_impl,
)
from mcode.launch.local_service import (
    launch_openai_compatible as launch_openai_compatible_impl,
)
from mcode.launch.local_service import (
    resolve_local_server as resolve_local_server_impl,
)
from mcode.launch.lsf import extract_lsf_job_id
from mcode.launch.models import (
    BenchSpec,
    BlueVelaTargetSpec,
    CommandResult,
    LaunchSpec,
    LocalOllamaTargetSpec,
    LocalVllmTargetSpec,
    OpenAICompatibleTargetSpec,
    ReuseMode,
    RunHandle,
    ServerHandle,
    ServingSpec,
    SyncMode,
    SyncSpec,
    TargetKind,
    WorkspaceHandle,
)
from mcode.launch.profiles import resolve_serving_profile
from mcode.launch.progress import NullProgressReporter, ProgressReporter
from mcode.launch.providers.bluevela import (
    _bluevela_podman_base_dirs,
    build_bluevela_benchmark_command,
    build_bluevela_lock_path,
    build_bluevela_server_registry_path,
    build_bluevela_server_reuse_key,
    build_bluevela_vllm_command,
    build_remote_workspace_prepare_command,
)
from mcode.launch.remote_scripts import (
    build_remote_healthcheck_command,
    build_uv_sync_command,
)
from mcode.launch.state import (
    LauncherState,
    load_state,
    merge_run,
    merge_server,
    merge_workspace,
    update_state,
)
from mcode.launch.sync import build_sync_plan, list_untracked_files, tracked_overlay_patch

BLUEVELA_VLLM_FAILED_MARKERS = (
    "Exited with exit code",
    "Failed to obtain podman configuration",
    "Engine core initialization failed",
    "No available memory for the cache blocks",
    "cannot re-exec process to join the existing user namespace",
    "invalid tool call parser",
)


def _normalize_task_ids_arg(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        path = Path(raw)
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        return raw
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        ids = _split_task_ids(text)
    else:
        if isinstance(data, list):
            ids = [str(item).strip() for item in data if str(item).strip()]
        elif isinstance(data, dict) and "tasks" in data:
            ids = []
            for value in data["tasks"].values():
                if isinstance(value, list):
                    ids.extend(str(item).strip() for item in value if str(item).strip())
        else:
            raise ValueError(f"Cannot parse task IDs from {raw}")
    return ",".join(ids)


def _split_task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _find_missing_task_ids(
    benchmark: str,
    split: str,
    dataset: str | None,
    task_ids: str | None,
) -> list[str] | None:
    if benchmark != "swebench-lite" or not dataset or not task_ids:
        return None
    from mcode.bench.swebench_lite import load_swebench_lite

    ids = _split_task_ids(task_ids)
    matched = {
        task.instance_id
        for task in load_swebench_lite(
            Path("/tmp"),
            split=split,
            instance_ids=ids,
            dataset_name=dataset,
        )
    }
    return [task_id for task_id in ids if task_id not in matched]


def build_launch_spec(
    *,
    config: LaunchConfig,
    target: str,
    model: str,
    benchmark: str,
    backend: str | None,
    split: str | None,
    loop_budget: int,
    timeout: int,
    parallelism: int,
    limit: int | None,
    task_ids: str | None,
    reuse: str,
    sync_mode: str,
    ref: str,
    json_mode: bool,
    yes: bool,
    follow: bool,
    tp: int,
    dp: int,
    api_server_count: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    port: int | None,
    serving_profile: str | None,
    no_auto_profile: bool,
    keep_alive: str | None,
    ollama_num_parallel: int | None,
    ollama_max_queue: int | None,
    openai_base_url: str | None,
    dataset: str | None = None,
) -> LaunchSpec:
    kind = TargetKind(target)
    profile = resolve_serving_profile(model)
    if serving_profile:
        profile = (
            profile
            if profile.name == serving_profile
            else type(profile)(name=serving_profile, flags=[])
        )
    if no_auto_profile:
        profile = type(profile)()

    if kind == TargetKind.BLUEVELA:
        target_spec = BlueVelaTargetSpec(
            kind=kind,
            login=config.bluevela.login,
            workspace_root=config.bluevela.workspace_root,
            queue=config.bluevela.queue,
            group=config.bluevela.group,
            shared_root=config.bluevela.shared_root,
            hf_env=config.bluevela.hf_env,
            podman_graphroot=config.bluevela.podman_graphroot,
            podman_runroot=config.bluevela.podman_runroot,
            results_root=config.bluevela.results_root,
        )
        engine = "vllm"
        default_port = 8321
    elif kind == TargetKind.LOCAL_VLLM:
        target_spec = LocalVllmTargetSpec(kind=kind, host=config.local_vllm.host)
        engine = "vllm"
        default_port = config.local_vllm.port
    elif kind == TargetKind.LOCAL_OLLAMA:
        target_spec = LocalOllamaTargetSpec(kind=kind, host=config.local_ollama.host)
        engine = "ollama"
        default_port = config.local_ollama.port
    else:
        target_spec = OpenAICompatibleTargetSpec(
            kind=kind, base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
        )
        engine = "openai-compatible"
        default_port = 0

    serving = ServingSpec(
        engine=engine,
        port=port or default_port,
        tensor_parallel=tp,
        data_parallel=dp,
        api_server_count=api_server_count,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        profile=profile,
        keep_alive=keep_alive or config.local_ollama.keep_alive,
        ollama_num_parallel=ollama_num_parallel or config.local_ollama.num_parallel,
        ollama_max_queue=ollama_max_queue or config.local_ollama.max_queue,
    )
    resolved_split = _resolve_benchmark_split(benchmark, split)
    resolved_dataset = dataset or (
        "SWE-bench/SWE-bench_Lite" if benchmark == "swebench-lite" else None
    )
    bench = BenchSpec(
        benchmark=benchmark,
        backend=backend or ("ollama" if kind == TargetKind.LOCAL_OLLAMA else "openai"),
        split=resolved_split,
        dataset=resolved_dataset,
        loop_budget=loop_budget,
        timeout=timeout,
        parallelism=parallelism,
        limit=limit,
        task_ids=_normalize_task_ids_arg(task_ids),
    )
    missing_task_ids = _find_missing_task_ids(
        bench.benchmark, bench.split, bench.dataset, bench.task_ids
    )
    if missing_task_ids is not None and bench.task_ids:
        total = len(_split_task_ids(bench.task_ids))
        matched = total - len(missing_task_ids)
        message = (
            f"--task-ids matched {matched}/{total} tasks in {bench.dataset} split={bench.split}"
        )
        if missing_task_ids:
            message += f". Missing examples: {', '.join(missing_task_ids[:3])}"
        if matched == 0:
            raise ValueError(message)
        if missing_task_ids:
            warn(message, stacklevel=2)
    sync = SyncSpec(mode=SyncMode(sync_mode), ref=ref)
    return LaunchSpec(
        target=target_spec,
        model=model,
        benchmark=bench,
        serving=serving,
        sync=sync,
        reuse=ReuseMode(reuse),
        json_mode=json_mode,
        yes=yes,
        follow=follow,
    )


def build_admin_launch_spec(
    *,
    config: LaunchConfig,
    target: str,
    model: str,
    json_mode: bool,
    openai_base_url: str | None = None,
    sync_mode: str = "git-overlay",
    ref: str = "HEAD",
) -> LaunchSpec:
    return build_launch_spec(
        config=config,
        target=target,
        model=model,
        benchmark="swebench-live",
        backend=None,
        split="verified",
        loop_budget=15,
        timeout=1800,
        parallelism=1,
        limit=None,
        task_ids=None,
        reuse="prefer",
        sync_mode=sync_mode,
        ref=ref,
        json_mode=json_mode,
        yes=False,
        follow=False,
        tp=1,
        dp=1,
        api_server_count=1,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        port=None,
        serving_profile=None,
        no_auto_profile=False,
        keep_alive=None,
        ollama_num_parallel=None,
        ollama_max_queue=None,
        openai_base_url=openai_base_url,
    )


def _resolve_benchmark_split(benchmark: str, split: str | None) -> str:
    if split:
        return split
    if benchmark == "swebench-lite":
        return "test"
    return "verified"


def launch_doctor(spec: LaunchSpec) -> CommandResult:
    if isinstance(spec.target, BlueVelaTargetSpec):
        return _run_bluevela_doctor(spec.target)
    if isinstance(spec.target, LocalVllmTargetSpec):
        return _run_local_vllm_doctor(spec.target)
    if isinstance(spec.target, LocalOllamaTargetSpec):
        return _run_local_ollama_doctor(spec.target)
    assert isinstance(spec.target, OpenAICompatibleTargetSpec)
    return _run_openai_compatible_doctor(spec.target)


def launch_sync(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state_path: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    return _launch_sync(
        spec,
        repo_root=repo_root,
        state_path=state_path,
        reporter=reporter,
    )


def _launch_sync(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state_path: Path | None = None,
    state: LauncherState | None = None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    if not isinstance(spec.target, BlueVelaTargetSpec):
        reporter.finish("Sync not required")
        return CommandResult(ok=True, message="Sync is only required for Blue Vela in V1.")

    reporter.set(5, "Planning workspace sync")
    if state is None:
        state = load_state(state_path)
    existing = next(
        (w for w in state.workspaces if w.path.startswith(spec.target.workspace_root)), None
    )
    plan = build_sync_plan(
        repo_root,
        sync=spec.sync,
        workspace_root=spec.target.workspace_root,
        existing=existing,
    )
    untracked_files = list_untracked_files(repo_root)
    blocking_untracked = [
        path
        for path in untracked_files
        if path.startswith(("src/", "tests/", "deploy/")) or path in {"README.md", "pyproject.toml"}
    ]
    data = {
        "signature": plan.signature,
        "remote_path": plan.remote_path,
        "ref_sha": plan.ref_sha,
        "diff_summary": plan.diff_summary,
        "is_noop": plan.is_noop,
        "untracked_files": untracked_files,
        "blocking_untracked_files": blocking_untracked,
    }
    if blocking_untracked:
        reporter.finish("Sync blocked")
        return CommandResult(
            ok=False,
            message=(
                "Untracked files are excluded from sync. Stage them with git add before using "
                "git-overlay or git-ref sync."
            ),
            data=data,
        )
    if spec.sync.check:
        reporter.set(60, "Checking remote workspace")
        manifest = _read_remote_workspace_manifest(spec.target, plan.remote_path)
        data["remote_manifest"] = manifest
        data["is_noop"] = bool(manifest and manifest.get("signature") == plan.signature)
        reporter.finish("Sync check complete")
        return CommandResult(ok=True, message="Sync plan computed.", data=data)

    reporter.set(20, "Preparing remote workspace")
    _sync_bluevela_workspace(spec.target, repo_root, plan, reporter=reporter)
    workspace = WorkspaceHandle(signature=plan.signature, path=plan.remote_path, metadata=data)
    merge_workspace(state_path, workspace)
    reporter.finish("Sync complete")
    return CommandResult(ok=True, message=plan.remote_path, data=data)


def launch_status(*, state_path: Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    return {
        "servers": [asdict(server) for server in state.servers],
        "runs": [asdict(run) for run in state.runs],
        "workspaces": [asdict(workspace) for workspace in state.workspaces],
    }


def launch_fetch(
    run_id: str,
    *,
    destination: Path,
    state_path: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    reporter.set(5, "Loading run state")
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        reporter.finish("Fetch failed")
        return CommandResult(ok=False, message=f"Unknown run id: {run_id}")
    if run.target != TargetKind.BLUEVELA.value:
        reporter.finish("Fetch failed")
        return CommandResult(ok=False, message="Fetch is only implemented for Blue Vela runs.")
    remote_path = run.metadata.get("run_dir")
    if not remote_path:
        reporter.finish("Fetch failed")
        return CommandResult(ok=False, message="Run has no remote directory recorded.")
    destination.mkdir(parents=True, exist_ok=True)
    login = run.metadata.get("login")
    reporter.set(30, "Copying results")
    cmd = ["rsync", "-az", f"{login}:{remote_path}/", str(destination)]
    subprocess.run(cmd, check=True)
    reporter.finish("Fetch complete")
    return CommandResult(ok=True, message=f"Fetched {run_id} into {destination}")


def launch_stop(run_id: str, *, state_path: Path | None = None) -> CommandResult:
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        server = next((entry for entry in state.servers if entry.id == run_id), None)
        if server is None:
            return CommandResult(ok=False, message=f"Unknown id: {run_id}")
        return _stop_server(server, state_path)
    return _stop_run(run, state_path)


def launch_stop_all(
    *,
    target: str | None = None,
    state_path: Path | None = None,
    bluevela_login: str | None = None,
    bluevela_workspace_root: str | None = None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    target_filter = TargetKind(target).value if target else None
    if target_filter == TargetKind.BLUEVELA.value and (
        not bluevela_login or not bluevela_workspace_root
    ):
        reporter.finish("Stop failed")
        return CommandResult(
            ok=False,
            message="Blue Vela stop --all requires Blue Vela login and workspace root.",
        )

    reporter.set(5, "Loading launcher state")
    state = load_state(state_path)
    runs = [run for run in state.runs if target_filter is None or run.target == target_filter]
    servers = [
        server
        for server in state.servers
        if target_filter is None or server.target == target_filter
    ]
    stopped_server_ids: set[str] = set()
    runs_stopped = 0
    servers_stopped = 0

    reporter.set(15, "Stopping tracked runs")
    for run in runs:
        result = _stop_run(run, state_path, state=state)
        if result.ok:
            runs_stopped += 1
            if run.metadata.get("startup_server_status") == "pending" and run.metadata.get(
                "server_id"
            ):
                stopped_server_ids.add(str(run.metadata["server_id"]))
            if runs:
                reporter.set(15 + int((runs_stopped / len(runs)) * 35), "Stopping tracked runs")

    reporter.set(50, "Stopping tracked servers")
    for server in servers:
        if server.id in stopped_server_ids:
            continue
        result = _stop_server(server, state_path, state=state)
        if result.ok:
            servers_stopped += 1
            if servers:
                reporter.set(
                    50 + int((servers_stopped / len(servers)) * 25),
                    "Stopping tracked servers",
                )

    data = {
        "runs_stopped": runs_stopped,
        "servers_stopped": servers_stopped,
    }
    if (
        target_filter in (None, TargetKind.BLUEVELA.value)
        and bluevela_login
        and bluevela_workspace_root
    ):
        reporter.set(80, "Sweeping Blue Vela jobs and locks")
        data.update(_stop_all_bluevela_cluster(bluevela_login, bluevela_workspace_root))
    reporter.finish("Stop complete")
    return CommandResult(ok=True, message="Stopped all", data=data)


def launch_attach(
    run_id: str,
    *,
    follow: bool = True,
    state_path: Path | None = None,
) -> CommandResult:
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        return CommandResult(ok=False, message=f"Unknown run id: {run_id}")
    if follow:
        return _follow_run_logs(run)
    return CommandResult(
        ok=True, message=run.log_path or "No log path recorded.", data=run.metadata
    )


def launch_run(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state_path: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    if spec.follow and not spec.yes:
        return CommandResult(ok=False, message="--follow requires --yes")
    state = load_state(state_path)
    if isinstance(spec.target, BlueVelaTargetSpec):
        result = _launch_bluevela(
            spec,
            repo_root=repo_root,
            state=state,
            state_path=state_path,
            reporter=reporter,
        )
    elif isinstance(spec.target, LocalVllmTargetSpec):
        result = _launch_local_vllm(spec, state=state, state_path=state_path, reporter=reporter)
    elif isinstance(spec.target, LocalOllamaTargetSpec):
        result = _launch_local_ollama(spec, state=state, state_path=state_path, reporter=reporter)
    else:
        result = _launch_openai_compatible(
            spec, state=state, state_path=state_path, reporter=reporter
        )
    if spec.follow and result.ok and "run_id" in result.data:
        attach_result = launch_attach(result.data["run_id"], state_path=state_path)
        if not attach_result.ok:
            return attach_result
    reporter.finish("Launch complete" if result.ok else "Launch failed")
    return result


def _follow_run_logs(run: RunHandle) -> CommandResult:
    return follow_run_logs(run)


def _launch_bluevela(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state: LauncherState,
    state_path: Path | None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    reporter = reporter or NullProgressReporter()
    return launch_bluevela_impl(
        spec,
        repo_root=repo_root,
        state=state,
        state_path=state_path,
        launch_sync=_launch_sync,
        resolve_podman_storage=_resolve_bluevela_podman_storage,
        build_server_reuse_key=build_bluevela_server_reuse_key,
        find_existing_server=_find_existing_server,
        resolve_bluevela_server=_resolve_bluevela_server,
        run_ssh=_run_ssh,
        build_vllm_command=build_bluevela_vllm_command,
        build_bsub_benchmark_command=_build_bluevela_bsub_benchmark_command,
        merge_workspace=merge_workspace,
        merge_run=merge_run,
        reporter=reporter,
    )


def _launch_local_vllm(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    return launch_local_vllm_impl(
        spec,
        state=state,
        state_path=state_path,
        resolve_local_server=_resolve_local_server,
        launch_local_benchmark=_launch_local_benchmark,
        reporter=reporter,
    )


def _launch_local_ollama(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    return launch_local_ollama_impl(
        spec,
        state=state,
        state_path=state_path,
        resolve_local_server=_resolve_local_server,
        launch_local_benchmark=_launch_local_benchmark,
        reporter=reporter,
    )


def _launch_openai_compatible(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    reporter: ProgressReporter | None = None,
) -> CommandResult:
    return launch_openai_compatible_impl(
        spec,
        state=state,
        state_path=state_path,
        launch_local_benchmark=_launch_local_benchmark,
        reporter=reporter,
    )


def _stop_run(
    run: RunHandle,
    state_path: Path | None,
    *,
    state: LauncherState | None = None,
) -> CommandResult:
    current_state = state or load_state(state_path)
    job_id = run.metadata.get("job_id")
    if run.target == TargetKind.BLUEVELA.value and run.metadata.get("job_ids"):
        for current in run.metadata["job_ids"]:
            normalized_job_id = extract_lsf_job_id(current)
            if normalized_job_id:
                _maybe_bkill_bluevela_job(run.metadata["login"], normalized_job_id)
    elif (
        run.target == TargetKind.BLUEVELA.value
        and run.metadata.get("startup_server_status") == "pending"
        and run.metadata.get("server_id")
    ):
        pending_server = next(
            (entry for entry in current_state.servers if entry.id == run.metadata["server_id"]),
            None,
        )
        if pending_server is not None:
            _stop_server(pending_server, state_path, state=current_state)
    elif run.target == TargetKind.BLUEVELA.value and job_id:
        normalized_job_id = extract_lsf_job_id(job_id)
        if normalized_job_id:
            _maybe_bkill_bluevela_job(run.metadata["login"], normalized_job_id)
    elif run.metadata.get("pids"):
        for current in run.metadata["pids"]:
            try:
                os.kill(int(current), signal.SIGTERM)
            except ProcessLookupError:
                continue
    elif "pid" in run.metadata:
        try:
            os.kill(int(run.metadata["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
    run.status = "stopped"
    if state is not None:
        state.runs = [entry for entry in state.runs if entry.id != run.id] + [run]
    merge_run(state_path, run)
    return CommandResult(ok=True, message=f"Stopped {run.id}")


def _stop_server(
    server: ServerHandle,
    state_path: Path | None,
    *,
    state: LauncherState | None = None,
) -> CommandResult:
    job_id = server.metadata.get("job_id")
    if server.target == TargetKind.BLUEVELA.value and job_id:
        normalized_job_id = extract_lsf_job_id(job_id)
        if normalized_job_id:
            _maybe_bkill_bluevela_job(server.metadata["login"], normalized_job_id)
        registry_path = server.metadata.get("registry_path")
        if registry_path:
            _run_ssh(server.metadata["login"], f"rm -f {shlex.quote(registry_path)}")
    elif "pid" in server.metadata:
        try:
            os.kill(int(server.metadata["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
    server.status = "stopped"
    if state is not None:
        state.servers = [
            entry for entry in state.servers if entry.reuse_key != server.reuse_key
        ] + [server]
    merge_server(state_path, server)
    return CommandResult(ok=True, message=f"Stopped {server.id}")


def _run_ssh_result(
    login: str, command: str, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    remote = f"bash -lc {shlex.quote(command)}"
    return subprocess.run(
        ["ssh", "-n", login, remote],
        check=check,
        text=True,
        capture_output=True,
    )


def _run_ssh(login: str, command: str) -> str:
    res = _run_ssh_result(login, command)
    return res.stdout.strip() or res.stderr.strip()


def _stop_all_bluevela_cluster(login: str, workspace_root: str) -> dict[str, int]:
    command = (
        "jobs=$(bjobs -w 2>/dev/null | awk '/mcode-vllm|mcode-bench/ {print $1}'); "
        'jobs_killed=0; for job in $jobs; do bkill "$job" >/dev/null 2>&1 || true; '
        "jobs_killed=$((jobs_killed+1)); done; "
        f"registries_removed=$(find {workspace_root}/state/servers -maxdepth 1 -name '*.json' "
        "2>/dev/null | wc -l | tr -d ' '); "
        f"rm -f {workspace_root}/state/servers/*.json 2>/dev/null || true; "
        f"locks_removed=$(find {workspace_root}/locks -mindepth 1 -maxdepth 1 "
        "2>/dev/null | wc -l | tr -d ' '); "
        f"rm -rf {workspace_root}/locks/* 2>/dev/null || true; "
        "printf 'jobs_killed=%s\nregistries_removed=%s\nlocks_removed=%s\n' "
        '"$jobs_killed" "$registries_removed" "$locks_removed"'
    )
    output = _run_ssh(login, command)
    data: dict[str, int] = {
        "jobs_killed": 0,
        "registries_removed": 0,
        "locks_removed": 0,
    }
    for line in output.splitlines():
        key, _, value = line.partition("=")
        if key in data:
            data[key] = int(value or "0")
    return data


def _maybe_bkill_bluevela_job(login: str, job_id: str) -> None:
    result = _run_ssh_result(login, f"bkill {job_id}", check=False)
    if result.returncode == 0:
        return
    output = f"{result.stdout}\n{result.stderr}"
    if "already finished" in output or "is not found" in output:
        return
    raise subprocess.CalledProcessError(
        result.returncode,
        result.args,
        output=result.stdout,
        stderr=result.stderr,
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_bluevela_doctor(target: BlueVelaTargetSpec) -> CommandResult:
    checks = {
        "ssh": "echo ok",
        "uv": "command -v uv >/dev/null",
        "bsub": "command -v bsub >/dev/null",
        "bjobs": "command -v bjobs >/dev/null",
        "podman": "command -v podman >/dev/null",
        "workspace_root": f"mkdir -p {target.workspace_root}",
    }
    results: dict[str, bool] = {}
    for name, command in checks.items():
        try:
            _run_ssh(target.login, command)
            results[name] = True
        except subprocess.CalledProcessError:
            results[name] = False
    try:
        graphroot, runroot = _resolve_bluevela_podman_storage(target)
        results["podman_storage"] = True
    except RuntimeError:
        graphroot, runroot = _bluevela_podman_base_dirs(target)
        results["podman_storage"] = False
    return CommandResult(
        ok=all(results.values()),
        message="Blue Vela doctor complete.",
        data={**results, "podman_graphroot": graphroot, "podman_runroot": runroot},
    )


def _run_local_vllm_doctor(target: LocalVllmTargetSpec) -> CommandResult:
    ok = shutil.which("vllm") is not None
    return CommandResult(
        ok=ok, message="Local vLLM doctor complete.", data={"host": target.host, "vllm": ok}
    )


def _run_local_ollama_doctor(target: LocalOllamaTargetSpec) -> CommandResult:
    ok = shutil.which("ollama") is not None
    return CommandResult(
        ok=ok, message="Local Ollama doctor complete.", data={"host": target.host, "ollama": ok}
    )


def _run_openai_compatible_doctor(target: OpenAICompatibleTargetSpec) -> CommandResult:
    ok = _endpoint_is_healthy(target.base_url)
    return CommandResult(
        ok=ok, message="OpenAI-compatible doctor complete.", data={"base_url": target.base_url}
    )


def _endpoint_is_healthy(base_url: str) -> bool:
    if not base_url:
        return False
    root = base_url.rstrip("/")
    candidates = [f"{root}/health", f"{root}/models", f"{root}/v1/models", f"{root}/api/tags"]
    for url in candidates:
        try:
            with urlopen(Request(url, method="GET"), timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, URLError, ValueError):
            continue
    return False


def _remote_endpoint_is_healthy(login: str, base_url: str) -> bool:
    if not base_url:
        return False
    command = build_remote_healthcheck_command(base_url)
    result = _run_ssh_result(login, command, check=False)
    return result.returncode == 0


def _wait_for_endpoint(base_url: str, *, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _endpoint_is_healthy(base_url):
            return
        time.sleep(2)
    raise RuntimeError(f"Endpoint did not become healthy: {base_url}")


def _wait_for_bluevela_endpoint(
    login: str,
    base_url: str,
    *,
    log_path: str,
    job_id: str | None = None,
    timeout_s: int = 300,
    max_timeout_s: int = 1800,
    reporter: ProgressReporter | None = None,
) -> None:
    reporter = reporter or NullProgressReporter()
    host = urlsplit(base_url).hostname or "remote host"
    started_at = time.time()
    last_progress_at = started_at
    last_log_tail = ""
    while time.time() - started_at < max_timeout_s:
        job_active = bool(job_id and _bluevela_job_is_active(login, job_id))
        if _remote_endpoint_is_healthy(login, base_url):
            return
        log_tail = _run_ssh(login, f"test -f {log_path} && tail -n 20 {log_path} || true")
        if log_tail != last_log_tail:
            last_progress_at = time.time()
            last_log_tail = log_tail
        reporter.set(
            _bluevela_health_wait_progress(log_tail),
            _describe_bluevela_health_wait(host, log_tail),
        )
        failed = any(marker in log_tail for marker in BLUEVELA_VLLM_FAILED_MARKERS)
        if failed:
            raise RuntimeError(
                f"Blue Vela vLLM job failed before endpoint was healthy:\n{log_tail}"
            )
        if job_id and not job_active:
            raise RuntimeError(
                f"Blue Vela vLLM job exited before endpoint was healthy:\n{log_tail}"
            )
        if not job_active and time.time() - last_progress_at > timeout_s:
            raise RuntimeError(
                "Endpoint did not become healthy before startup progress stalled: "
                f"{base_url}\n{log_tail}"
            )
        time.sleep(2)
    raise RuntimeError(f"Endpoint did not become healthy: {base_url}")


def _describe_bluevela_health_wait(host: str, log_tail: str) -> str:
    lowered = log_tail.lower()
    # Check most-advanced stage first so early markers lingering in the
    # 20-line tail don't shadow later stages.
    if "compile and warming up model" in lowered or "initial profiling/warmup run took" in lowered:
        return f"Warming up model on {host}"
    if "capturing cuda graphs" in lowered:
        return f"Capturing CUDA graphs on {host}"
    if "gpu kv cache size" in lowered or "available kv cache memory" in lowered:
        return f"Allocating KV cache on {host}"
    if any(
        marker in lowered
        for marker in (
            "dynamo bytecode transform time",
            "compiling a graph",
            "torch.compile took",
        )
    ):
        return f"Compiling model on {host}"
    if "loading safetensors checkpoint shards" in lowered or "starting to load model" in lowered:
        return f"Loading model weights on {host}"
    if any(
        marker in lowered
        for marker in (
            "trying to pull",
            "copying blob",
            "getting image source signatures",
            "writing manifest to image destination",
            "storing signatures",
        )
    ):
        return f"Pulling vLLM image on {host}"
    if any(
        marker in lowered
        for marker in (
            "non-default args:",
            "api server",
            "engine process",
            "uvicorn",
            "starting vllm",
        )
    ):
        return f"Starting vLLM on {host}"
    return f"Waiting for server health on {host}"


def _bluevela_health_wait_progress(log_tail: str) -> int:
    """Return 0-100 progress within the health-wait phase.

    Checks most-advanced stage first so early markers lingering in the
    20-line tail don't shadow later stages.
    """
    lowered = log_tail.lower()
    if "compile and warming up model" in lowered or "initial profiling/warmup run took" in lowered:
        return 95
    if "capturing cuda graphs" in lowered:
        return 80
    if "gpu kv cache size" in lowered or "available kv cache memory" in lowered:
        return 72
    if any(
        marker in lowered
        for marker in (
            "dynamo bytecode transform time",
            "compiling a graph",
            "torch.compile took",
        )
    ):
        return 60
    if "loading safetensors checkpoint shards" in lowered:
        match = re.search(r"(\d+)%\s+Completed", log_tail)
        if match:
            return 25 + int(int(match.group(1)) * 0.30)
        return 25
    if "starting to load model" in lowered or "resolved architecture" in lowered:
        return 20
    if any(
        marker in lowered
        for marker in (
            "trying to pull",
            "copying blob",
            "getting image source signatures",
            "writing manifest to image destination",
            "storing signatures",
        )
    ):
        return 10
    if any(
        marker in lowered
        for marker in (
            "non-default args:",
            "api server",
            "engine process",
            "uvicorn",
            "starting vllm",
        )
    ):
        return 5
    return 2


def _read_remote_workspace_manifest(
    target: BlueVelaTargetSpec, remote_path: str
) -> dict[str, Any] | None:
    command = (
        f"test -f {remote_path}/.mcode-launch-workspace.json "
        f"&& cat {remote_path}/.mcode-launch-workspace.json || true"
    )
    output = _run_ssh(target.login, command)
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _read_remote_json(login: str, path: str) -> dict[str, Any] | None:
    output = _run_ssh(login, f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || true")
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _write_remote_json(login: str, path: str, payload: dict[str, Any]) -> None:
    parent = shlex.quote(str(Path(path).parent))
    encoded = shlex.quote(json.dumps(payload))
    _run_ssh(
        login,
        f"mkdir -p {parent} && printf %s {encoded} > {shlex.quote(path)}",
    )


def _bluevela_job_is_active(login: str, job_id: str) -> bool:
    result = _run_ssh_result(
        login,
        f"bjobs {shlex.quote(job_id)} >/dev/null 2>&1",
        check=False,
    )
    return result.returncode == 0


def _resolve_bluevela_podman_storage(target: BlueVelaTargetSpec) -> tuple[str, str]:
    graphroot_base, runroot_base = _bluevela_podman_base_dirs(target)
    host_tag = _run_ssh(target.login, "hostname -s").strip()
    graphroot = f"{graphroot_base}/{host_tag}/graphroot"
    runroot = f"{runroot_base}/{host_tag}/runroot"
    command = (
        "export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u); "
        f"mkdir -p ${{XDG_RUNTIME_DIR}} {graphroot} {runroot} && "
        "podman --cgroup-manager=cgroupfs --storage-driver=overlay "
        f"--root={graphroot} --runroot={runroot} info >/dev/null"
    )
    result = _run_ssh_result(
        target.login,
        command,
        check=False,
    )
    if result.returncode == 0:
        return graphroot, runroot
    output = f"{result.stdout}\n{result.stderr}".strip()
    raise RuntimeError(output or "Failed to verify Podman storage")


def _find_existing_server(state: LauncherState, *, reuse_key: str) -> ServerHandle | None:
    return next(
        (
            server
            for server in state.servers
            if server.reuse_key == reuse_key
            and server.status == "healthy"
            and (
                _remote_endpoint_is_healthy(server.metadata["login"], server.endpoint)
                if server.target == TargetKind.BLUEVELA.value
                else _endpoint_is_healthy(server.endpoint)
            )
        ),
        None,
    )


def _acquire_remote_lock(
    login: str,
    lock_path: str,
    *,
    timeout_s: int = 180,
    stale_after_s: int = 60,
) -> None:
    deadline = time.time() + timeout_s
    parent_path = str(Path(lock_path).parent)
    created_path = f"{lock_path}/created_at"
    while time.time() < deadline:
        res = _run_ssh_result(
            login,
            (
                f"mkdir -p {shlex.quote(parent_path)}; "
                "now=$(date +%s); "
                f"if mkdir {shlex.quote(lock_path)} 2>/dev/null; then "
                f'printf %s "$now" > {shlex.quote(created_path)}; '
                "exit 0; "
                "fi; "
                f"created=$(cat {shlex.quote(created_path)} 2>/dev/null || echo 0); "
                "age=$((now-created)); "
                f'if [ "$age" -gt {stale_after_s} ]; then '
                f"rm -rf {shlex.quote(lock_path)}; "
                "exit 3; "
                "fi; "
                "exit 1"
            ),
            check=False,
        )
        if res.returncode == 0:
            return
        if res.returncode == 3:
            continue
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for remote lock: {lock_path}")


def _release_remote_lock(login: str, lock_path: str) -> None:
    _run_ssh(login, f"rm -rf {shlex.quote(lock_path)}")


def _resolve_bluevela_server(
    spec: LaunchSpec,
    *,
    state: LauncherState | None = None,
    state_path: Path | None,
    reuse_key: str,
    workspace_signature: str,
    existing_server: ServerHandle | None,
    on_pending_server=None,
    reporter: ProgressReporter | None = None,
) -> ServerHandle:
    reporter = reporter or NullProgressReporter()
    assert isinstance(spec.target, BlueVelaTargetSpec)
    target = spec.target
    registry_path = build_bluevela_server_registry_path(target, reuse_key=reuse_key)
    lock_path = build_bluevela_lock_path(target, kind="server", key=reuse_key)

    def _record_server(server: ServerHandle) -> ServerHandle:
        if state is not None:
            state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
                server
            ]
        return merge_server(state_path, server)

    def _start_or_reuse_pending(*, allow_reuse: bool) -> ServerHandle:
        pending_server: ServerHandle | None = None

        _acquire_remote_lock(target.login, lock_path)
        try:
            remote_server = _read_remote_json(target.login, registry_path)
            if remote_server and (spec.reuse == ReuseMode.STOP_AND_REPLACE or not allow_reuse):
                job_id = extract_lsf_job_id(remote_server.get("metadata", {}).get("job_id"))
                if job_id and _bluevela_job_is_active(target.login, job_id):
                    _maybe_bkill_bluevela_job(target.login, job_id)
                _run_ssh(target.login, f"rm -f {shlex.quote(registry_path)}")
                remote_server = None
            if allow_reuse and remote_server and spec.reuse == ReuseMode.PREFER:
                candidate = ServerHandle(**remote_server)
                if candidate.status == "healthy" and _remote_endpoint_is_healthy(
                    target.login, candidate.endpoint
                ):
                    return _record_server(candidate)
                pending_job_id = extract_lsf_job_id(candidate.metadata.get("job_id"))
                if (
                    candidate.status == "pending"
                    and pending_job_id
                    and _bluevela_job_is_active(target.login, pending_job_id)
                ):
                    pending_server = candidate
                else:
                    _run_ssh(target.login, f"rm -f {shlex.quote(registry_path)}")
            if (
                allow_reuse
                and pending_server is None
                and existing_server
                and spec.reuse == ReuseMode.PREFER
            ):
                return _record_server(existing_server)
            if pending_server is None:
                run_dir = Path(target.workspace_root) / "runs" / uuid.uuid4().hex[:12]
                _run_ssh(target.login, f"mkdir -p {shlex.quote(str(run_dir))}")
                command = build_bluevela_vllm_command(spec, run_dir=run_dir)
                job_id_output = _run_ssh(target.login, command).strip()
                job_id = extract_lsf_job_id(job_id_output) or job_id_output
                pending_server = ServerHandle(
                    id=f"server-{uuid.uuid4().hex[:8]}",
                    target=TargetKind.BLUEVELA.value,
                    reuse_key=reuse_key,
                    endpoint=f"http://pending:{spec.serving.port}/v1",
                    status="pending",
                    metadata={
                        "job_id": job_id,
                        "login": target.login,
                        "run_dir": str(run_dir),
                        "registry_path": registry_path,
                        "workspace_signature": workspace_signature,
                    },
                    log_path=str(run_dir / "vllm.log"),
                )
                _write_remote_json(target.login, registry_path, asdict(pending_server))
            return pending_server
        finally:
            _release_remote_lock(target.login, lock_path)

    pending_server = _start_or_reuse_pending(allow_reuse=True)
    if on_pending_server is not None:
        on_pending_server(pending_server)
    reporter.set(5, "Waiting for Blue Vela server startup")
    retries = 0
    while True:
        assert pending_server is not None
        if pending_server.status == "healthy":
            server = pending_server
            break
        try:
            host_path = Path(pending_server.metadata["run_dir"]) / "vllm_host.txt"
            host = _wait_for_remote_file(
                target.login, str(host_path), reporter=reporter, host_label=host_path.parent.name
            )
            endpoint = f"http://{host}:{spec.serving.port}/v1"
            _wait_for_bluevela_endpoint(
                target.login,
                endpoint,
                log_path=pending_server.log_path,
                job_id=extract_lsf_job_id(pending_server.metadata.get("job_id")),
                reporter=reporter,
            )
            server = replace(pending_server, endpoint=endpoint, status="healthy")
            break
        except RuntimeError:
            if retries >= 1:
                raise
            retries += 1
            pending_server = _start_or_reuse_pending(allow_reuse=False)
            if on_pending_server is not None:
                on_pending_server(pending_server)
            reporter.set(5, "Retrying Blue Vela server startup")

    _acquire_remote_lock(target.login, lock_path)
    try:
        remote_server = _read_remote_json(target.login, registry_path)
        if remote_server:
            candidate = ServerHandle(**remote_server)
            candidate_job_id = extract_lsf_job_id(candidate.metadata.get("job_id"))
            server_job_id = extract_lsf_job_id(server.metadata.get("job_id"))
            if candidate.status == "healthy" and _remote_endpoint_is_healthy(
                target.login, candidate.endpoint
            ):
                server = candidate
            elif candidate_job_id == server_job_id:
                _write_remote_json(target.login, registry_path, asdict(server))
        else:
            _write_remote_json(target.login, registry_path, asdict(server))
    finally:
        _release_remote_lock(target.login, lock_path)

    return _record_server(server)


def _wait_for_remote_file(
    login: str,
    path: str,
    *,
    timeout_s: int = 300,
    reporter: ProgressReporter | None = None,
    host_label: str | None = None,
) -> str:
    reporter = reporter or NullProgressReporter()
    label = host_label or "remote host"
    deadline = time.time() + timeout_s
    started = time.time()
    while time.time() < deadline:
        output = _run_ssh(login, f"test -f {path} && cat {path} || true")
        if output:
            return output.strip()
        elapsed = int(time.time() - started)
        reporter.set(3 + min(elapsed // 5, 4), f"Waiting for LSF job to start on {label}")
        time.sleep(2)
    raise RuntimeError(f"Remote file did not appear: {path}")


def _resolve_local_server(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
    reuse_key: str,
    endpoint: str,
    target: TargetKind,
    executable: str,
    command: str,
    warmup_command: str | None = None,
) -> ServerHandle:
    return resolve_local_server_impl(
        spec,
        state=state,
        state_path=state_path,
        reuse_key=reuse_key,
        endpoint=endpoint,
        target=target,
        executable=executable,
        command=command,
        endpoint_is_healthy=_endpoint_is_healthy,
        wait_for_endpoint=_wait_for_endpoint,
        load_state=load_state,
        merge_server=merge_server,
        update_state=update_state,
        which=shutil.which,
        pid_is_alive=_pid_is_alive,
        sleep=time.sleep,
        now=time.time,
        warmup_command=warmup_command,
    )


def _sync_bluevela_workspace(
    target: BlueVelaTargetSpec,
    repo_root: Path,
    plan,
    *,
    reporter: ProgressReporter | None = None,
) -> None:
    reporter = reporter or NullProgressReporter()
    lock_path = build_bluevela_lock_path(target, kind="workspace", key=plan.signature)
    reporter.set(25, "Acquiring remote workspace lock")
    _acquire_remote_lock(target.login, lock_path)
    try:
        reporter.set(35, "Checking remote workspace manifest")
        manifest = _read_remote_workspace_manifest(target, plan.remote_path)
        if manifest and manifest.get("signature") == plan.signature:
            reporter.set(90, "Remote workspace already current")
            return

        reporter.set(45, "Preparing remote workspace")
        _run_ssh(target.login, build_remote_workspace_prepare_command(target, plan))
        remote_path = f"{target.login}:{plan.remote_path}/"
        reporter.set(55, "Cleaning remote workspace")
        _run_ssh(
            target.login,
            (
                f"find {plan.remote_path} -mindepth 1 -maxdepth 1 "
                "! -name '.mcode-launch-workspace.json' "
                "! -name '.mcode-bootstrap-key' "
                "-exec rm -rf {} +"
            ),
        )
        if plan.mode == SyncMode.WORKING_TREE:
            reporter.set(65, "Uploading tracked working tree")
            paths = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                handle.write(paths)
                manifest_path = handle.name
            try:
                subprocess.run(
                    [
                        "rsync",
                        "-az",
                        "--from0",
                        f"--files-from={manifest_path}",
                        f"{repo_root}/",
                        remote_path,
                    ],
                    check=True,
                )
            finally:
                Path(manifest_path).unlink(missing_ok=True)
        else:
            reporter.set(65, "Uploading base tree")
            archive_cmd = ["git", "archive", "--format=tar", plan.ref_sha]
            remote_cmd = [
                "ssh",
                target.login,
                f"mkdir -p {plan.remote_path} && tar -xf - -C {plan.remote_path}",
            ]
            archive = subprocess.Popen(archive_cmd, cwd=repo_root, stdout=subprocess.PIPE)
            assert archive.stdout is not None
            subprocess.run(remote_cmd, stdin=archive.stdout, check=True)
            archive.wait()
            if plan.mode == SyncMode.GIT_OVERLAY:
                patch = tracked_overlay_patch(repo_root, plan.ref_sha)
                if patch.strip():
                    reporter.set(80, "Applying tracked overlay patch")
                    subprocess.run(
                        [
                            "ssh",
                            target.login,
                            f"cd {plan.remote_path} && git apply --allow-empty --binary",
                        ],
                        input=patch,
                        text=True,
                        check=True,
                    )
        bootstrap_marker = f"{plan.remote_path}/.mcode-bootstrap-key"
        bootstrap_value = json.dumps(plan.bootstrap_key)
        mode_check = (
            f"if [ ! -f {bootstrap_marker} ] || "
            f'[ "$(cat {bootstrap_marker} 2>/dev/null)" != {bootstrap_value} ]; then '
        )
        sync_command = build_uv_sync_command(plan.bootstrap_key)
        bootstrap_cmd = (
            f"mkdir -p {plan.remote_path}; "
            f"{mode_check}"
            f"cd {plan.remote_path} && uv python pin 3.11 && "
            f"{sync_command} && "
            f"printf %s {bootstrap_value} > {bootstrap_marker}; "
            "fi"
        )
        reporter.set(85, "Bootstrapping remote environment")
        _run_ssh(target.login, bootstrap_cmd)
        reporter.set(90, "Writing workspace manifest")
        _write_remote_json(
            target.login,
            f"{plan.remote_path}/.mcode-launch-workspace.json",
            {
                "signature": plan.signature,
                "remote_path": plan.remote_path,
                "ref_sha": plan.ref_sha,
                "repo_url": plan.repo_url,
            },
        )
    finally:
        _release_remote_lock(target.login, lock_path)


def _build_bluevela_bsub_benchmark_command(
    spec: LaunchSpec,
    *,
    workspace_path: Path,
    run_dir: Path,
    shard_index: int,
    endpoint: str,
) -> str:
    assert isinstance(spec.target, BlueVelaTargetSpec)
    benchmark_cmd = build_bluevela_benchmark_command(
        spec,
        workspace_path=workspace_path,
        db_path=run_dir / f"diagnostic-shard-{shard_index}.db",
        shard_index=shard_index,
        endpoint=endpoint,
    )
    log_path = run_dir / f"benchmark-shard-{shard_index}.log"
    return (
        "bsub "
        f"-q {shlex.quote(spec.target.queue)} "
        f"-G {shlex.quote(spec.target.group)} "
        f'-J "mcode-bench-{shard_index}" '
        "-n 8 -R 'span[hosts=1]' -R 'rusage[mem=16000]' "
        f"-o {shlex.quote(str(log_path))} -e {shlex.quote(str(log_path))} "
        f"{shlex.quote(benchmark_cmd)}"
    )


def _launch_local_benchmark(
    spec: LaunchSpec,
    *,
    env: dict[str, str],
    state: LauncherState,
    state_path: Path | None,
) -> CommandResult:
    return launch_local_benchmark_impl(
        spec,
        env=env,
        state=state,
        state_path=state_path,
        merge_run=merge_run,
    )


def _build_local_benchmark_command(
    spec: LaunchSpec,
    *,
    run_dir: Path,
    shard_count: int,
    shard_index: int,
) -> str:
    from mcode.launch.local_service import build_local_benchmark_command

    return build_local_benchmark_command(
        spec,
        run_dir=run_dir,
        shard_count=shard_count,
        shard_index=shard_index,
    )


def render_result(result: CommandResult, *, json_mode: bool) -> str:
    if json_mode:
        return json.dumps(
            {"ok": result.ok, "message": result.message, "data": result.data},
            indent=2,
            sort_keys=True,
        )
    if result.data:
        return f"{result.message}\n{json.dumps(result.data, indent=2, sort_keys=True)}"
    return result.message
