from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcode.launch.config import LaunchConfig
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
from mcode.launch.providers.bluevela import (
    _bluevela_podman_base_dirs,
    build_bluevela_benchmark_command,
    build_bluevela_lock_path,
    build_bluevela_server_registry_path,
    build_bluevela_server_reuse_key,
    build_bluevela_vllm_command,
    build_remote_workspace_prepare_command,
)
from mcode.launch.providers.local_ollama import (
    build_ollama_serve_command,
    build_ollama_warmup_command,
)
from mcode.launch.providers.local_vllm import (
    build_local_vllm_command,
    build_local_vllm_reuse_key,
)
from mcode.launch.remote_scripts import (
    build_remote_healthcheck_command,
    build_uv_sync_command,
)
from mcode.launch.state import LauncherState, load_state, save_state
from mcode.launch.sync import build_sync_plan, list_untracked_files, tracked_overlay_patch

BLUEVELA_VLLM_FAILED_MARKERS = (
    "Exited with exit code",
    "Failed to obtain podman configuration",
    "Engine core initialization failed",
    "No available memory for the cache blocks",
    "cannot re-exec process to join the existing user namespace",
)


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
    detach: bool,
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
    bench = BenchSpec(
        benchmark=benchmark,
        backend=backend or ("ollama" if kind == TargetKind.LOCAL_OLLAMA else "openai"),
        split=resolved_split,
        loop_budget=loop_budget,
        timeout=timeout,
        parallelism=parallelism,
        limit=limit,
        task_ids=task_ids,
    )
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
        detach=detach,
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
        detach=False,
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
    spec: LaunchSpec, *, repo_root: Path, state_path: Path | None = None
) -> CommandResult:
    if not isinstance(spec.target, BlueVelaTargetSpec):
        return CommandResult(ok=True, message="Sync is only required for Blue Vela in V1.")

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
        return CommandResult(
            ok=False,
            message=(
                "Untracked files are excluded from sync. Stage them with git add before using "
                "git-overlay or git-ref sync."
            ),
            data=data,
        )
    if spec.sync.check:
        manifest = _read_remote_workspace_manifest(spec.target, plan.remote_path)
        data["remote_manifest"] = manifest
        data["is_noop"] = bool(manifest and manifest.get("signature") == plan.signature)
        return CommandResult(ok=True, message="Sync plan computed.", data=data)

    _sync_bluevela_workspace(spec.target, repo_root, plan)
    workspace = WorkspaceHandle(signature=plan.signature, path=plan.remote_path, metadata=data)
    state.workspaces = [
        entry for entry in state.workspaces if entry.signature != workspace.signature
    ] + [workspace]
    save_state(state_path, state)
    return CommandResult(ok=True, message=plan.remote_path, data=data)


def launch_status(*, state_path: Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    return {
        "servers": [asdict(server) for server in state.servers],
        "runs": [asdict(run) for run in state.runs],
        "workspaces": [asdict(workspace) for workspace in state.workspaces],
    }


def launch_fetch(
    run_id: str, *, destination: Path, state_path: Path | None = None
) -> CommandResult:
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        return CommandResult(ok=False, message=f"Unknown run id: {run_id}")
    if run.target != TargetKind.BLUEVELA.value:
        return CommandResult(ok=False, message="Fetch is only implemented for Blue Vela runs.")
    remote_path = run.metadata.get("run_dir")
    if not remote_path:
        return CommandResult(ok=False, message="Run has no remote directory recorded.")
    destination.mkdir(parents=True, exist_ok=True)
    login = run.metadata.get("login")
    cmd = ["rsync", "-az", f"{login}:{remote_path}/", str(destination)]
    subprocess.run(cmd, check=True)
    return CommandResult(ok=True, message=f"Fetched {run_id} into {destination}")


def launch_stop(run_id: str, *, state_path: Path | None = None) -> CommandResult:
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        server = next((entry for entry in state.servers if entry.id == run_id), None)
        if server is None:
            return CommandResult(ok=False, message=f"Unknown id: {run_id}")
        return _stop_server(server, state, state_path)
    return _stop_run(run, state, state_path)


def launch_attach(run_id: str, *, state_path: Path | None = None) -> CommandResult:
    state = load_state(state_path)
    run = next((entry for entry in state.runs if entry.id == run_id), None)
    if run is None:
        return CommandResult(ok=False, message=f"Unknown run id: {run_id}")
    return CommandResult(
        ok=True, message=run.log_path or "No log path recorded.", data=run.metadata
    )


def launch_run(
    spec: LaunchSpec, *, repo_root: Path, state_path: Path | None = None
) -> CommandResult:
    state = load_state(state_path)
    if isinstance(spec.target, BlueVelaTargetSpec):
        return _launch_bluevela(spec, repo_root=repo_root, state=state, state_path=state_path)
    if isinstance(spec.target, LocalVllmTargetSpec):
        return _launch_local_vllm(spec, state=state, state_path=state_path)
    if isinstance(spec.target, LocalOllamaTargetSpec):
        return _launch_local_ollama(spec, state=state, state_path=state_path)
    return _launch_openai_compatible(spec, state=state, state_path=state_path)


def _launch_bluevela(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    state: LauncherState,
    state_path: Path | None,
) -> CommandResult:
    if spec.yes:
        _resolve_bluevela_podman_storage(spec.target)
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
    reuse_key = build_bluevela_server_reuse_key(spec, workspace_signature=workspace_signature)
    existing_server = _find_existing_server(state, reuse_key=reuse_key)
    if spec.yes:
        server = _resolve_bluevela_server(
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
        command = build_bluevela_vllm_command(spec, run_dir=run_dir)
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
        _run_ssh(spec.target.login, f"mkdir -p {run_dir}")
    commands = [
        _build_bluevela_bsub_benchmark_command(
            spec,
            workspace_path=Path(workspace.path),
            run_dir=run_dir,
            shard_index=shard_index,
            endpoint=server.endpoint,
        )
        for shard_index in range(spec.benchmark.parallelism)
    ]
    job_ids = [
        _run_ssh(spec.target.login, command).strip() if spec.yes else command
        for command in commands
    ]
    run = RunHandle(
        id=run_id,
        target=TargetKind.BLUEVELA.value,
        benchmark=spec.benchmark.benchmark,
        status="running" if spec.yes else "planned",
        metadata={
            "job_ids": job_ids,
            "login": spec.target.login,
            "run_dir": str(run_dir),
            "workspace_signature": workspace_signature,
        },
        log_path=str(run_dir / "benchmark.log"),
    )
    state.runs.append(run)
    save_state(state_path, state)
    return CommandResult(
        ok=True,
        message="\n".join(job_ids) if spec.yes else "\n".join(commands),
        data={"run_id": run_id, "server_id": server.id, "workspace_signature": workspace_signature},
    )


def _launch_local_vllm(
    spec: LaunchSpec, *, state: LauncherState, state_path: Path | None
) -> CommandResult:
    reuse_key = build_local_vllm_reuse_key(spec)
    endpoint = f"http://127.0.0.1:{spec.serving.port}/v1"
    existing = next(
        (
            server
            for server in state.servers
            if server.reuse_key == reuse_key and server.status == "healthy"
        ),
        None,
    )
    if existing and spec.reuse == ReuseMode.PREFER and _endpoint_is_healthy(existing.endpoint):
        server = existing
    elif spec.reuse == ReuseMode.PREFER and _endpoint_is_healthy(endpoint):
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=TargetKind.LOCAL_VLLM.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status="healthy",
            metadata={"discovered": True},
            log_path=None,
        )
        state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
            server
        ]
        save_state(state_path, state)
    else:
        command = build_local_vllm_command(spec)
        log_path = str(Path.cwd() / f".mcode-vllm-{spec.serving.port}.log")
        if spec.yes:
            with open(log_path, "a", encoding="utf-8") as handle:
                process = subprocess.Popen(command, shell=True, stdout=handle, stderr=handle)
            _wait_for_endpoint(endpoint)
            metadata = {"pid": process.pid}
            status = "healthy"
        else:
            metadata = {"command": command}
            status = "planned"
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=TargetKind.LOCAL_VLLM.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status=status,
            metadata=metadata,
            log_path=log_path,
        )
        state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
            server
        ]
        save_state(state_path, state)
    run = _launch_local_benchmark(
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


def _launch_local_ollama(
    spec: LaunchSpec, *, state: LauncherState, state_path: Path | None
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
    existing = next(
        (
            server
            for server in state.servers
            if server.reuse_key == reuse_key and server.status == "healthy"
        ),
        None,
    )
    if existing and spec.reuse == ReuseMode.PREFER and _endpoint_is_healthy(existing.endpoint):
        server = existing
    elif spec.reuse == ReuseMode.PREFER and _endpoint_is_healthy(endpoint):
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=TargetKind.LOCAL_OLLAMA.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status="healthy",
            metadata={"discovered": True},
            log_path=None,
        )
        state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
            server
        ]
        save_state(state_path, state)
    else:
        serve_cmd = build_ollama_serve_command(spec)
        warm_cmd = build_ollama_warmup_command(spec)
        log_path = str(Path.cwd() / f".mcode-ollama-{spec.serving.port}.log")
        if spec.yes and shutil.which("ollama"):
            with open(log_path, "a", encoding="utf-8") as handle:
                process = subprocess.Popen(serve_cmd, shell=True, stdout=handle, stderr=handle)
            subprocess.run(warm_cmd, shell=True, check=False)
            _wait_for_endpoint(endpoint)
            metadata = {"pid": process.pid, "warmup": warm_cmd}
            status = "healthy"
        else:
            metadata = {"command": serve_cmd, "warmup": warm_cmd}
            status = "planned"
        server = ServerHandle(
            id=f"server-{uuid.uuid4().hex[:8]}",
            target=TargetKind.LOCAL_OLLAMA.value,
            reuse_key=reuse_key,
            endpoint=endpoint,
            status=status,
            metadata=metadata,
            log_path=log_path,
        )
        state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
            server
        ]
        save_state(state_path, state)
    run = _launch_local_benchmark(spec, env={}, state=state, state_path=state_path)
    run.data["server_id"] = server.id
    return run


def _launch_openai_compatible(
    spec: LaunchSpec,
    *,
    state: LauncherState,
    state_path: Path | None,
) -> CommandResult:
    env = {
        "OPENAI_BASE_URL": spec.target.base_url,
        "OPENAI_API_KEY": os.environ.get(
            spec.target.api_key_env, os.environ.get("OPENAI_API_KEY", "dummy")
        ),
    }
    return _launch_local_benchmark(spec, env=env, state=state, state_path=state_path)


def _stop_run(run: RunHandle, state: LauncherState, state_path: Path | None) -> CommandResult:
    job_id = run.metadata.get("job_id")
    if run.target == TargetKind.BLUEVELA.value and run.metadata.get("job_ids"):
        for current in run.metadata["job_ids"]:
            _run_ssh(run.metadata["login"], f"bkill {current}")
    elif run.target == TargetKind.BLUEVELA.value and job_id:
        _run_ssh(run.metadata["login"], f"bkill {job_id}")
    elif "pid" in run.metadata:
        os.kill(int(run.metadata["pid"]), 15)
    run.status = "stopped"
    save_state(state_path, state)
    return CommandResult(ok=True, message=f"Stopped {run.id}")


def _stop_server(
    server: ServerHandle, state: LauncherState, state_path: Path | None
) -> CommandResult:
    job_id = server.metadata.get("job_id")
    if server.target == TargetKind.BLUEVELA.value and job_id:
        _run_ssh(server.metadata["login"], f"bkill {job_id}")
        registry_path = server.metadata.get("registry_path")
        if registry_path:
            _run_ssh(server.metadata["login"], f"rm -f {shlex.quote(registry_path)}")
    elif "pid" in server.metadata:
        os.kill(int(server.metadata["pid"]), 15)
    server.status = "stopped"
    save_state(state_path, state)
    return CommandResult(ok=True, message=f"Stopped {server.id}")


def _run_ssh_result(
    login: str, command: str, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    remote = f"bash -lc {shlex.quote(command)}"
    return subprocess.run(
        ["ssh", login, remote],
        check=check,
        text=True,
        capture_output=True,
    )


def _run_ssh(login: str, command: str) -> str:
    res = _run_ssh_result(login, command)
    return res.stdout.strip() or res.stderr.strip()


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
    timeout_s: int = 300,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _remote_endpoint_is_healthy(login, base_url):
            return
        log_tail = _run_ssh(login, f"test -f {log_path} && tail -n 20 {log_path} || true")
        failed = any(marker in log_tail for marker in BLUEVELA_VLLM_FAILED_MARKERS)
        if failed:
            raise RuntimeError(
                f"Blue Vela vLLM job failed before endpoint was healthy:\n{log_tail}"
            )
        time.sleep(2)
    raise RuntimeError(f"Endpoint did not become healthy: {base_url}")


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
    stale_after_s: int = 1800,
) -> None:
    deadline = time.time() + timeout_s
    parent_path = str(Path(lock_path).parent)
    created_path = f"{lock_path}/created_at"
    while time.time() < deadline:
        res = _run_ssh_result(
            login,
            (
                "bash -lc '"
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
                "exit 1'"
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
    state: LauncherState,
    state_path: Path | None,
    reuse_key: str,
    workspace_signature: str,
    existing_server: ServerHandle | None,
) -> ServerHandle:
    assert isinstance(spec.target, BlueVelaTargetSpec)
    target = spec.target
    registry_path = build_bluevela_server_registry_path(target, reuse_key=reuse_key)
    lock_path = build_bluevela_lock_path(target, kind="server", key=reuse_key)
    _acquire_remote_lock(target.login, lock_path)
    try:
        remote_server = _read_remote_json(target.login, registry_path)
        if remote_server and spec.reuse == ReuseMode.STOP_AND_REPLACE:
            job_id = remote_server.get("metadata", {}).get("job_id")
            if job_id:
                _run_ssh(target.login, f"bkill {job_id}")
            _run_ssh(target.login, f"rm -f {shlex.quote(registry_path)}")
            remote_server = None
        if (
            remote_server
            and spec.reuse == ReuseMode.PREFER
            and _remote_endpoint_is_healthy(target.login, remote_server.get("endpoint", ""))
        ):
            server = ServerHandle(**remote_server)
        elif existing_server and spec.reuse == ReuseMode.PREFER:
            server = existing_server
        else:
            run_dir = Path(target.workspace_root) / "runs" / uuid.uuid4().hex[:12]
            _run_ssh(target.login, f"mkdir -p {shlex.quote(str(run_dir))}")
            command = build_bluevela_vllm_command(spec, run_dir=run_dir)
            job_id = _run_ssh(target.login, command).strip()
            host = _wait_for_remote_file(target.login, str(run_dir / "vllm_host.txt"))
            endpoint = f"http://{host}:{spec.serving.port}/v1"
            _wait_for_bluevela_endpoint(
                target.login,
                endpoint,
                log_path=str(run_dir / "vllm.log"),
            )
            server = ServerHandle(
                id=f"server-{uuid.uuid4().hex[:8]}",
                target=TargetKind.BLUEVELA.value,
                reuse_key=reuse_key,
                endpoint=endpoint,
                status="healthy",
                metadata={
                    "job_id": job_id,
                    "login": target.login,
                    "run_dir": str(run_dir),
                    "registry_path": registry_path,
                    "workspace_signature": workspace_signature,
                },
                log_path=str(run_dir / "vllm.log"),
            )
            registry_root = f"{target.workspace_root.rstrip('/')}/state/servers"
            server_payload = shlex.quote(json.dumps(asdict(server)))
            registry_cmd = (
                f"mkdir -p {shlex.quote(registry_root)} && "
                f"printf %s {server_payload} > {shlex.quote(registry_path)}"
            )
            _run_ssh(
                target.login,
                registry_cmd,
            )
        state.servers = [entry for entry in state.servers if entry.reuse_key != reuse_key] + [
            server
        ]
        save_state(state_path, state)
        return server
    finally:
        _release_remote_lock(target.login, lock_path)


def _wait_for_remote_file(login: str, path: str, *, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        output = _run_ssh(login, f"test -f {path} && cat {path} || true")
        if output:
            return output.strip()
        time.sleep(2)
    raise RuntimeError(f"Remote file did not appear: {path}")


def _sync_bluevela_workspace(target: BlueVelaTargetSpec, repo_root: Path, plan) -> None:
    lock_path = build_bluevela_lock_path(target, kind="workspace", key=plan.signature)
    _acquire_remote_lock(target.login, lock_path)
    try:
        manifest = _read_remote_workspace_manifest(target, plan.remote_path)
        if manifest and manifest.get("signature") == plan.signature:
            return

        _run_ssh(target.login, build_remote_workspace_prepare_command(target, plan))
        remote_path = f"{target.login}:{plan.remote_path}/"
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
            "bash -lc '"
            f"mkdir -p {plan.remote_path}; "
            f"{mode_check}"
            f"cd {plan.remote_path} && uv python pin 3.11 && "
            f"{sync_command} && "
            f"printf %s {bootstrap_value} > {bootstrap_marker}; "
            "fi'"
        )
        _run_ssh(target.login, bootstrap_cmd)
        payload = json.dumps(
            {
                "signature": plan.signature,
                "remote_path": plan.remote_path,
                "ref_sha": plan.ref_sha,
                "repo_url": plan.repo_url,
            }
        )
        _run_ssh(
            target.login,
            (
                "bash -lc "
                f"'printf %s {json.dumps(payload)} > "
                f"{plan.remote_path}/.mcode-launch-workspace.json'"
            ),
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
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    log_path = str(Path.cwd() / f".mcode-{run_id}.log")
    command = _build_local_benchmark_command(spec, run_id=run_id)
    if spec.yes:
        merged_env = os.environ.copy()
        merged_env.update(env)
        with open(log_path, "a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, shell=True, stdout=handle, stderr=handle, env=merged_env
            )
        metadata = {"pid": process.pid, **env}
        status = "running"
    else:
        metadata = {"command": command, **env}
        status = "planned"
    run = RunHandle(
        id=run_id,
        target=spec.target.kind.value,
        benchmark=spec.benchmark.benchmark,
        status=status,
        metadata=metadata,
        log_path=log_path,
    )
    state.runs.append(run)
    save_state(state_path, state)
    return CommandResult(ok=True, message=command, data={"run_id": run.id})


def _build_local_benchmark_command(spec: LaunchSpec, *, run_id: str) -> str:
    task_ids = f"--task-ids {spec.benchmark.task_ids}" if spec.benchmark.task_ids else ""
    limit = f"--limit {spec.benchmark.limit}" if spec.benchmark.limit is not None else ""
    db_path = Path("results") / f"{run_id}.db"
    return (
        f"uv run mcode bench {spec.benchmark.benchmark} "
        f"--model {spec.model} "
        f"--backend {spec.benchmark.backend} "
        f"--loop-budget {spec.benchmark.loop_budget} "
        f"--timeout {spec.benchmark.timeout} "
        f"--split {spec.benchmark.split} "
        f"--shard-count {spec.benchmark.parallelism} "
        f"--shard-index 0 "
        f"--db {db_path} "
        f"{task_ids} {limit}"
    ).strip()


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
