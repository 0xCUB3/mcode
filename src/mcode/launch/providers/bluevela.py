from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from mcode.launch.models import BlueVelaTargetSpec, LaunchSpec
from mcode.launch.sync import SyncPlan


def build_bluevela_server_reuse_key(spec: LaunchSpec) -> str:
    target = spec.target
    assert isinstance(target, BlueVelaTargetSpec)
    parts = [
        "bluevela",
        spec.model,
        f"tp={spec.serving.tensor_parallel}",
        f"dp={spec.serving.data_parallel}",
        f"api={spec.serving.api_server_count}",
        f"port={spec.serving.port}",
        f"ctx={spec.serving.max_model_len}",
        f"mem={spec.serving.gpu_memory_utilization}",
        f"profile={spec.serving.profile.name}",
        f"image={spec.serving.image or ''}",
    ]
    return "|".join(parts)


def build_bluevela_server_registry_path(target: BlueVelaTargetSpec, *, reuse_key: str) -> str:
    digest = hashlib.sha256(reuse_key.encode()).hexdigest()[:16]
    root = target.workspace_root.rstrip("/")
    return f"{root}/state/servers/{digest}.json"


def build_bluevela_lock_path(target: BlueVelaTargetSpec, *, kind: str, key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{target.workspace_root.rstrip('/')}/locks/{kind}-{digest}.lock"


def _bluevela_podman_roots(target: BlueVelaTargetSpec) -> tuple[str, str]:
    graphroot = target.podman_graphroot or f"{target.shared_root.rstrip('/')}/podman/graphroot"
    runroot = target.podman_runroot or f"{target.shared_root.rstrip('/')}/podman/runroot"
    return graphroot, runroot


def _bluevela_podman_base_dirs(target: BlueVelaTargetSpec) -> tuple[str, str]:
    graphroot, runroot = _bluevela_podman_roots(target)
    if graphroot.endswith("/graphroot"):
        graphroot = graphroot[: -len("/graphroot")]
    if runroot.endswith("/runroot"):
        runroot = runroot[: -len("/runroot")]
    return graphroot.rstrip("/"), runroot.rstrip("/")


def _bluevela_shared_image_cache_dir(target: BlueVelaTargetSpec) -> str:
    return f"{target.shared_root.rstrip('/')}/podman-image-cache"


def _bluevela_benchmark_podman_root(target: BlueVelaTargetSpec) -> str:
    return f"{target.shared_root.rstrip('/')}/podman-bench"


def _bluevela_podman_cli() -> str:
    return (
        "podman --cgroup-manager=cgroupfs --storage-driver=overlay "
        '--root="$GRAPHROOT" --runroot="$RUNROOT" '
    )


def _build_hf_offline_probe(hf_cache: str, model: str) -> str:
    if "/" not in model or model.startswith("/"):
        return ""
    owner, name = model.split("/", 1)
    cache_dir = f"{hf_cache}/hub/models--{owner}--{name}"
    return (
        f"if [ -d {shlex.quote(cache_dir)} ]; then "
        "export HF_HUB_OFFLINE=1; "
        "export TRANSFORMERS_OFFLINE=1; "
        "fi; "
    )


def _default_bluevela_image(spec: LaunchSpec) -> str:
    if spec.serving.profile.name == "gemma4":
        return "docker.io/vllm/vllm-openai:gemma4"
    return "docker.io/vllm/vllm-openai:v0.17.0"


def _quote_bsub_shell(script: str) -> str:
    return shlex.quote(f"bash -lc {shlex.quote(script)}")


def build_bluevela_vllm_command(spec: LaunchSpec, *, run_dir: Path) -> str:
    target = spec.target
    assert isinstance(target, BlueVelaTargetSpec)
    image = spec.serving.image or _default_bluevela_image(spec)
    flags = " ".join(shlex.quote(flag) for flag in spec.serving.profile.flags)
    log_path = run_dir / "vllm.log"
    host_file = run_dir / "vllm_host.txt"
    hf_cache = f"{target.shared_root.rstrip('/')}/hf-cache"
    offline_probe = _build_hf_offline_probe(hf_cache, spec.model)
    container_hf_home = "/root/.cache/huggingface"
    container_hf_hub_cache = f"{container_hf_home}/hub"
    graphroot_base, runroot_base = _bluevela_podman_base_dirs(target)
    image_cache_dir = _bluevela_shared_image_cache_dir(target)
    image_archive = f"{image_cache_dir}/{hashlib.sha256(image.encode()).hexdigest()[:16]}.tar"
    podman = _bluevela_podman_cli()
    script = (
        f"mkdir -p {shlex.quote(str(run_dir))}; "
        f"hostname > {shlex.quote(str(host_file))}; "
        f"GRAPHROOT_BASE={shlex.quote(graphroot_base)}; "
        f"RUNROOT_BASE={shlex.quote(runroot_base)}; "
        "HOST_TAG=$(hostname -s); "
        'GRAPHROOT="${GRAPHROOT_BASE}/${HOST_TAG}/graphroot"; '
        'RUNROOT="${RUNROOT_BASE}/${HOST_TAG}/runroot"; '
        f'mkdir -p "$GRAPHROOT" "$RUNROOT" {shlex.quote(hf_cache)}; '
        f"if [ -f {shlex.quote(target.hf_env)} ]; then . {shlex.quote(target.hf_env)}; fi; "
        f"{offline_probe}"
        f"IMAGE={shlex.quote(image)}; "
        f"IMAGE_ARCHIVE={shlex.quote(image_archive)}; "
        'mkdir -p "$(dirname "$IMAGE_ARCHIVE")"; '
        "export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u); "
        "mkdir -p /tmp/podman-run-$(id -u); "
        'exec 9>"${IMAGE_ARCHIVE}.lock"; '
        "flock 9; "
        f'if {podman} image exists "$IMAGE" >/dev/null 2>&1; then :; '
        f'elif [ -f "$IMAGE_ARCHIVE" ]; then {podman} load -i "$IMAGE_ARCHIVE" >/dev/null; '
        f'else {podman} pull "$IMAGE" >/dev/null; '
        f'{podman} save -o "$IMAGE_ARCHIVE" "$IMAGE" >/dev/null; '
        "fi; "
        "flock -u 9; "
        f"{podman}run --rm --device nvidia.com/gpu=all "
        "--security-opt=label=disable --ipc=host --net=host "
        "--storage-opt ignore_chown_errors=true "
        "-e HF_HUB_OFFLINE -e TRANSFORMERS_OFFLINE "
        '-e HF_TOKEN="${HF_TOKEN:-}" '
        '-e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" '
        f'-e HF_HOME="{container_hf_home}" '
        f'-e HF_HUB_CACHE="{container_hf_hub_cache}" '
        f"-v {shlex.quote(hf_cache)}:{container_hf_home} "
        '"$IMAGE" '
        f"--model {shlex.quote(spec.model)} "
        f"--port {spec.serving.port} "
        f"--max-model-len {spec.serving.max_model_len} "
        f"--gpu-memory-utilization {spec.serving.gpu_memory_utilization} "
        f"--tensor-parallel-size {spec.serving.tensor_parallel} "
        f"{flags}"
    )
    return (
        "bsub "
        f"-q {shlex.quote(target.queue)} "
        f"-G {shlex.quote(target.group)} "
        '-J "mcode-vllm" '
        f'-gpu "num={spec.serving.tensor_parallel}:mode=exclusive_process" '
        "-n 1 -R 'span[hosts=1]' "
        f"-o {shlex.quote(str(log_path))} -e {shlex.quote(str(log_path))} "
        f"{_quote_bsub_shell(script)}"
    )


def build_bluevela_benchmark_command(
    spec: LaunchSpec,
    *,
    workspace_path: Path,
    db_path: Path,
    shard_index: int,
    endpoint: str,
) -> str:
    target = spec.target
    assert isinstance(target, BlueVelaTargetSpec)
    task_ids = (
        f"--task-ids {shlex.quote(spec.benchmark.task_ids)}" if spec.benchmark.task_ids else ""
    )
    dataset = (
        f"--dataset {shlex.quote(spec.benchmark.dataset or 'SWE-bench/SWE-bench_Lite')}"
        if spec.benchmark.benchmark == "swebench-lite"
        else ""
    )
    limit = f"--limit {spec.benchmark.limit}" if spec.benchmark.limit is not None else ""
    benchmark_podman_root = _bluevela_benchmark_podman_root(target)
    hf_cache = f"{target.shared_root.rstrip('/')}/hf-cache"
    podman = _bluevela_podman_cli()
    return (
        f"cd {shlex.quote(str(workspace_path))}; "
        f"PODMAN_ROOT_BASE={shlex.quote(benchmark_podman_root)}; "
        'JOB_KEY="${LSB_JOBID:-0}"; '
        'GRAPHROOT="${PODMAN_ROOT_BASE}/${JOB_KEY}/graphroot"; '
        'RUNROOT="${PODMAN_ROOT_BASE}/${JOB_KEY}/runroot"; '
        f'mkdir -p "$GRAPHROOT" "$RUNROOT" {shlex.quote(hf_cache)}; '
        f"if [ -f {shlex.quote(target.hf_env)} ]; then . {shlex.quote(target.hf_env)}; fi; "
        "export XDG_RUNTIME_DIR=/tmp/podman-$(id -u)-swb-${JOB_KEY}; "
        "mkdir -p ${XDG_RUNTIME_DIR}; "
        "SOCK=${XDG_RUNTIME_DIR}/podman.sock; "
        "rm -f ${SOCK}; "
        'rm -rf "$RUNROOT/networks/rootless-netns"; '
        f"{podman}rm -af >/dev/null 2>&1 || true; "
        f"{podman}system service --time=0 unix://${{SOCK}} & "
        "PODMAN_PID=$!; "
        'trap "kill ${PODMAN_PID} 2>/dev/null; wait ${PODMAN_PID} 2>/dev/null" EXIT; '
        'export DOCKER_HOST="unix://${SOCK}"; '
        "for attempt in $(seq 1 30); do "
        'if uv run python -c "import docker; client = docker.from_env(); client.ping()" '
        ">/dev/null 2>&1; then "
        "break; "
        "fi; "
        "sleep 1; "
        "if [ ${attempt} -eq 30 ]; then echo Docker socket did not become ready >&2; exit 1; fi; "
        "done; "
        f"export OPENAI_BASE_URL={shlex.quote(endpoint)}; "
        "export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}; "
        f"export HF_HOME={shlex.quote(hf_cache)}; "
        f"export HF_HUB_CACHE={shlex.quote(hf_cache + '/hub')}; "
        f"export HF_DATASETS_CACHE={shlex.quote(hf_cache + '/datasets')}; "
        f"uv run mcode bench {shlex.quote(spec.benchmark.benchmark)} "
        f"--model {shlex.quote(spec.model)} "
        f"--backend {shlex.quote(spec.benchmark.backend)} "
        f"--loop-budget {spec.benchmark.loop_budget} "
        f"--timeout {spec.benchmark.timeout} "
        f"--split {shlex.quote(spec.benchmark.split)} "
        f"--mem-limit {shlex.quote(spec.benchmark.mem_limit)} "
        f"--pids-limit {spec.benchmark.pids_limit} "
        f"--shard-count {spec.benchmark.parallelism} "
        f"--shard-index {shard_index} "
        f"--n-samples {spec.benchmark.n_samples} "
        f"--db {shlex.quote(str(db_path))} "
        f"{task_ids} {dataset} {limit}"
    )


def build_remote_workspace_prepare_command(target: BlueVelaTargetSpec, plan: SyncPlan) -> str:
    root = target.workspace_root.rstrip("/")
    return (
        f"mkdir -p {shlex.quote(root)}/workspaces "
        f"{shlex.quote(root)}/runs "
        f"{shlex.quote(root)}/locks "
        f"{shlex.quote(root)}/state/servers "
        f"{shlex.quote(plan.remote_path)}"
    )
