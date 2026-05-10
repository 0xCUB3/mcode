"""Remote Blue Vela benchmark script construction."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from mcode.launch.config import BluevelaConfig

_SHARDED_INFRA_EXIT_CODE = 86


@dataclass(frozen=True)
class RemoteArtifactFetch:
    remote_dir: str
    local_dir: Path


@dataclass(frozen=True)
class RemoteBenchPlan:
    argv: list[str]
    remote_dir: str
    remote_db: str
    remote_logs_dir: str
    remote_log: str
    exit_sentinel: str
    remote_script_path: str
    podman_svc_log: str
    script: str
    local_artifact_fetch: RemoteArtifactFetch | None = None


def build_remote_bench_plan(
    *,
    bench_argv: list[str],
    bv: BluevelaConfig,
    run_id: str,
    attempt_token: str,
    endpoint: str,
    api_key: str,
    forwarded_env: dict[str, str],
) -> RemoteBenchPlan:
    remote_dir = f"{bv.workspace_root}/bench-runs/{run_id}"
    remote_db = f"{remote_dir}/results.db"
    remote_logs_dir = f"{remote_dir}/logs"
    remote_log = f"{remote_logs_dir}/bench-{attempt_token}.log"
    exit_sentinel = f"{remote_dir}/exit-{attempt_token}.code"
    remote_script_path = f"{remote_dir}/bench-{attempt_token}.sh"
    podman_svc_log = f"{remote_logs_dir}/podman-svc-{attempt_token}.log"

    argv = [*bench_argv]
    local_artifact_fetch = _resolve_remote_artifact_dir(
        workspace_root=bv.workspace_root,
        argv=argv,
    )
    _replace_or_append_option(argv, "--db", remote_db)

    remote_benchmark_setup = _prepare_remote_benchmark_root(
        argv,
        workspace_root=bv.workspace_root,
        shared_root=bv.shared_root,
    )
    bench_cmd = "uv run mcode bench " + " ".join(shlex.quote(a) for a in argv)
    forwarded_exports = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in forwarded_env.items()
    )

    runtime_dir = f"{bv.shared_root}/podman-runtime/{run_id}"
    tmp_dir = f"{bv.shared_root}/podman-tmp/{run_id}"
    graphroot_base = bv.podman.graphroot_base or f"{bv.shared_root}/podman-graphroot"
    runroot_base = bv.podman.runroot_base or f"{bv.shared_root}/podman-runroot"
    shared_auth = f"{bv.shared_root}/containers-auth.json"

    script = f"""
set -euo pipefail
if [ -z "${{LSB_JOBID:-}}" ]; then
  echo "refusing to start podman outside an LSF compute job" >&2
  exit 98
fi
cd {shlex.quote(bv.workspace_root)}
[ -f {shlex.quote(bv.hf_env)} ] && source {shlex.quote(bv.hf_env)}
export XDG_RUNTIME_DIR={shlex.quote(runtime_dir)}
export WORKSPACE_TMP={shlex.quote(tmp_dir)}
export TMPDIR="$WORKSPACE_TMP"
GRAPHROOT_BASE={shlex.quote(graphroot_base)}
RUNROOT_BASE={shlex.quote(runroot_base)}
LOCKROOT_BASE="$GRAPHROOT_BASE/locks"
HOST_TAG="$(hostname -s)"
GRAPHROOT="$GRAPHROOT_BASE/$HOST_TAG"
RUNROOT="$RUNROOT_BASE/$HOST_TAG"
CONTAINERS_CONF="$XDG_RUNTIME_DIR/containers.conf"
SOCK="$XDG_RUNTIME_DIR/podman.sock"
export MCODE_PODMAN_LOCK_DIR="$LOCKROOT_BASE/$HOST_TAG"
# Use shared docker.io creds if present so we get the higher pull rate limit
# (~200/6h vs ~100/6h for anonymous on the cluster's shared egress IP). The
# launcher never reads login-node home paths during remote bench startup.
if [ -f {shlex.quote(shared_auth)} ]; then
  export REGISTRY_AUTH_FILE={shlex.quote(shared_auth)}
fi

wait_for_pid_exit() {{
  local pid="$1"
  local timeout_s="$2"
  for _ in $(seq 1 "$timeout_s"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}}

run_with_timeout() {{
  local timeout_s="$1"
  shift
  "$@" &
  local cmd_pid=$!
  if wait_for_pid_exit "$cmd_pid" "$timeout_s"; then
    wait "$cmd_pid" 2>/dev/null
    return $?
  fi
  kill "$cmd_pid" 2>/dev/null || true
  if ! wait_for_pid_exit "$cmd_pid" 5; then
    kill -KILL "$cmd_pid" 2>/dev/null || true
    wait_for_pid_exit "$cmd_pid" 1 || true
  fi
  if ! kill -0 "$cmd_pid" 2>/dev/null; then
    wait "$cmd_pid" 2>/dev/null || true
  fi
  return 124
}}

cleanup_dir() {{
  local target="$1"
  local label="$2"
  local cleanup_target
  if [ ! -e "$target" ]; then
    return 0
  fi
  cleanup_target="$target.stale.${{LSB_JOBID:-0}}.$$.$(date +%s)"
  if ! run_with_timeout 5 mv "$target" "$cleanup_target"; then
    echo "$label rename did not finish, cleaning in place" >&2
    cleanup_target="$target"
  fi
  if run_with_timeout 20 rm -rf "$cleanup_target"; then
    return 0
  fi
  echo "plain $label cleanup did not finish, trying podman unshare" >&2
  if run_with_timeout 20 podman unshare rm -rf "$cleanup_target"; then
    return 0
  fi
  echo "$label cleanup still not finished for $cleanup_target" >&2
}}

cleanup_runtime_dir() {{
  cleanup_dir "$XDG_RUNTIME_DIR" "runtime"
}}

prepare_runtime() {{
  mkdir -p "$XDG_RUNTIME_DIR" "$WORKSPACE_TMP" "$GRAPHROOT" "$RUNROOT" "$MCODE_PODMAN_LOCK_DIR"
  printf '[containers]\nkeyring=false\n' > "$CONTAINERS_CONF"
  export CONTAINERS_CONF
  rm -rf "$WORKSPACE_TMP"/* 2>/dev/null || true
  rm -f "$SOCK"
}}

stop_podman() {{
  if [ -n "${{PODMAN_PID:-}}" ]; then
    local pid="$PODMAN_PID"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      if ! wait_for_pid_exit "$pid" 15; then
        echo "podman service did not stop after TERM, sending KILL" >&2
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        if ! wait_for_pid_exit "$pid" 5; then
          echo "podman service still alive after KILL" >&2
        fi
      fi
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
    fi
    PODMAN_PID=""
  fi
}}

reset_podman_runtime() {{
  stop_podman
  cleanup_runtime_dir
  prepare_runtime
}}

reset_persistent_podman_store() {{
  stop_podman
  cleanup_dir "$GRAPHROOT" "graphroot"
  cleanup_dir "$RUNROOT" "runroot"
  prepare_runtime
}}

start_podman() {{
  prepare_runtime
  setsid podman \
    --cgroup-manager=cgroupfs --storage-driver=overlay \
    --root "$GRAPHROOT" --runroot "$RUNROOT" \
    --storage-opt ignore_chown_errors=true \
    system service --time=0 "unix://$SOCK" \
    >{shlex.quote(podman_svc_log)} 2>&1 &
  PODMAN_PID=$!
  svc_ready=0
  for _ in $(seq 1 30); do
    if curl -s --unix-socket "$SOCK" http://localhost/version >/dev/null 2>&1; then
      svc_ready=1
      break
    fi
    sleep 1
  done
  if [ "$svc_ready" != "1" ]; then
    echo "✗ podman socket did not come up at $SOCK" >&2
    tail -n 40 {shlex.quote(podman_svc_log)} >&2 || true
    if grep -q "database configuration mismatch" {shlex.quote(podman_svc_log)}; then
      echo "podman store mismatch under $GRAPHROOT, clearing persistent store once" >&2
      return 86
    fi
    return 97
  fi
}}

cleanup() {{
  stop_podman
  cleanup_runtime_dir
}}
trap cleanup EXIT

set +e
start_podman
rc=$?
set -e
if [ "$rc" = "86" ]; then
  reset_persistent_podman_store
  set +e
  start_podman
  rc=$?
  set -e
fi
if [ "$rc" != "0" ]; then
  echo "$rc" > {shlex.quote(exit_sentinel)}
  exit "$rc"
fi
export DOCKER_HOST="unix://$SOCK"
echo "podman storage host=$HOST_TAG graphroot=$GRAPHROOT runroot=$RUNROOT"
export OPENAI_BASE_URL={shlex.quote(endpoint)}
export OPENAI_API_KEY={shlex.quote(api_key)}
{forwarded_exports}\
{remote_benchmark_setup}
infra_retries=0
max_infra_retries=1
while true; do
  set +e
  {bench_cmd}
  rc=$?
  set -e
  if [ "$rc" = "{_SHARDED_INFRA_EXIT_CODE}" ] && [ "$infra_retries" -lt "$max_infra_retries" ]; then
    infra_retries=$((infra_retries + 1))
    echo "retryable podman infra failure, resetting runtime ($infra_retries/$max_infra_retries)" >&2
    reset_podman_runtime
    start_podman || {{ rc=97; break; }}
    export DOCKER_HOST="unix://$SOCK"
    continue
  fi
  break
done
echo "$rc" > {shlex.quote(exit_sentinel)}
exit $rc
""".strip()

    return RemoteBenchPlan(
        argv=argv,
        remote_dir=remote_dir,
        remote_db=remote_db,
        remote_logs_dir=remote_logs_dir,
        remote_log=remote_log,
        exit_sentinel=exit_sentinel,
        remote_script_path=remote_script_path,
        podman_svc_log=podman_svc_log,
        script=script,
        local_artifact_fetch=local_artifact_fetch,
    )


def build_lsf_submit_command(*, plan: RemoteBenchPlan, bv: BluevelaConfig, queue: str) -> str:
    run_id = plan.remote_dir.rsplit("/", 1)[-1]
    job_name = shlex.quote(f"mcode-bench-{run_id[:40]}")
    return (
        f"bsub -G {shlex.quote(bv.group)} -q {shlex.quote(queue)} "
        f"-J {job_name} "
        f"-n 8 -R {shlex.quote('span[hosts=1]')} "
        f"-R {shlex.quote('rusage[mem=16000]')} "
        f"-o {shlex.quote(plan.remote_log)} -e {shlex.quote(plan.remote_log)} "
        f"bash {shlex.quote(plan.remote_script_path)}"
    )


def _argv_option_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _resolve_remote_artifact_dir(
    *,
    workspace_root: str,
    argv: list[str],
) -> RemoteArtifactFetch | None:
    raw = _argv_option_value(argv, "--artifact-dir")
    if not raw:
        return None
    local_path = Path(raw)
    if local_path.is_absolute():
        remote_path = raw
    else:
        remote_path = f"{workspace_root}/{raw}"
    return RemoteArtifactFetch(remote_dir=remote_path, local_dir=local_path)


def _replace_or_append_option(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        i = argv.index(flag)
        argv[i + 1] = value
        return
    argv.extend([flag, value])


def _prepare_remote_benchmark_root(
    argv: list[str],
    *,
    workspace_root: str,
    shared_root: str,
) -> str:
    if not argv or argv[0] not in {"aider-polyglot", "suite"}:
        return ""

    remote_root = f"{workspace_root}/benchmarks/polyglot-benchmark"
    toolchain_root = f"{shared_root}/toolchains/aider-polyglot"
    if argv[0] == "aider-polyglot":
        _replace_or_append_option(argv, "--benchmark-root", remote_root)
    root_q = shlex.quote(remote_root)
    parent_q = shlex.quote(str(Path(remote_root).parent))
    toolchain_q = shlex.quote(toolchain_root)
    return f"""
mkdir -p {parent_q}
(
  flock 9
  if [ -d {root_q}/.git ]; then
    git -C {root_q} fetch --depth=1 origin main
    git -C {root_q} reset --hard origin/main
  elif [ -e {root_q} ]; then
    echo "remote benchmark root exists but is not a git checkout: {remote_root}" >&2
    exit 96
  else
    git clone --depth=1 https://github.com/Aider-AI/polyglot-benchmark.git {root_q}
  fi
) 9>{parent_q}/.polyglot-benchmark.lock
TOOLCHAIN_ROOT={toolchain_q}
if [ -d "$TOOLCHAIN_ROOT" ]; then
  export GOROOT="$TOOLCHAIN_ROOT/go"
  export JAVA_HOME="$TOOLCHAIN_ROOT/jdk"
  export CARGO_HOME="$TOOLCHAIN_ROOT/cargo"
  export RUSTUP_HOME="$TOOLCHAIN_ROOT/rustup"
  export PATH="$TOOLCHAIN_ROOT/go/bin:$TOOLCHAIN_ROOT/node/bin:$PATH"
  export PATH="$TOOLCHAIN_ROOT/cmake/bin:$TOOLCHAIN_ROOT/jdk/bin:$PATH"
  export PATH="$TOOLCHAIN_ROOT/cargo/bin:$PATH"
  export npm_config_cache="$TOOLCHAIN_ROOT/npm-cache"
fi
""".strip()


__all__ = [
    "RemoteArtifactFetch",
    "RemoteBenchPlan",
    "build_lsf_submit_command",
    "build_remote_bench_plan",
]
