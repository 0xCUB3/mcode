"""Run a bench command on Blue Vela via SSH."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from mcode.launch import config as launch_config
from mcode.launch import state as launch_state
from mcode.launch.models import Target
from mcode.launch.ssh import SshClient


class RemoteBenchError(RuntimeError):
    """User-facing remote execution error."""


_FORWARDED_ENV_VARS = (
    "MCODE_CONTEXT_WINDOW",
    "MCODE_MAX_NEW_TOKENS",
    "MCODE_HARNESS_EXPERIMENTS",
)


def _resolve_endpoint(model: str, *, cfg: launch_config.LaunchConfig) -> str:
    if endpoint := os.environ.get("OPENAI_BASE_URL"):
        return endpoint

    # Refresh Blue Vela server records first so stale "healthy" entries from
    # EXITed jobs are dropped — otherwise we pick up a dead endpoint that
    # silently fails every API call in the bench run.
    from mcode.launch import bluevela as launch_bluevela
    from mcode.launch.models import ServerRecord

    def _refresh(s: launch_state.State) -> int:
        count = 0
        for srv in list(s.servers):
            if srv.target != Target.BLUEVELA:
                continue
            try:
                updated = launch_bluevela.refresh(srv, cfg=cfg)
            except Exception:
                continue
            if isinstance(updated, ServerRecord):
                s.upsert_server(updated)
                count += 1
        return count

    try:
        launch_state.update(None, _refresh)
    except Exception:
        pass  # best-effort

    snap = launch_state.load()
    for s in snap.servers:
        if (
            s.target == Target.BLUEVELA
            and s.model == model
            and s.status == "healthy"
            and s.endpoint
        ):
            return s.endpoint
    raise RemoteBenchError(
        f"no healthy Blue Vela vLLM for {model!r} in launch state; "
        f"run `mcode launch bluevela --model {model}` first",
    )


def run_bench_on_bluevela(
    *,
    bench_argv: list[str],
    model: str,
    local_db: Path,
    fetch_db: bool = True,
) -> int:
    """Run `uv run mcode bench <bench_argv>` on Blue Vela login3.

    `bench_argv` is the full list after `mcode bench` (e.g. `["smoke",
    "--model", "X", "--db", "<remote>", ...]`). The caller is responsible for
    passing the remote DB path inside bench_argv (we compute and inject it).
    """
    cfg = launch_config.load()
    errs = launch_config.validate_for_bluevela(cfg)
    if errs:
        raise RemoteBenchError("launch config incomplete: " + "; ".join(errs))
    bv = cfg.bluevela

    endpoint = _resolve_endpoint(model, cfg=cfg)
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    ssh = SshClient(bv.login)
    run_id = f"bench-{int(time.time())}-{model.replace('/', '-')[:24]}"
    remote_dir = f"{bv.workspace_root}/bench-runs/{run_id}"
    remote_db = f"{remote_dir}/results.db"
    remote_log = f"{remote_dir}/bench.log"
    runtime_dir = f"/tmp/mcode-bench-{run_id}"
    forwarded_env = {name: value for name in _FORWARDED_ENV_VARS if (value := os.environ.get(name))}
    forwarded_exports = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in forwarded_env.items()
    )

    # Replace/append --db so the bench writes to the remote path.
    argv = [*bench_argv]
    if "--db" in argv:
        i = argv.index("--db")
        argv[i + 1] = remote_db
    else:
        argv += ["--db", remote_db]

    ssh.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=30)

    hf_env = bv.hf_env
    bench_cmd = "uv run mcode bench " + " ".join(shlex.quote(a) for a in argv)
    exit_sentinel = f"{remote_dir}/exit_code"
    svc_log = f"{remote_dir}/podman-svc.log"
    remote_script = f"""
set -euo pipefail
cd {shlex.quote(bv.workspace_root)}
[ -f {shlex.quote(hf_env)} ] && source {shlex.quote(hf_env)}
export XDG_RUNTIME_DIR={shlex.quote(runtime_dir)}
WORKSPACE_TMP="$XDG_RUNTIME_DIR/tmp"
GRAPHROOT="$XDG_RUNTIME_DIR/graphroot"
RUNROOT="$XDG_RUNTIME_DIR/runroot"
mkdir -p "$WORKSPACE_TMP" "$GRAPHROOT" "$RUNROOT"
export TMPDIR="$WORKSPACE_TMP"
SOCK="$XDG_RUNTIME_DIR/podman.sock"
rm -f "$SOCK"
cleanup() {{
  if [ -n "${{PODMAN_PID:-}}" ]; then
    kill "$PODMAN_PID" 2>/dev/null || true
    wait "$PODMAN_PID" 2>/dev/null || true
  fi
}}
trap cleanup EXIT
podman \\
  --cgroup-manager=cgroupfs --storage-driver=overlay \\
  --root "$GRAPHROOT" --runroot "$RUNROOT" \\
  --storage-opt ignore_chown_errors=true \\
  system service --time=0 "unix://$SOCK" \\
  >{shlex.quote(svc_log)} 2>&1 &
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
  tail -n 40 {shlex.quote(svc_log)} >&2 || true
  echo 97 > {shlex.quote(exit_sentinel)}
  exit 97
fi
export DOCKER_HOST="unix://$SOCK"
export OPENAI_BASE_URL={shlex.quote(endpoint)}
export OPENAI_API_KEY={shlex.quote(api_key)}
{forwarded_exports}\
set +e
{bench_cmd}
rc=$?
set -e
echo "$rc" > {shlex.quote(exit_sentinel)}
exit $rc
""".strip()

    # Use a detached run + streaming tail so we can show progress without
    # holding a single multi-hour ssh session.
    launch_cmd = (
        f"nohup bash -lc {shlex.quote(remote_script)} > {shlex.quote(remote_log)} 2>&1 & echo $!"
    )
    r = ssh.run(launch_cmd, timeout=30)
    if not r.ok:
        raise RemoteBenchError(f"failed to launch remote bench: {r.stderr.strip()}")
    pid = r.stdout.strip().splitlines()[-1]
    print(f"▶ remote bench started: pid={pid} host={bv.login} log={remote_log}")

    # Stream remote log until the pid exits.
    try:
        _stream_remote_log(ssh, remote_log, pid=pid)
    except KeyboardInterrupt:
        print("\n⚠ interrupted; remote job still running. Check with:")
        print(f"  ssh {bv.login} 'tail -f {remote_log}'")
        return 130

    # Trust the sentinel, not the DB size.
    sentinel_r = ssh.run(
        f"cat {shlex.quote(exit_sentinel)} 2>/dev/null || echo 99",
        timeout=30,
    )
    try:
        exit_code = int((sentinel_r.stdout.strip() or "99").splitlines()[-1])
    except ValueError:
        exit_code = 99
    if exit_code == 99:
        print("✗ remote bench did not write an exit sentinel (likely killed)")

    # Fetch DB whenever it exists — non-zero exit can still have useful rows
    # for debugging. Caller can inspect terminal_reason counts.
    if fetch_db:
        size_r = ssh.run(
            f"test -f {shlex.quote(remote_db)} && stat -c %s {shlex.quote(remote_db)} || echo 0",
            timeout=30,
        )
        size = int((size_r.stdout.strip() or "0").splitlines()[-1])
        if size > 0:
            local_db.parent.mkdir(parents=True, exist_ok=True)
            ssh.download(remote_db, local_db, timeout=120)
            print(f"✓ fetched DB: {local_db} ({size} bytes)")
        else:
            print("⚠ remote DB is empty; nothing to fetch")

    return exit_code


def _stream_remote_log(ssh: SshClient, remote_log: str, *, pid: str) -> None:
    """Tail `remote_log` until the remote pid is gone.

    Uses a single nested ssh session: `tail -F` in the background plus a
    `kill -0` poll loop so the session self-terminates when the remote bench
    exits. A non-zero SSH exit is treated as transport failure — callers need
    to verify the exit sentinel afterward, but we surface the drop visibly.
    """
    from mcode.launch.ssh import DEFAULT_SSH_OPTIONS

    argv = [
        "ssh",
        *DEFAULT_SSH_OPTIONS,
        ssh.login,
        f"tail -F -n +1 {shlex.quote(remote_log)} & TPID=$!; "
        f"while kill -0 {shlex.quote(pid)} 2>/dev/null; do sleep 5; done; "
        "sleep 2; kill $TPID 2>/dev/null || true",
    ]
    proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
    rc = proc.wait()
    if rc != 0:
        print(f"⚠ log-stream ssh exited {rc}; remote job may still be running")


__all__ = ["RemoteBenchError", "run_bench_on_bluevela"]
