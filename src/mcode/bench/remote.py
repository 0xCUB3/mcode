"""Run a bench command on Blue Vela without user-facing shell scripts.

Cluster-side prerequisites are handled by `launch/scripts/bluevela_shard.sh`:
source HF env, start podman socket, export DOCKER_HOST, invoke `mcode bench ...`.

Workflow when a user runs `mcode bench smoke --on bluevela --model X`:

1. Load the bluevela launch config.
2. Resolve a healthy vLLM endpoint for the model from launch state (or error
   with a hint to run `mcode launch bluevela --model X` first).
3. SSH to the login node, launch `bluevela_shard.sh` on the login host,
   forwarding the bench argv. Log/DB are written into the cluster workspace.
4. Stream the remote log back while running.
5. Rsync the resulting DB file back to the local `--db` path on success
   unless `--no-fetch-db` was passed.
"""

from __future__ import annotations

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


def _resolve_endpoint(model: str) -> str:
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

    endpoint = _resolve_endpoint(model)

    ssh = SshClient(bv.login)
    run_id = f"bench-{int(time.time())}-{model.replace('/', '-')[:24]}"
    remote_dir = f"{bv.workspace_root}/bench-runs/{run_id}"
    remote_db = f"{remote_dir}/results.db"
    remote_log = f"{remote_dir}/bench.log"

    # Replace/append --db so the bench writes to the remote path.
    argv = [*bench_argv]
    if "--db" in argv:
        i = argv.index("--db")
        argv[i + 1] = remote_db
    else:
        argv += ["--db", remote_db]

    ssh.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=30)

    # Assemble the remote shell: source hf-env (optional), start podman socket
    # if absent, export DOCKER_HOST + OPENAI_BASE_URL, cd, exec bench.
    hf_env = bv.hf_env
    bench_cmd = "uv run mcode bench " + " ".join(shlex.quote(a) for a in argv)
    remote_script = f"""
set -euo pipefail
cd {shlex.quote(bv.workspace_root)}
[ -f {shlex.quote(hf_env)} ] && source {shlex.quote(hf_env)}
SOCK="/tmp/podman-run-$(id -u)/podman.sock"
if ! curl -s --unix-socket "$SOCK" http://localhost/version >/dev/null 2>&1; then
  rm -f "$SOCK"
  mkdir -p "$(dirname "$SOCK")"
  nohup podman system service --time=0 "unix://$SOCK" >/tmp/mcode-podman-svc.log 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -s --unix-socket "$SOCK" http://localhost/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
export DOCKER_HOST="unix://$SOCK"
export OPENAI_BASE_URL={shlex.quote(endpoint)}
export OPENAI_API_KEY=dummy
{bench_cmd}
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

    # Stream remote log until the pid exits. `tail -F --pid` would be ideal
    # but is not portable; poll instead.
    try:
        _stream_remote_log(ssh, remote_log, pid=pid)
    except KeyboardInterrupt:
        print("\n⚠ interrupted; remote job still running. Check with:")
        print(f"  ssh {bv.login} 'tail -f {remote_log}'")
        return 130

    # Remote done. Probe exit — the bench process itself doesn't write a
    # sentinel, so we use the log tail to decide success/failure. Prefer the
    # DB row count: 0 rows means the bench died before writing anything.
    try_db = ssh.run(
        f"test -f {shlex.quote(remote_db)} && stat -c %s {shlex.quote(remote_db)} || echo 0",
        timeout=30,
    )
    size = int((try_db.stdout.strip() or "0").splitlines()[-1])
    exit_code = 0 if size > 0 else 1

    if fetch_db and size > 0:
        local_db.parent.mkdir(parents=True, exist_ok=True)
        ssh.download(remote_db, local_db, timeout=120)
        print(f"✓ fetched DB: {local_db}")
    elif size == 0:
        print("✗ remote DB is empty; see log on the cluster")

    return exit_code


def _stream_remote_log(ssh: SshClient, remote_log: str, *, pid: str) -> None:
    """Tail `remote_log` until the remote pid is gone."""
    # Open a long-running ssh pipe that runs `tail -f` and also polls the pid.
    # Keep it simple: spawn ssh tail with a trailing guard that exits once pid
    # is dead.
    argv = [
        "ssh",
        *_ssh_opts_for(ssh),
        ssh.login,
        f"tail -F -n +1 {shlex.quote(remote_log)} & TPID=$!; "
        f"while kill -0 {shlex.quote(pid)} 2>/dev/null; do sleep 5; done; "
        "sleep 2; kill $TPID 2>/dev/null || true",
    ]
    proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()


def _ssh_opts_for(ssh: SshClient) -> list[str]:
    # mirror SshClient's default options; avoids us reaching into private state.
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


__all__ = ["RemoteBenchError", "run_bench_on_bluevela"]
