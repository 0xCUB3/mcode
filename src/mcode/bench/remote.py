"""Run a bench command on Blue Vela via SSH."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from mcode.bench.remote_script import build_lsf_submit_command, build_remote_bench_plan
from mcode.bench.summary import safe_rerun_metadata
from mcode.launch import config as launch_config
from mcode.launch import state as launch_state
from mcode.launch.models import Target
from mcode.launch.ssh import SshClient
from mcode.util import temporary_directory


class RemoteBenchError(RuntimeError):
    """User-facing remote execution error."""


_FORWARDED_ENV_VARS = (
    "MCODE_CONTEXT_WINDOW",
    "MCODE_MAX_NEW_TOKENS",
    "MCODE_REACT_TIMEOUT",
)


def _emit_remote_event(json_mode: bool, kind: str, message: str, **data: object) -> None:
    if not json_mode:
        print(message)
        return
    event_data = dict(data)
    if kind == "remote_stdout" and event_data.get("line") == message:
        payload_data = event_data
    else:
        payload_data = {"message": message, **event_data}
    payload = {"kind": kind, "data": payload_data}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _remote_run_key(
    *,
    model: str,
    local_db: Path,
    bench_argv: list[str],
    forwarded_env: dict[str, str],
) -> str:
    payload = {
        "model": model,
        "local_db": str(local_db.resolve()),
        "bench_argv": bench_argv,
        "forwarded_env": forwarded_env,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    safe_model = model.replace("/", "-")[:24]
    return f"bench-{digest}-{safe_model}"


_LSF_JOB_ID_RE = re.compile(r"Job\s*<(\d+)>")


def _parse_lsf_job_id(text: str) -> str:
    match = _LSF_JOB_ID_RE.search(text)
    if not match:
        raise RemoteBenchError(f"could not parse LSF job id from bsub output: {text.strip()[:200]}")
    return match.group(1)


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
    fetch_artifacts: bool = False,
) -> int:
    """Submit `uv run mcode bench <bench_argv>` to a Blue Vela compute node.

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
    forwarded_env = {name: value for name in _FORWARDED_ENV_VARS if (value := os.environ.get(name))}
    run_id = _remote_run_key(
        model=model,
        local_db=local_db,
        bench_argv=bench_argv,
        forwarded_env=forwarded_env,
    )
    attempt_token = f"{int(time.time() * 1000)}-{os.getpid()}"
    plan = build_remote_bench_plan(
        bench_argv=bench_argv,
        bv=bv,
        run_id=run_id,
        attempt_token=attempt_token,
        endpoint=endpoint,
        api_key=api_key,
        forwarded_env=forwarded_env,
    )
    argv = plan.argv
    remote_dir = plan.remote_dir
    remote_db = plan.remote_db
    remote_logs_dir = plan.remote_logs_dir
    remote_log = plan.remote_log
    exit_sentinel = plan.exit_sentinel
    remote_script_path = plan.remote_script_path
    svc_log = plan.podman_svc_log
    remote_script = plan.script
    local_artifact_fetch = plan.local_artifact_fetch

    ssh.run(
        f"mkdir -p {shlex.quote(remote_dir)} {shlex.quote(remote_logs_dir)}",
        timeout=30,
    )

    # Open a RunRecord so `mcode bench list` / `mcode bench cancel` can find
    # this run. The LSF job id is filled in after bsub accepts the script.
    # Wrapped in try/finally so unexpected exceptions (parse failure, SSH I/O,
    # download error) close the run instead of leaving it RUNNING forever.
    benchmark_name = bench_argv[0] if bench_argv else "unknown"
    remote_artifact_metadata: dict[str, str] = {}
    if local_artifact_fetch is not None:
        remote_artifact_metadata = {
            "remote_artifact_dir": local_artifact_fetch.remote_dir,
            "local_artifact_dir": str(local_artifact_fetch.local_dir),
        }

    _upsert_run(
        run_id=run_id,
        benchmark=benchmark_name,
        status=launch_state.RunStatus.RUNNING,
        remote={
            "login": bv.login,
            "run_dir": remote_dir,
            "attempt_token": attempt_token,
            "remote_db": remote_db,
            "remote_log": remote_log,
            "exit_sentinel": exit_sentinel,
            "podman_svc_log": svc_log,
            "remote_script": remote_script_path,
            **remote_artifact_metadata,
        },
        db_path=str(local_db),
        started_at=time.time(),
        metadata=safe_rerun_metadata(),
    )
    final_status: launch_state.RunStatus = launch_state.RunStatus.FAILED
    cancel_reason: str | None = None
    exit_code = 1
    try:
        with temporary_directory(prefix="mcode-remote-script-") as tmp_dir_local:
            local_script = Path(tmp_dir_local) / Path(remote_script_path).name
            local_script.write_text(remote_script + "\n", encoding="utf-8")
            ssh.upload(local_script, remote_script_path, timeout=60)

        queue = bv.queue_order[0]
        submit_cmd = build_lsf_submit_command(plan=plan, bv=bv, queue=queue)
        r = ssh.run(submit_cmd, timeout=60)
        if not r.ok:
            detail = (r.stderr or r.stdout).strip()
            raise RemoteBenchError(f"failed to submit remote bench: {detail}")
        job_id = _parse_lsf_job_id(r.stdout + r.stderr)
        _upsert_run(
            run_id=run_id,
            remote={
                "login": bv.login,
                "run_dir": remote_dir,
                "attempt_token": attempt_token,
                "job_id": job_id,
                "queue": queue,
                "remote_db": remote_db,
                "remote_log": remote_log,
                "exit_sentinel": exit_sentinel,
                "podman_svc_log": svc_log,
                "remote_script": remote_script_path,
                **remote_artifact_metadata,
            },
            log_paths=[remote_log],
        )
        json_mode = "--json" in argv
        _emit_remote_event(
            json_mode,
            "remote_submit",
            f"remote bench submitted: job={job_id} queue={queue} log={remote_log}",
            job_id=job_id,
            queue=queue,
            log=remote_log,
        )

        try:
            _stream_remote_log(
                ssh, remote_log, exit_sentinel=exit_sentinel, job_id=job_id, json_mode=json_mode
            )
        except KeyboardInterrupt:
            final_status = launch_state.RunStatus.STOPPED
            cancel_reason = "interrupt"
            _emit_remote_event(
                json_mode,
                "remote_interrupt",
                "interrupted; remote LSF job is still running",
                job_id=job_id,
                log=remote_log,
                login=bv.login,
            )
            return 130
        except Exception as exc:
            _emit_remote_event(
                json_mode,
                "remote_warning",
                f"log streaming failed: {exc}; checking remote sentinel and DB",
            )

        sentinel_r = ssh.run(
            f"cat {shlex.quote(exit_sentinel)} 2>/dev/null",
            timeout=30,
        )
        sentinel_ok = bool(sentinel_r.ok)
        if sentinel_ok:
            try:
                exit_code = int((sentinel_r.stdout.strip() or "99").splitlines()[-1])
            except ValueError:
                exit_code = 99
        else:
            exit_code = 99
            _emit_remote_event(
                json_mode,
                "remote_warning",
                "remote bench did not write a readable exit sentinel",
            )

        qdb = shlex.quote(remote_db)
        size_r = ssh.run(
            f"test -f {qdb} && stat -c %s {qdb} || echo 0",
            timeout=30,
        )
        try:
            size = int((size_r.stdout.strip() or "0").splitlines()[-1])
        except ValueError:
            size = 0

        if fetch_db:
            if size > 0:
                local_db.parent.mkdir(parents=True, exist_ok=True)
                try:
                    ssh.download(remote_db, local_db, timeout=120)
                    _emit_remote_event(
                        json_mode,
                        "remote_fetch_db",
                        f"fetched DB: {local_db} ({size} bytes)",
                        path=str(local_db),
                        bytes=size,
                    )
                except Exception as exc:
                    _emit_remote_event(
                        json_mode,
                        "remote_warning",
                        f"remote DB exists but download failed: {exc}; remote_db={remote_db}",
                    )
                    exit_code = exit_code or 99
            else:
                _emit_remote_event(
                    json_mode,
                    "remote_warning",
                    "remote DB is empty; nothing to fetch",
                )

        if fetch_artifacts and local_artifact_fetch is not None:
            remote_artifact_dir = local_artifact_fetch.remote_dir
            local_artifact_dir = local_artifact_fetch.local_dir
            artifact_check = ssh.run(
                f"test -d {shlex.quote(remote_artifact_dir)} && echo ok || echo missing",
                timeout=30,
            )
            if artifact_check.ok and artifact_check.stdout.strip().endswith("ok"):
                try:
                    ssh.download_tree(
                        remote_artifact_dir,
                        local_artifact_dir,
                        timeout=300,
                    )
                    _emit_remote_event(
                        json_mode,
                        "remote_fetch_artifacts",
                        f"fetched artifacts: {local_artifact_dir}",
                        path=str(local_artifact_dir),
                    )
                except Exception as exc:
                    _emit_remote_event(
                        json_mode,
                        "remote_warning",
                        "remote artifacts exist but download failed: "
                        f"{exc}; remote_artifact_dir={remote_artifact_dir}",
                    )
                    exit_code = exit_code or 99
            else:
                _emit_remote_event(
                    json_mode,
                    "remote_warning",
                    "remote artifact directory is missing; nothing to fetch",
                )

        if sentinel_ok:
            try:
                if _wait_for_lsf_job_inactive(ssh, job_id, timeout_s=15):
                    pass
                elif _lsf_job_is_active(ssh, job_id):
                    _emit_remote_event(
                        json_mode,
                        "remote_cleanup",
                        f"benchmark finished but LSF job {job_id} is still active; sending bkill",
                        job_id=job_id,
                    )
                    kill_r = ssh.run(f"bkill {shlex.quote(job_id)}", timeout=10)
                    if not kill_r.ok:
                        detail = (kill_r.stderr or kill_r.stdout or "bkill failed").strip()
                        _emit_remote_event(
                            json_mode,
                            "remote_warning",
                            f"failed to clean up lingering LSF job {job_id}: {detail}",
                            job_id=job_id,
                        )
                    else:
                        time.sleep(10)
                        if _lsf_job_is_active(ssh, job_id):
                            _emit_remote_event(
                                json_mode,
                                "remote_warning",
                                (
                                    f"lingering LSF job {job_id} is still active after bkill; "
                                    f"inspect {remote_log}"
                                ),
                                job_id=job_id,
                                log=remote_log,
                            )
                        else:
                            _emit_remote_event(
                                json_mode,
                                "remote_cleanup",
                                f"cleaned up lingering LSF job {job_id}",
                                job_id=job_id,
                            )
            except Exception as exc:
                _emit_remote_event(
                    json_mode,
                    "remote_warning",
                    f"failed to clean up lingering LSF job {job_id}: {exc}",
                    job_id=job_id,
                )

        final_status = (
            launch_state.RunStatus.DONE if exit_code == 0 else launch_state.RunStatus.FAILED
        )
        return exit_code
    finally:
        try:
            patch: dict = {
                "run_id": run_id,
                "status": final_status,
                "ended_at": time.time(),
            }
            if cancel_reason is not None:
                patch["metadata"] = {"cancel_reason": cancel_reason}
            _upsert_run(**patch)
        except Exception:
            pass


def _upsert_run(
    *,
    run_id: str,
    benchmark: str | None = None,
    status: launch_state.RunStatus | None = None,
    remote: dict | None = None,
    db_path: str | None = None,
    log_paths: list[str] | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
    metadata: dict | None = None,
) -> None:
    """Create-or-update a RunRecord under fcntl lock.

    Patches only the fields supplied; leaves the rest intact. Used during
    Blue Vela bench runs so `mcode bench list` and `mcode bench cancel` have
    something to find.
    """

    def _mutator(s: launch_state.State) -> None:
        existing = s.run(run_id)
        if existing is None:
            existing = launch_state.RunRecord(
                id=run_id,
                target=Target.BLUEVELA,
                benchmark=benchmark or "unknown",
            )
        if benchmark is not None:
            existing.benchmark = benchmark
        if status is not None:
            existing.status = status
        if remote is not None:
            existing.remote = {**existing.remote, **remote}
        if db_path is not None:
            existing.db_path = db_path
        if log_paths is not None:
            existing.log_paths = log_paths
        if started_at is not None:
            existing.started_at = started_at
        if ended_at is not None:
            existing.ended_at = ended_at
        if metadata is not None:
            existing.metadata = {**existing.metadata, **metadata}
        existing.updated_at = str(time.time())
        s.upsert_run(existing)

    launch_state.update(None, _mutator)


def _wait_for_lsf_job_inactive(ssh: SshClient, job_id: str, *, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _lsf_job_is_active(ssh, job_id):
            return True
        time.sleep(1)
    return not _lsf_job_is_active(ssh, job_id)


def _lsf_job_is_active(ssh: SshClient, job_id: str) -> bool:
    check = ssh.run(
        "STAT=$(bjobs -noheader -o stat "
        f"{shlex.quote(job_id)} 2>/dev/null | tr -d '[:space:]' || true); "
        'case "$STAT" in PEND|RUN|PSUSP|USUSP|SSUSP) exit 1 ;; *) exit 0 ;; esac',
        timeout=10,
    )
    return not bool(getattr(check, "ok", True))


def _stream_remote_log(
    ssh: SshClient,
    remote_log: str,
    *,
    exit_sentinel: str,
    job_id: str,
    json_mode: bool = False,
) -> None:
    """Tail one attempt log until the bench finishes or the LSF job exits."""
    from mcode.launch.ssh import DEFAULT_SSH_OPTIONS

    sentinel_q = shlex.quote(exit_sentinel)
    job_q = shlex.quote(job_id)
    log_q = shlex.quote(remote_log)
    grace_s = 3
    sentinel_poll = (
        f'if [ "$SENTINEL_SEEN" = 0 ] && test -f {sentinel_q}; then '
        "SENTINEL_SEEN=1; "
        f"DEADLINE=$(( $(date +%s) + {grace_s} )); "
        "fi; "
    )
    job_stat_poll = (
        f"STAT=$(bjobs -noheader -o stat {job_q} 2>/dev/null | tr -d '[:space:]' || true); "
    )
    sentinel_grace = (
        'if [ "$SENTINEL_SEEN" = 1 ]; then '
        'if [ "$(date +%s)" -ge "$DEADLINE" ]; then break; fi; '
        "sleep 1; continue; "
        "fi; "
    )
    sentinel_warning = (
        'if [ "$SENTINEL_SEEN" = 1 ]; then '
        'case "$STAT" in PEND|RUN|PSUSP|USUSP|SSUSP) '
        f'echo "WARNING: benchmark finished but LSF job {job_id} is still $STAT; " '
        f'"inspect {remote_log}" >&2 ;; '
        "esac; "
        "fi"
    )
    argv = [
        "ssh",
        *DEFAULT_SSH_OPTIONS,
        ssh.login,
        (
            f"tail -n 0 -F {log_q} & TPID=$!; "
            "STAT=''; SENTINEL_SEEN=0; DEADLINE=0; "
            "while true; do "
            f"{sentinel_poll}{job_stat_poll}{sentinel_grace}"
            'case "$STAT" in PEND|RUN|PSUSP|USUSP|SSUSP) sleep 5 ;; *) break ;; esac; '
            "done; "
            "kill $TPID 2>/dev/null || true; wait $TPID 2>/dev/null || true; "
            f"{sentinel_warning}"
        ),
    ]
    if not json_mode:
        proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
        rc = proc.wait()
        if rc != 0:
            print(f"⚠ log-stream ssh exited {rc}; remote job may still be running")
        return

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                print(stripped, flush=True)
                continue
        if _drop_remote_json_line(stripped):
            continue
        _emit_remote_event(json_mode, "remote_stdout", line, line=line)
    rc = proc.wait()
    if rc != 0:
        _emit_remote_event(
            json_mode,
            "remote_warning",
            f"log-stream ssh exited {rc}; remote job may still be running",
            returncode=rc,
        )


def _drop_remote_json_line(line: str) -> bool:
    return (
        line.startswith("tail: cannot open ")
        or (line.startswith("tail: '") and " has appeared;" in line)
        or line.startswith("WARNING: benchmark finished but LSF job ")
    )


__all__ = ["RemoteBenchError", "run_bench_on_bluevela"]
