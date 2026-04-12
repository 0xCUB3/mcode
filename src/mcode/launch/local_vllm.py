"""Local vLLM target.

Simplest end-to-end path — no SSH, no LSF. Spawns `vllm serve` as a local
subprocess, polls the HTTP health endpoint until ready, writes a
ServerRecord with the endpoint, and returns.

v1 scope: server lifecycle only. Benchmark invocation is out-of-scope for
local-vllm; user runs `OPENAI_BASE_URL=... mcode bench ...` themselves
against the printed endpoint. Bluevela.py is where shard lifecycle lives
because LSF is what the launcher exists to tame.

Phases:
    submit   → spawning vllm serve
    starting → model load + warmup
    ready    → HTTP /v1/models returns 200

Server stop: `stop(record_id)` sends SIGTERM, waits, SIGKILL, cleans state.
Refresh: `refresh(record)` checks the pid; stale records flip to stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from mcode.launch import state
from mcode.launch.config import LaunchConfig
from mcode.launch.models import (
    Check,
    LaunchError,
    LaunchSpec,
    Phase,
    PhaseStatus,
    RunRecord,
    RunStatus,
    ServerRecord,
    Target,
)
from mcode.launch.progress import _ReporterBase as Reporter

PHASES: list[Phase] = [
    Phase("submit", "Start vLLM server"),
    Phase("starting", "Load model + warmup"),
    Phase("ready", "Server healthy"),
]

_STARTUP_ABSOLUTE_DEADLINE_S = 1800  # 30 min from spawn to HTTP 200
_HEALTH_POLL_INTERVAL_S = 2.0
_LOG_DIR = Path.home() / ".local" / "state" / "mcode" / "launch"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _config_hash(spec: LaunchSpec) -> str:
    p = spec.profile
    payload = {
        "model": spec.model,
        "flags": p.flags,
        "tp": p.tensor_parallel,
        "max_model_len": p.max_model_len,
        "extra_env": p.extra_env,
        "chat_template": p.chat_template,
        "image": p.image,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _run_dir_for(run_id: str) -> Path:
    d = _LOG_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _endpoint(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def _health_check(port: int, *, timeout_s: float = 2.0) -> tuple[bool, int]:
    url = f"{_endpoint(port)}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return (200 <= resp.status < 300), resp.status
    except urllib.error.HTTPError as e:
        return False, e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False, 0


def _resolve_chat_template(spec: LaunchSpec) -> Path | None:
    """Locate a chat-template file for the profile, or raise LaunchError if
    the profile requires one but it's missing. Silent-drop is not acceptable
    (Codex review fix) — Gemma4 etc. produces a server that looks healthy
    but silently breaks tool calls without the template.

    Lookup order: absolute path as given, then bundled launch/resources/<name>.
    """
    tmpl = spec.profile.chat_template
    if not tmpl:
        return None
    candidate = Path(tmpl)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    bundled = (Path(__file__).parent / "resources" / tmpl).resolve()
    if bundled.exists():
        return bundled
    raise LaunchError(
        what=f"chat template {tmpl!r} required by profile '{spec.profile.name}' not found",
        why=f"tried {bundled} and absolute path {candidate}",
        next=(
            f"drop the file at {bundled}, or set profile.chat_template to an "
            "absolute path that exists"
        ),
    )


def _build_vllm_argv(spec: LaunchSpec, port: int) -> list[str]:
    p = spec.profile
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec.model,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--tensor-parallel-size",
        str(p.tensor_parallel),
        "--max-model-len",
        str(p.max_model_len),
    ]
    tmpl_path = _resolve_chat_template(spec)
    if tmpl_path is not None:
        argv += ["--chat-template", str(tmpl_path)]
    argv += list(p.flags)
    return argv


def _process_identity(pid: int) -> str | None:
    """Return a stable identifier for the process at pid, or None if the
    process doesn't exist. Used to detect PID reuse across refresh()/stop().

    Uses `ps -o lstart=,etime=` which is portable across macOS and Linux.
    """
    try:
        r = subprocess.run(
            ["ps", "-o", "lstart=,etime=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _terminate_pid(pid: int, *, expected_identity: str | None, grace_s: float) -> bool:
    """Send SIGTERM then (if needed) SIGKILL to pid, verifying the process is
    still the one we spawned. Returns True if the target was our process (or
    already gone). Returns False if the pid now refers to an unrelated process.
    """
    current = _process_identity(pid)
    if current is None:
        return True  # already gone
    if expected_identity is not None and current != expected_identity:
        return False  # pid reuse — don't touch
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return True
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if _process_identity(pid) != current:
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return True


def _merge_env(spec: LaunchSpec) -> dict[str, str]:
    env = os.environ.copy()
    for k, v in spec.profile.extra_env.items():
        env[k] = v
    return env


def _read_tail(log_path: Path, max_lines: int = 3) -> str:
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return " · ".join(lines[-max_lines:])[-200:]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def launch(
    spec: LaunchSpec,
    reporter: Reporter,
    *,
    cfg: LaunchConfig | None = None,
    state_path: Path | None = None,
) -> ServerRecord:
    """Spin up a local vLLM server. Returns the ServerRecord once healthy.

    Raises LaunchError on failure with an actionable `next:` hint. vLLM logs
    land in ~/.local/state/mcode/launch/<run-id>/vllm.log.
    """
    if spec.target != Target.LOCAL_VLLM:
        raise LaunchError(
            what="local_vllm.launch called with wrong target",
            why=f"spec.target = {spec.target!r}",
            next="use the bluevela.launch or local_ollama.launch module",
        )

    cfg = cfg or LaunchConfig()
    port = cfg.local_vllm.port
    run_id = f"local-vllm-{uuid.uuid4().hex[:8]}"
    run_dir = _run_dir_for(run_id)
    log_path = run_dir / "vllm.log"

    reporter.add_phases(PHASES)

    # --- submit phase ------------------------------------------------------
    reporter.start(
        "submit",
        feed=lambda: f"preparing {spec.profile.name} on port {port}",
    )
    argv = _build_vllm_argv(spec, port)
    env = _merge_env(spec)

    # Fail fast if port is occupied by an existing process.
    ok, status = _health_check(port, timeout_s=0.5)
    if ok:
        reporter.finish(PhaseStatus.FAILED, detail=f"port {port} already serving")
        raise LaunchError(
            what=f"port {port} is already in use",
            why=f"something is already answering /v1/models on 127.0.0.1:{port}",
            next="stop the existing process or change [local_vllm].port in your config",
        )

    log_fh = log_path.open("w", buffering=1)
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # detach so Ctrl-C in our process doesn't kill it
        )
    except FileNotFoundError as e:
        log_fh.close()
        reporter.finish(PhaseStatus.FAILED, detail="vllm not on PATH")
        raise LaunchError(
            what="vllm binary not found",
            why=str(e),
            next="install vLLM: `uv pip install vllm` (or activate an env that has it)",
        ) from e
    except Exception as e:
        log_fh.close()
        reporter.finish(PhaseStatus.FAILED, detail=str(e))
        raise LaunchError(what="failed to spawn vllm", why=str(e), next=f"see {log_path}") from e
    finally:
        # Close the parent's write-end of the log immediately after spawn.
        # Popen dups it into the child; we don't want to hold it ourselves
        # (leak across failure paths; also stops the log from being flushed
        # on our side when we don't write to it).
        log_fh.close()

    reporter.finish(PhaseStatus.DONE, detail=f"pid {proc.pid}")

    # Capture process identity immediately so PID-reuse detection in
    # refresh()/stop() has something to compare against.
    identity = _process_identity(proc.pid) or ""

    # From here on, any LaunchError path must tear down the child, otherwise
    # we leak a detached server holding a port + GPU.
    def _kill_child_on_failure() -> None:
        if proc is None:
            return
        try:
            _terminate_pid(proc.pid, expected_identity=identity, grace_s=5.0)
        except Exception:
            pass

    try:
        # --- starting phase ------------------------------------------------
        deadline = time.monotonic() + _STARTUP_ABSOLUTE_DEADLINE_S
        reporter.start(
            "starting",
            feed=lambda: _starting_detail(proc, log_path),
            mode="fast",
        )
        last_status = 0
        while True:
            if proc.poll() is not None:
                reporter.finish(
                    PhaseStatus.FAILED,
                    detail=f"vllm exited with code {proc.returncode}",
                )
                tail = _read_tail(log_path, max_lines=10)
                raise LaunchError(
                    what="vllm exited before the endpoint became healthy",
                    why=tail or f"exit code {proc.returncode}",
                    next=_startup_hint(tail, spec),
                    logs=str(log_path),
                )
            ok, last_status = _health_check(port)
            if ok:
                break
            if time.monotonic() > deadline:
                _kill_child_on_failure()
                reporter.finish(PhaseStatus.FAILED, detail="startup deadline exceeded")
                raise LaunchError(
                    what=f"vllm did not become ready within {_STARTUP_ABSOLUTE_DEADLINE_S}s",
                    why=_read_tail(log_path) or f"last HTTP status: {last_status}",
                    next="check GPU memory and --max-model-len; inspect the log below",
                    logs=str(log_path),
                )
            time.sleep(_HEALTH_POLL_INTERVAL_S)
        reporter.finish(PhaseStatus.DONE)

        # --- ready phase ---------------------------------------------------
        reporter.start("ready")
        endpoint = _endpoint(port)
        server = ServerRecord(
            id=f"server-{run_id}",
            target=Target.LOCAL_VLLM,
            endpoint=endpoint,
            model=spec.model,
            config_hash=_config_hash(spec),
            job_id=str(proc.pid),
            log_path=str(log_path),
            started_at=_now_iso(),
            status="healthy",
            metadata={"port": port, "argv": argv, "proc_identity": identity},
        )
        try:
            state.update(state_path, lambda s: s.upsert_server(server))
        except Exception:
            # If we can't persist the record, we MUST kill the child — otherwise
            # we've orphaned a server with no way to stop it via our CLI.
            _kill_child_on_failure()
            reporter.finish(PhaseStatus.FAILED, detail="state persistence failed")
            raise
        reporter.finish(PhaseStatus.DONE, detail=endpoint)
        return server
    except BaseException:
        # Catch-all safety net: any unexpected exception (incl. KeyboardInterrupt)
        # must tear down the detached child. The specific LaunchError paths above
        # already called _kill_child_on_failure(); this handles everything else.
        if proc is not None and proc.poll() is None:
            _kill_child_on_failure()
        raise


def _starting_detail(proc: subprocess.Popen, log_path: Path) -> str:
    tail = _read_tail(log_path, max_lines=1)
    alive = proc.poll() is None
    status = f"pid {proc.pid}" + (" alive" if alive else f" exited {proc.returncode}")
    return f"{status} · {tail}" if tail else status


def _startup_hint(tail: str, spec: LaunchSpec) -> str:
    t = (tail or "").lower()
    if "no available memory" in t or "out of memory" in t or "cudamemoryerror" in t:
        return (
            "model too large for the GPU — reduce --max-model-len in the profile, "
            "or pick a smaller model"
        )
    if "invalid tool call parser" in t:
        return (
            f"profile '{spec.profile.name}' has a parser vLLM doesn't know "
            "— check launch/profiles.py"
        )
    if "cuda" in t and "not available" in t:
        return "vLLM can't see a CUDA device — check `nvidia-smi` and your Python env"
    if "chat template" in t or "chat-template" in t:
        return "chat template missing or wrong — verify profile.chat_template points to a real file"
    return "inspect the log file; re-run with a smaller model to isolate the issue"


def doctor(cfg: LaunchConfig | None = None) -> list[Check]:
    cfg = cfg or LaunchConfig()
    checks: list[Check] = []

    # vllm importable?
    try:
        import vllm  # noqa: F401

        checks.append(Check(name="vllm importable", ok=True, detail=vllm.__version__))
    except ImportError as e:
        checks.append(
            Check(
                name="vllm importable",
                ok=False,
                detail=str(e),
                next="`uv pip install vllm` in the env mcode runs in",
            )
        )

    # port free?
    ok, status = _health_check(cfg.local_vllm.port, timeout_s=0.5)
    checks.append(
        Check(
            name=f"port {cfg.local_vllm.port} free",
            ok=not ok,
            detail="in use" if ok else "free",
            next=("stop the existing process or change [local_vllm].port" if ok else ""),
        )
    )
    return checks


def stop(server_id: str, *, state_path: Path | None = None, grace_s: float = 5.0) -> bool:
    """SIGTERM the server's pid, wait, SIGKILL if needed, clear state.

    PID reuse safe (Codex review fix): the recorded proc_identity from launch
    is compared against the live pid's identity before any signal is sent. If
    they don't match, the pid has been reused — we DO NOT touch the unrelated
    process, but we do clean up the stale state record.

    Returns True if a record was found (and termination was safe to attempt),
    False if there was no such record.
    """
    srv = state.load(state_path).server(server_id)
    if srv is None:
        return False

    pid: int | None = None
    if srv.job_id:
        try:
            pid = int(srv.job_id)
        except ValueError:
            pid = None

    if pid is not None:
        expected = (srv.metadata or {}).get("proc_identity") or None
        _terminate_pid(pid, expected_identity=expected, grace_s=grace_s)

    # Only drop the state record AFTER termination decisions are made.
    def _drop(s: state.State) -> None:
        s.servers = [x for x in s.servers if x.id != server_id]

    state.update(state_path, _drop)
    return True


def refresh(record: RunRecord | ServerRecord) -> RunRecord | ServerRecord:
    """Re-check whether the backing process is still the one we spawned.

    Uses proc_identity from metadata (captured at launch) to distinguish a
    still-running server from a reused PID. If the live PID belongs to an
    unrelated process, the record's status flips to `stopped`.
    """
    pid_raw = record.job_id if isinstance(record, ServerRecord) else None
    if pid_raw is None:
        return record
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        return record
    current_identity = _process_identity(pid)
    expected_identity = (
        (record.metadata or {}).get("proc_identity") if isinstance(record, ServerRecord) else None
    )
    if current_identity is None:
        alive_same_proc = False
    elif expected_identity and current_identity != expected_identity:
        # PID reuse — the pid exists but it's someone else.
        alive_same_proc = False
    else:
        alive_same_proc = True
    if isinstance(record, ServerRecord):
        record.status = "healthy" if alive_same_proc else "stopped"
    elif isinstance(record, RunRecord):
        if not alive_same_proc and record.status == RunStatus.RUNNING:
            record.status = RunStatus.DONE
    return record
