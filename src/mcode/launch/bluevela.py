"""Blue Vela LSF target — server launch.

v1 scope: vLLM server lifecycle on Blue Vela. Submits a vLLM job, waits until
the advertised endpoint is healthy, records the ServerRecord. Shard
submission, fetch, and snapshot are follow-ons.

Architecture invariants (see the plan):

- Every bsub carries `-G <group>`. Empty group is a hard LaunchError.
- Login nodes only. No ssh to compute nodes ever.
- `env.json` + `jq @sh` carries all Python→shell config. No Python-side
  `export KEY=VALUE` assembly (B7).
- Podman paths derived inside the shell script from $LSB_JOBID etc (B5).
- Queue "auto" = config-declared queue_order + `bsub -H` hold validation (B4).
- Automatic server reuse is OFF in v1 — opt-in via LaunchSpec.reuse_server
  with full config_hash match (B1).
- Absolute startup deadline in addition to stall timeout (don't-regress).
- Transitions driven by explicit signals: bsub accept, LSF state leaves PEND,
  host file appears, HTTP /v1/models returns 200. Log markers decorate detail
  line but never drive transitions (M1).
- stop --all scopes to the caller's recorded runs — never bkill 0/-u.

Phases:
    submit   → `bsub` returns a job id
    queued   → LSF state is PEND (may take minutes on busy queues)
    starting → container pull + model load + warmup (single phase per M1)
    ready    → vllm_host.txt on remote AND HTTP /v1/models returns 200
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from mcode.launch import config as config_mod
from mcode.launch import state
from mcode.launch.config import BluevelaConfig, LaunchConfig, validate_for_bluevela
from mcode.launch.models import (
    Check,
    LaunchError,
    LaunchSpec,
    Phase,
    PhaseStatus,
    RunRecord,
    ServerRecord,
    Target,
)
from mcode.launch.progress import TransportError
from mcode.launch.progress import _ReporterBase as Reporter
from mcode.launch.ssh import SshClient

PHASES: list[Phase] = [
    Phase("submit", "Submit vLLM job to LSF"),
    Phase("queued", "Waiting in queue"),
    Phase("starting", "Container pull + model load"),
    Phase("ready", "Server healthy"),
]

_STARTUP_ABSOLUTE_DEADLINE_S = 1800  # 30 min from bsub accept to HTTP 200
_HOST_FILE_DEADLINE_S = 900  # 15 min for LSF to start the job + write host file
_HEALTH_POLL_SLOW_S = 10.0
_DEFAULT_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.17.0"
_DEFAULT_VLLM_PORT = 8321

_SCRIPTS_DIR = Path(__file__).parent / "scripts"
_RESOURCES_DIR = Path(__file__).parent / "resources"


# ---------------------------------------------------------------------------
# Failure catalog: list of (regex, [hints]) — multiple hints per pattern,
# labeled by severity. No destructive hints (M6). Applied against stderr +
# the tail of vllm.log to generate the LaunchError.next: line.
# ---------------------------------------------------------------------------
_FAILURE_CATALOG: list[tuple[re.Pattern, list[str]]] = [
    (
        re.compile(r"Connection refused|Connection timed out|No route to host", re.I),
        [
            "VPN disconnected? try `ping <your login host>`; reconnect the IBM VPN if needed",
        ],
    ),
    (
        re.compile(r"Permission denied \(publickey", re.I),
        [
            "add your key: `ssh-add ~/.ssh/id_ed25519`; check ~/.ssh/config sets "
            "IdentitiesOnly yes for the bluevela host",
        ],
    ),
    (
        re.compile(r"not a member of (the )?group|not authorized to use project", re.I),
        [
            "ask an admin to add you to the configured LSF group, or edit "
            "[bluevela].group in your launch.toml to a group you belong to",
        ],
    ),
    (
        re.compile(r"queue .* closed|queue .* is closed|Queue only accepts interactive", re.I),
        [
            "configured queue is closed or interactive-only; run `mcode launch doctor "
            "bluevela --init` to re-detect available queues",
        ],
    ),
    (
        re.compile(r"No available memory for the cache blocks", re.I),
        [
            "model too large for the allocated GPUs; increase tensor_parallel in "
            "the profile, or pick a smaller model",
        ],
    ),
    (
        re.compile(r"invalid tool call parser|unknown tool call parser", re.I),
        [
            "profile has a parser vLLM doesn't recognize — check launch/profiles.py "
            "for your model and update the parser name",
        ],
    ),
    (
        re.compile(r"HF_TOKEN|401 Client Error", re.I),
        [
            "HF token missing or invalid; populate <$HOME>/.config/mcode/hf-env.sh "
            "with `export HF_TOKEN=...` and accept the model's HF gate if any",
        ],
    ),
]


def _hint_for(error_text: str) -> str:
    for pattern, hints in _FAILURE_CATALOG:
        if pattern.search(error_text):
            return hints[0]
    return "inspect the log and re-run `mcode launch doctor bluevela` to isolate"


# ---------------------------------------------------------------------------
# env.json construction
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _config_hash(spec: LaunchSpec) -> str:
    p = spec.profile
    payload = {
        "model": spec.model,
        "image": p.image or _DEFAULT_VLLM_IMAGE,
        "flags": p.flags,
        "tp": p.tensor_parallel,
        "max_model_len": p.max_model_len,
        "port": _DEFAULT_VLLM_PORT,
        "extra_env": p.extra_env,
        "chat_template": p.chat_template,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_env_json(spec: LaunchSpec, cfg: BluevelaConfig, *, run_dir: str) -> dict:
    """Construct the env.json payload uploaded alongside the vLLM script.

    This is the single Python→shell boundary. Every path the shell needs is
    here, pre-resolved. The Python side does NO shell escaping — the script
    loads this via `jq @sh`.

    The builder is responsible for appending `--chat-template /chat-template.jinja`
    to VLLM_FLAGS when profile.chat_template is set (the shell mounts the file
    at that fixed container path). Without this, Gemma4 tool calls silently fail.
    """
    p = spec.profile
    vllm_flags = list(p.flags)
    chat_template_path: str | None = None
    if p.chat_template:
        # The launcher ships bundled templates in launch/resources/ and uploads
        # them to {shared_root}/templates/ on first use. The env carries the
        # *remote* path.
        chat_template_path = f"{cfg.shared_root}/templates/{p.chat_template}"
        # Codex fix B2 for progress.py had a parallel: the Python builder MUST
        # inject the flag; shell never guesses.
        vllm_flags += ["--chat-template", "/chat-template.jinja"]
    env: dict = {
        "MODEL": spec.model,
        "VLLM_IMAGE": p.image or _DEFAULT_VLLM_IMAGE,
        "VLLM_FLAGS": vllm_flags,
        "VLLM_PORT": str(_DEFAULT_VLLM_PORT),
        "VLLM_MAX_MODEL_LEN": str(p.max_model_len),
        "GPU_COUNT": str(p.tensor_parallel),
        "QUEUE": cfg.queue_order[0] if cfg.queue_order else "normal",
        "GROUP": cfg.group,
        "RUN_DIR": run_dir,
        "BV_SHARED_DIR": cfg.shared_root,
        "BV_HF_ENV": cfg.hf_env,
        "EXTRA_ENV": dict(p.extra_env),
    }
    if chat_template_path:
        env["CHAT_TEMPLATE_PATH"] = chat_template_path
    return env


def _shared_path(cfg: BluevelaConfig, *parts: str) -> str:
    """Join under cfg.shared_root using POSIX semantics (remote is always POSIX)."""
    base = cfg.shared_root.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts)
    return f"{base}/{tail}" if tail else base


# ---------------------------------------------------------------------------
# LSF primitives
# ---------------------------------------------------------------------------
_JOB_ID_RE = re.compile(r"Job\s*<(\d+)>")

# Codex fix: tight allowlists on anything we're about to interpolate into a
# remote shell command. Values that don't match are rejected up-front, so we
# never pass injection-prone characters to `bsub`/`bjobs`/`bkill`.
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")  # queue, group, job name
_SAFE_DIGITS_RE = re.compile(r"^\d{1,18}$")  # LSF job id
_SAFE_POSIX_PATH_RE = re.compile(r"^/[A-Za-z0-9_./\-]{0,511}$")  # remote paths


def _require_safe(name: str, value: str, pattern: re.Pattern) -> str:
    if not value or not pattern.match(value):
        raise LaunchError(
            what=f"unsafe {name} value for shell interpolation: {value!r}",
            why=f"does not match {pattern.pattern}",
            next=f"fix [bluevela].{name} in your launch.toml",
        )
    return value


def _q(value: str) -> str:
    """Shell-quote a single interpolation for remote sh. Even when the caller
    has allowlisted the value, we still quote — defense in depth."""
    return shlex.quote(value)


def _parse_job_id(text: str) -> str:
    m = _JOB_ID_RE.search(text)
    if not m:
        raise LaunchError(
            what="could not parse LSF job id from bsub output",
            why=text.strip()[:200],
            next="run `bsub` manually with the same flags to see what's going on",
        )
    return m.group(1)


def _bjobs_state(ssh: SshClient, job_id: str) -> str | None:
    """Return the raw LSF state (PEND/RUN/DONE/EXIT/PSUSP/USUSP) or None if
    bjobs can't see the job anymore (typically because it's been archived)."""
    _require_safe("job_id", job_id, _SAFE_DIGITS_RE)
    r = ssh.run(f"bjobs -noheader -o stat {_q(job_id)} 2>/dev/null", timeout=15)
    if not r.ok:
        return None
    states = {line.strip() for line in r.stdout.splitlines() if line.strip()}
    return next(iter(states)) if len(states) == 1 else None


def _validate_queue(
    ssh: SshClient, cfg: BluevelaConfig, queue: str, *, timeout: float = 60.0
) -> str | None:
    """bsub -H a no-op to validate queue+group+gpu submission. Returns None
    on success (the no-op is bkill'd immediately), error text on rejection.

    Per plan B4: cheap validation that catches 'queue closed', 'not in group',
    resource-string mismatches without waiting in the real queue.

    Timeout 60s (not 30): we disable SSH ControlPath multiplexing on purpose,
    so each call pays the full connect+auth cost. Over VPN that's 5-10s per
    call before bsub even runs. bsub itself is <100ms.
    """
    _require_safe("queue", queue, _SAFE_IDENT_RE)
    _require_safe("group", cfg.group, _SAFE_IDENT_RE)
    _require_safe("gpu_mode", cfg.gpu_mode, _SAFE_IDENT_RE)
    tag = f"mcode-qval-{uuid.uuid4().hex[:6]}"
    cmd = (
        f"bsub -H -G {_q(cfg.group)} -q {_q(queue)} -J {_q(tag)} -n 1 "
        f"-R {_q('span[hosts=1]')} -gpu {_q(f'num=1:mode={cfg.gpu_mode}')} "
        f"-W 2 -o /tmp/{_q(tag)}.out bash -c true"
    )
    r = ssh.run(cmd, timeout=timeout)
    if not r.ok:
        return (r.stderr or r.stdout).strip()[:300]
    m = _JOB_ID_RE.search(r.stdout + r.stderr)
    if m:
        ssh.run(f"bkill {_q(m.group(1))} >/dev/null 2>&1 || true", timeout=10)
    return None


def _pick_queue(ssh: SshClient, cfg: BluevelaConfig) -> str:
    """Try queue_order in order; first one that accepts a hold-validation job
    wins. All failures rolled up into a single LaunchError with per-queue
    rejection reasons.
    """
    if not cfg.queue_order:
        raise LaunchError(
            what="no queues configured",
            why="[bluevela].queue_order is empty",
            next="run `mcode launch doctor bluevela --init` to detect and write queue_order",
        )
    rejections: list[str] = []
    for q in cfg.queue_order:
        err = _validate_queue(ssh, cfg, q)
        if err is None:
            return q
        rejections.append(f"  {q}: {err}")
    raise LaunchError(
        what="no configured queue accepted the submission",
        why="\n".join(rejections),
        next="run `mcode launch doctor bluevela --init` to refresh queue_order",
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def launch(
    spec: LaunchSpec,
    reporter: Reporter,
    *,
    cfg: LaunchConfig | None = None,
    state_path: Path | None = None,
    ssh_client: SshClient | None = None,
) -> ServerRecord:
    if spec.target != Target.BLUEVELA:
        raise LaunchError(
            what="bluevela.launch called with wrong target",
            why=f"spec.target = {spec.target!r}",
            next="dispatch via cli.py or use the matching target module",
        )
    cfg = cfg or LaunchConfig()
    errs = validate_for_bluevela(cfg)
    if errs:
        raise LaunchError(
            what="bluevela config incomplete",
            why="; ".join(errs),
            next="run `mcode launch doctor bluevela --init`",
        )
    ssh = ssh_client or SshClient(cfg.bluevela.login)
    bv = cfg.bluevela

    run_id = f"bv-{uuid.uuid4().hex[:8]}"
    run_dir = _shared_path(bv, "runs", run_id)
    local_log = Path(f"/tmp/mcode-bluevela-{run_id}.log")

    reporter.add_phases(PHASES)

    # --- submit phase ------------------------------------------------------
    queue_chosen: str | None = None

    def submit_feed() -> str:
        return f"validating queues: {', '.join(bv.queue_order)}"

    reporter.start("submit", feed=submit_feed)
    try:
        queue = _pick_queue(ssh, bv)
        queue_chosen = queue
    except TransportError as e:
        reporter.finish(PhaseStatus.FAILED, detail=str(e))
        raise LaunchError(
            what="cannot reach Blue Vela",
            why=str(e),
            next=_hint_for(str(e)),
        ) from e

    # Validate everything that will be interpolated into remote shell
    # commands before we touch the cluster. This is defense-in-depth on top of
    # shlex.quote — a malicious config never reaches the command line.
    _require_safe("queue", queue, _SAFE_IDENT_RE)
    _require_safe("group", bv.group, _SAFE_IDENT_RE)
    _require_safe("gpu_mode", bv.gpu_mode, _SAFE_IDENT_RE)
    _require_safe("run_dir", run_dir, _SAFE_POSIX_PATH_RE)
    tp = spec.profile.tensor_parallel
    if not isinstance(tp, int) or tp <= 0 or tp > 32:
        raise LaunchError(
            what=f"profile.tensor_parallel out of range: {tp!r}",
            why="must be a positive int ≤ 32",
            next="fix the profile in launch/profiles.py",
        )

    # Upload env.json and the vLLM script.
    env_payload = build_env_json(spec, bv, run_dir=run_dir)
    env_payload["QUEUE"] = queue

    staging = Path(f"/tmp/mcode-bv-stage-{run_id}")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "env.json").write_text(json.dumps(env_payload, indent=2))

    # Upload the script and env.json into run_dir (created by remote mkdir).
    ssh.run(f"mkdir -p {_q(run_dir)}", timeout=30)
    ssh.upload(_SCRIPTS_DIR / "bluevela_vllm.sh", f"{run_dir}/vllm.sh", timeout=60)
    ssh.upload(staging / "env.json", f"{run_dir}/env.json", timeout=60)
    # Copy the chat template up on first use.
    if spec.profile.chat_template:
        tmpl = _RESOURCES_DIR / spec.profile.chat_template
        if not tmpl.exists():
            raise LaunchError(
                what=f"chat template {spec.profile.chat_template!r} missing locally",
                why=f"expected at {tmpl}",
                next="add the template to src/mcode/launch/resources/ and retry",
            )
        remote_tmpl = _shared_path(bv, "templates")
        _require_safe("templates_dir", remote_tmpl, _SAFE_POSIX_PATH_RE)
        ssh.run(f"mkdir -p {_q(remote_tmpl)}", timeout=30)
        ssh.upload(tmpl, f"{remote_tmpl}/{spec.profile.chat_template}", timeout=60)

    bsub_cmd = (
        f"bsub -G {_q(bv.group)} -q {_q(queue)} "
        f"-J {_q(f'mcode-vllm-{run_id}')} -n 1 -R {_q('span[hosts=1]')} "
        f"-gpu {_q(f'num={tp}:mode={bv.gpu_mode}')} "
        f"-o {_q(f'{run_dir}/vllm.log')} -e {_q(f'{run_dir}/vllm.log')} "
        f"bash {_q(f'{run_dir}/vllm.sh')}"
    )
    r = ssh.run(bsub_cmd, timeout=60)
    if not r.ok:
        reporter.finish(PhaseStatus.FAILED, detail=(r.stderr or "bsub failed").strip()[:80])
        raise LaunchError(
            what="bsub rejected the submission",
            why=(r.stderr or r.stdout).strip()[:400],
            next=_hint_for(r.stderr or r.stdout),
            logs=str(local_log),
        )
    job_id = _parse_job_id(r.stdout + r.stderr)
    _require_safe("job_id", job_id, _SAFE_DIGITS_RE)
    reporter.finish(PhaseStatus.DONE, detail=f"job {job_id} in queue {queue}")

    # Codex fix: persist a pending ServerRecord IMMEDIATELY after bsub accept.
    # If any later step fails, the caller still has a handle to bkill the job.
    pending_server = ServerRecord(
        id=f"server-{run_id}",
        target=Target.BLUEVELA,
        endpoint="",
        model=spec.model,
        config_hash=_config_hash(spec),
        job_id=job_id,
        log_path=f"{run_dir}/vllm.log",
        started_at=_now_iso(),
        status="pending",
        metadata={
            "queue": queue,
            "group": bv.group,
            "run_dir": run_dir,
            "login": bv.login,
        },
    )
    state.update(state_path, lambda s: s.upsert_server(pending_server))

    # Any exception from here on must bkill the accepted job, otherwise we
    # orphan a long-running GPU job with no way for the user to discover it.
    def _tear_down() -> None:
        try:
            ssh.run(f"bkill {_q(job_id)} >/dev/null 2>&1 || true", timeout=15)
        except Exception:
            pass

    try:
        # --- queued phase --------------------------------------------------
        def queued_feed() -> str:
            try:
                stat = _bjobs_state(ssh, job_id)
            except TransportError as e:
                raise TransportError(str(e)) from e
            return f"LSF state: {stat or '?'}"

        reporter.start("queued", feed=queued_feed, mode="slow")
        host_deadline = time.monotonic() + _HOST_FILE_DEADLINE_S
        while True:
            try:
                stat = _bjobs_state(ssh, job_id)
            except TransportError as e:
                raise LaunchError(
                    what="lost SSH during queue wait",
                    why=str(e),
                    next=_hint_for(str(e)),
                ) from e
            if stat is None or stat.upper() in ("RUN", "DONE"):
                break
            if stat.upper() in ("EXIT", "PSUSP", "USUSP"):
                tail = _remote_log_tail(ssh, run_dir)
                reporter.finish(PhaseStatus.FAILED, detail=f"LSF {stat}")
                raise LaunchError(
                    what=f"LSF job reached {stat} before running",
                    why=tail or f"bjobs state: {stat}",
                    next=_hint_for(tail or stat),
                    logs=f"ssh {bv.login} tail -n 50 {run_dir}/vllm.log",
                )
            if time.monotonic() > host_deadline:
                reporter.finish(PhaseStatus.FAILED, detail="queued too long")
                raise LaunchError(
                    what=f"LSF job stayed in {stat} past {_HOST_FILE_DEADLINE_S}s",
                    why="queue backlog",
                    next="try a different queue (edit [bluevela].queue_order) or retry later",
                )
            time.sleep(_HEALTH_POLL_SLOW_S)
        reporter.finish(PhaseStatus.DONE, detail=f"job {job_id} running")

        # --- starting phase ------------------------------------------------
        def starting_feed() -> str:
            tail = _remote_log_tail(ssh, run_dir, lines=1)
            host_inner = _remote_host_file(ssh, run_dir)
            where = f"host {host_inner}" if host_inner else "waiting for host file"
            return f"{where} · {tail}" if tail else where

        reporter.start("starting", feed=starting_feed, mode="slow")
        deadline = time.monotonic() + _STARTUP_ABSOLUTE_DEADLINE_S
        host: str | None = None
        while True:
            host = _remote_host_file(ssh, run_dir)
            if host:
                ok, status = _http_health(ssh, host, _DEFAULT_VLLM_PORT)
                if ok:
                    break
            if _bjobs_state(ssh, job_id) in (None, "EXIT"):
                tail = _remote_log_tail(ssh, run_dir)
                reporter.finish(PhaseStatus.FAILED, detail="job exited early")
                raise LaunchError(
                    what="vLLM job exited before endpoint became healthy",
                    why=tail or "LSF reports EXIT",
                    next=_hint_for(tail),
                    logs=f"ssh {bv.login} tail -n 100 {run_dir}/vllm.log",
                )
            if time.monotonic() > deadline:
                reporter.finish(PhaseStatus.FAILED, detail="startup deadline exceeded")
                raise LaunchError(
                    what=f"server did not become ready within {_STARTUP_ABSOLUTE_DEADLINE_S}s",
                    why=_remote_log_tail(ssh, run_dir) or "no progress",
                    next=(
                        "check log for OOM / chat-template / parser issues; "
                        "reduce max_model_len or TP in the profile"
                    ),
                    logs=f"ssh {bv.login} tail -n 200 {run_dir}/vllm.log",
                )
            time.sleep(_HEALTH_POLL_SLOW_S)
        reporter.finish(PhaseStatus.DONE, detail=f"host {host}")

        # --- ready phase ---------------------------------------------------
        reporter.start("ready")
        endpoint = f"http://{host}:{_DEFAULT_VLLM_PORT}/v1"
        server = ServerRecord(
            id=f"server-{run_id}",
            target=Target.BLUEVELA,
            endpoint=endpoint,
            model=spec.model,
            config_hash=_config_hash(spec),
            job_id=job_id,
            log_path=f"{run_dir}/vllm.log",
            started_at=_now_iso(),
            status="healthy",
            metadata={
                "queue": queue_chosen,
                "group": bv.group,
                "run_dir": run_dir,
                "login": bv.login,
                "env_json": env_payload,
            },
        )
        state.update(state_path, lambda s: s.upsert_server(server))
        reporter.finish(PhaseStatus.DONE, detail=endpoint)
        return server
    except BaseException:
        # Codex fix: any failure after bsub accept must bkill the orphan job.
        # We keep the pending state record so `mcode launch status` still
        # shows the attempt, but the job itself is no longer consuming GPUs.
        _tear_down()
        raise


def _remote_host_file(ssh: SshClient, run_dir: str) -> str | None:
    _require_safe("run_dir", run_dir, _SAFE_POSIX_PATH_RE)
    path = f"{run_dir}/vllm_host.txt"
    r = ssh.run(f"test -f {_q(path)} && cat {_q(path)} || true", timeout=10)
    if not r.ok:
        return None
    s = r.stdout.strip()
    return s or None


def _remote_log_tail(ssh: SshClient, run_dir: str, *, lines: int = 5) -> str:
    _require_safe("run_dir", run_dir, _SAFE_POSIX_PATH_RE)
    if not isinstance(lines, int) or lines < 1 or lines > 10000:
        lines = 5
    path = f"{run_dir}/vllm.log"
    r = ssh.run(f"test -f {_q(path)} && tail -n {lines} {_q(path)} || true", timeout=15)
    return r.stdout.strip() if r.ok else ""


def _http_health(ssh: SshClient, host: str, port: int) -> tuple[bool, int]:
    """Run curl remotely against the vLLM endpoint — from the login node, the
    compute node is reachable, but from our workstation it isn't.
    """
    # Hostname comes from a remote-written file (vllm_host.txt) — sanitize it
    # before it enters a shell command. Real compute-node names match this.
    _HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,253}$")
    if not _HOSTNAME_RE.match(host):
        return False, 0
    if not (0 < port < 65536):
        return False, 0
    url = f"http://{host}:{port}/v1/models"
    cmd = f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 {_q(url)}"
    r = ssh.run(cmd, timeout=15)
    if not r.ok:
        return False, 0
    try:
        code = int(r.stdout.strip())
    except ValueError:
        return False, 0
    return (200 <= code < 300), code


# ---------------------------------------------------------------------------
# control plane
# ---------------------------------------------------------------------------
def stop(
    server_id: str,
    *,
    cfg: LaunchConfig | None = None,
    state_path: Path | None = None,
    ssh_client: SshClient | None = None,
) -> bool:
    """Codex final-review fix: route through the login the record was created
    with (not the current config), and do NOT drop the state record if the
    bkill couldn't confirm — that would strand the live job on the cluster
    with no local handle to stop it.
    """
    srv = state.load(state_path).server(server_id)
    if srv is None:
        return False

    # Prefer the login captured at launch time; fall back to current cfg.
    record_login = (srv.metadata or {}).get("login") or ((cfg or LaunchConfig()).bluevela.login)
    if not record_login:
        # No way to reach the cluster; keep the record for later retry.
        return False
    ssh = ssh_client or SshClient(record_login)

    # If there's no job id we have nothing remote to kill — safe to drop.
    if not srv.job_id:
        state.update(state_path, lambda s: _drop_server(s, server_id))
        return True

    # Poisoned job_id: never touch the cluster, but drop the stale record so
    # `--all` cleanup doesn't keep tripping on it.
    if not _SAFE_DIGITS_RE.match(srv.job_id):
        state.update(state_path, lambda s: _drop_server(s, server_id))
        return True

    try:
        ssh.run(f"bkill {_q(srv.job_id)} 2>&1 || true", timeout=30)
    except TransportError:
        # Transport unreachable — preserve the record so the user can retry
        # when connectivity is back. Mark it stop-pending so status reflects
        # the in-flight intent.
        def _mark_pending_stop(s: state.State) -> None:
            entry = s.server(server_id)
            if entry is not None:
                entry.status = "stop-pending"
                s.upsert_server(entry)

        state.update(state_path, _mark_pending_stop)
        return False
    # Remote kill attempt made (the `|| true` means we don't verify bkill's
    # exit). Drop the record; the LSF job will terminate asynchronously.
    state.update(state_path, lambda s: _drop_server(s, server_id))
    return True


def _drop_server(s: state.State, server_id: str) -> None:
    s.servers = [x for x in s.servers if x.id != server_id]


def refresh(
    record: ServerRecord | RunRecord,
    *,
    cfg: LaunchConfig | None = None,
    ssh_client: SshClient | None = None,
) -> ServerRecord | RunRecord:
    """Codex fix: LSF `RUN` does NOT imply readiness. Only mark healthy when
    both LSF says RUN *and* the HTTP endpoint responds 200 (same contract as
    launch()'s ready phase). Anything in between stays `pending`.

    Final-review fix: use the login captured on the record, not the mutable
    current config. A user whose current config points at a different host
    still gets correct per-record refresh.
    """
    login = None
    if isinstance(record, ServerRecord):
        login = (record.metadata or {}).get("login")
    if not login:
        login = (cfg or LaunchConfig()).bluevela.login
    ssh = ssh_client or SshClient(login) if login else ssh_client
    if not isinstance(record, ServerRecord) or not record.job_id:
        return record
    if not _SAFE_DIGITS_RE.match(record.job_id):
        return record
    try:
        stat = _bjobs_state(ssh, record.job_id)
    except TransportError:
        return record  # can't verify; leave the record alone
    if stat is None:
        record.status = "stopped"
    elif stat.upper() == "EXIT":
        record.status = "failed"
    elif stat.upper() == "RUN":
        # Verify the endpoint is ACTUALLY serving before upgrading to healthy.
        host: str | None = None
        if record.endpoint.startswith("http://"):
            try:
                host = record.endpoint.split("//", 1)[1].split(":", 1)[0]
            except IndexError:
                host = None
        if host:
            try:
                ok, _ = _http_health(ssh, host, _DEFAULT_VLLM_PORT)
                record.status = "healthy" if ok else "pending"
            except TransportError:
                pass  # leave status alone
        else:
            # No endpoint recorded yet (e.g. still in starting phase from a
            # crashed launch): LSF RUN alone is just `pending`.
            record.status = "pending"
    elif stat.upper() in ("PEND", "PSUSP", "USUSP"):
        record.status = "pending"
    return record


def doctor(cfg: LaunchConfig | None = None, *, ssh_client: SshClient | None = None) -> list[Check]:
    cfg = cfg or LaunchConfig()
    bv = cfg.bluevela
    checks: list[Check] = []

    # Config sanity (offline).
    cfg_errs = validate_for_bluevela(cfg)
    checks.append(
        Check(
            name="config complete",
            ok=not cfg_errs,
            detail="ok" if not cfg_errs else "; ".join(cfg_errs),
            next=("" if not cfg_errs else "run `mcode launch doctor bluevela --init`"),
        )
    )
    if cfg_errs:
        return checks

    # SSH reachable.
    ssh = ssh_client or SshClient(bv.login)
    try:
        r = ssh.run("lsid 2>&1 || true", timeout=15)
        # First non-blank line of `lsid` carries the LSF version; the rest is
        # an IBM copyright banner we don't need cluttering the check row.
        first_line = next((ln for ln in (r.stdout or r.stderr).splitlines() if ln.strip()), "")
        checks.append(
            Check(
                name="ssh reachable",
                ok=r.ok,
                detail=first_line.strip()[:120],
                next=("" if r.ok else _hint_for(r.stderr or "")),
            )
        )
    except TransportError as e:
        checks.append(
            Check(
                name="ssh reachable",
                ok=False,
                detail=str(e)[:120],
                next=_hint_for(str(e)),
            )
        )
        return checks

    # Group membership — use the same whole-word filter as doctor_init so a
    # row like `lsfadmins user1 user2 ... ( admin )` doesn't land in the
    # membership list when the configured user isn't actually in it.
    user = bv.login.split("@", 1)[0]
    try:
        r = ssh.run("bugroup 2>/dev/null || true", timeout=15)
        groups = _parse_bugroup(r.stdout or "", user=user)
        has = bv.group in groups
        checks.append(
            Check(
                name=f"member of {bv.group}",
                ok=has,
                detail=", ".join(groups) or "(none)",
                next=(
                    ""
                    if has
                    else (
                        f"ask admin to add you to {bv.group}, or switch "
                        f"[bluevela].group to one of: {', '.join(groups) or '(none found)'}"
                    )
                ),
            )
        )
    except TransportError as e:
        checks.append(Check(name="bugroup probe", ok=False, detail=str(e)))

    # Queue presence.
    try:
        r = ssh.run("bqueues -u $USER -o QUEUE_NAME -noheader 2>&1", timeout=15)
        visible = {line.strip() for line in (r.stdout or "").splitlines() if line.strip()}
        missing = [q for q in bv.queue_order if q not in visible]
        checks.append(
            Check(
                name="queues visible",
                ok=not missing,
                detail=(
                    f"{len(visible)} visible; missing from config: {missing}"
                    if missing
                    else f"{len(visible)} visible"
                ),
                next=("" if not missing else "edit [bluevela].queue_order or re-run --init"),
            )
        )
    except TransportError as e:
        checks.append(Check(name="bqueues probe", ok=False, detail=str(e)))
    return checks


# ---------------------------------------------------------------------------
# doctor --init: bootstrap launch.toml for a fresh Blue Vela account
# ---------------------------------------------------------------------------
def _parse_bugroup(raw: str, *, user: str = "") -> list[str]:
    """Extract groups from `bugroup` output.

    The cluster's `bugroup` (no args) lists ALL groups with their members in
    the format `GROUP_NAME  user1 user2 ... ( admin )`. We want only groups
    that contain `user` — without that filter, admin/catchall groups like
    `lsfadmins` bleed through and doctor-init picks the wrong one.

    An empty `user` returns every well-formed row (internal use only). Member
    match is whole-word, not substring, so `skula` doesn't match `skulapp`.
    """
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("GROUP_NAME"):
            continue
        parts = line.split()
        if not parts:
            continue
        head = parts[0]
        if not _SAFE_IDENT_RE.match(head):
            continue
        if user:
            # Members run from after GROUP_NAME to the opening `(` of the
            # admin column.
            members: list[str] = []
            for tok in parts[1:]:
                if tok.startswith("("):
                    break
                members.append(tok)
            if user not in members:
                continue
        out.append(head)
    return out


def _parse_bqueues(raw: str) -> list[tuple[str, int, str]]:
    """Parse `bqueues -u $USER -o 'QUEUE_NAME PRIO STATUS'` output.

    Returns (name, priority, status) triples, open queues first, sorted by
    priority descending. Queues with names that fail the allowlist are skipped
    (defense in depth — we'll write these to TOML and later feed into bsub).
    """
    rows: list[tuple[str, int, str]] = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts or parts[0] in ("QUEUE_NAME",):
            continue
        if len(parts) < 3:
            continue
        name, prio_raw, status = parts[0], parts[1], parts[2]
        if not _SAFE_IDENT_RE.match(name):
            continue
        try:
            prio = int(prio_raw)
        except ValueError:
            continue
        rows.append((name, prio, status))
    # Open-active queues first (by priority desc), then closed (informational).
    rows.sort(key=lambda r: (not r[2].startswith("Open"), -r[1], r[0]))
    return rows


def doctor_init(
    cfg_path: Path | None = None,
    *,
    login: str,
    ssh_client: SshClient | None = None,
) -> Path:
    """Probe a Blue Vela account and write a launch.toml seeded with real
    values (user's home, groups, open queues). Caller supplies the login
    string since we can't invent a hostname.

    Per M7: strict SSH preflight — a 5s connect test runs first. If it fails,
    the function raises LaunchError with actionable next-steps and does NOT
    attempt probes against a broken connection.

    Per the portability requirement: never writes the developer's username
    into a shared file. The only user-specific values it records are the ones
    the caller's own account actually reports.
    """
    if "@" not in login:
        raise LaunchError(
            what=f"login {login!r} must be user@host",
            why="doctor --init needs to know who and where",
            next="pass --login user@login-host.example",
        )
    ssh = ssh_client or SshClient(login)

    # --- preflight: 5s connect test ---------------------------------------
    try:
        pf = ssh.run("echo mcode-doctor-init-ok", timeout=10)
    except TransportError as e:
        raise LaunchError(
            what="SSH preflight failed",
            why=str(e),
            next=_hint_for(str(e)),
        ) from e
    if not pf.ok or "mcode-doctor-init-ok" not in pf.stdout:
        raise LaunchError(
            what="SSH preflight did not echo the expected marker",
            why=(pf.stderr or pf.stdout).strip()[:200],
            next="verify your shell on the login host isn't printing banners that corrupt stdout",
        )

    # --- home + user ------------------------------------------------------
    home = ssh.run("echo $HOME", timeout=10).stdout.strip() or "$HOME"
    if not _SAFE_POSIX_PATH_RE.match(home):
        raise LaunchError(
            what=f"unexpected $HOME value: {home!r}",
            why="doctor --init needs a POSIX path we can safely write into TOML",
            next="set [bluevela].workspace_root and shared_root manually",
        )

    # --- groups -----------------------------------------------------------
    # bugroup (no args) lists ALL groups + members. Pass the SSH user so
    # _parse_bugroup filters to just this account's groups — without it,
    # catch-all groups like `lsfadmins` win the first-row lottery.
    user = login.split("@", 1)[0]
    bg = ssh.run("bugroup 2>/dev/null || true", timeout=15)
    groups = _parse_bugroup(bg.stdout, user=user)
    group = groups[0] if groups else ""

    # --- queues -----------------------------------------------------------
    bq = ssh.run("bqueues -u $USER -o 'QUEUE_NAME PRIO STATUS' 2>/dev/null", timeout=15)
    queue_rows = _parse_bqueues(bq.stdout)

    # `bqueues -u` doesn't surface the ONLY_INTERACTIVE policy. We probe each
    # candidate with `bqueues -l <q>` so an interactive-only queue like
    # `interactive` doesn't land in queue_order and fail every batch launch.
    # One SSH call per queue, but this runs only at init.
    def _is_batch_queue(q: str) -> bool:
        if not _SAFE_IDENT_RE.match(q):
            return False
        probe = ssh.run(f"bqueues -l {_q(q)} 2>/dev/null", timeout=15)
        if not probe.ok:
            return False
        return "ONLY_INTERACTIVE" not in probe.stdout

    # Drop closed + interactive-only queues.
    open_queues = [name for name, _, status in queue_rows if status.startswith("Open")]
    queue_order = [q for q in open_queues if _is_batch_queue(q)][:3]
    if not queue_order:
        queue_order = ["normal"]

    # --- compose config ---------------------------------------------------
    cfg = config_mod.LaunchConfig()
    cfg.bluevela.login = login
    cfg.bluevela.workspace_root = f"{home}/mcode-launch"
    cfg.bluevela.shared_root = f"{home}/mcode-shared"
    cfg.bluevela.hf_env = f"{home}/.config/mcode/hf-env.sh"
    cfg.bluevela.group = group
    cfg.bluevela.queue_order = queue_order
    cfg.bluevela.gpu_mode = "exclusive_process"  # Phase 0.5 finding

    dst = cfg_path or config_mod.default_config_path()
    config_mod.save(cfg, dst)
    return dst
