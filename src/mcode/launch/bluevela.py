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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

_STARTUP_ABSOLUTE_DEADLINE_S = 2400  # 40 min from bsub accept to HTTP 200
# 40 min not 30: the per-job podman graphroot (in bluevela_vllm.sh) trades
# cross-run image caching for reliability — every launch cold-pulls the vLLM
# container (~5-10 min on this cluster) before model load. The old 1800s
# budget left too little slack for model load + warmup under load.
_HOST_FILE_DEADLINE_S = 3600  # 1 h for LSF to start the job + write host file
_HEALTH_POLL_SLOW_S = 10.0
_DEFAULT_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.17.0"
_DEFAULT_VLLM_PORT = 8321

_SCRIPTS_DIR = Path(__file__).parent / "scripts"
_RESOURCES_DIR = Path(__file__).parent / "resources"

_MAX_SSH_FAILS = 5  # consecutive transport errors before we give up on a phase


def _absorb_ssh_blip(streak: int, exc: TransportError, *, phase: str) -> int:
    """Update SSH-fail streak after a TransportError during a polling loop.

    Returns the new streak count (caller stores it). Sleeps with exponential
    backoff bounded at 30s. Raises LaunchError when the streak exceeds
    `_MAX_SSH_FAILS` so the caller can stop polling immediately.

    Used by the queued and starting phases of `launch()` — both polled
    `_bjobs_state` / `_remote_host_file` against the cluster login and used
    to duplicate this exact streak/sleep/raise pattern inline.
    """
    streak += 1
    if streak >= _MAX_SSH_FAILS:
        raise LaunchError(
            what=f"lost SSH during {phase}",
            why=f"{_MAX_SSH_FAILS} consecutive ssh failures: {exc}",
            next=_hint_for(str(exc)),
        ) from exc
    time.sleep(min(2**streak, 30))
    return streak


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
@dataclass
class _LaunchContext:
    """Shared state passed between the four phase functions of `launch()`.

    Mutable. Each phase reads what's been populated by earlier phases and
    writes its own outputs. Defaults to the values produced by `_phase_submit`
    so the queued/starting phases can rely on them as non-None.
    """

    spec: LaunchSpec
    reporter: Reporter
    ssh: SshClient
    cfg: LaunchConfig
    state_path: Path | None
    run_id: str
    run_dir: str
    local_log: Path
    queue_chosen: str | None = None
    job_id: str = ""
    env_payload: dict[str, Any] = field(default_factory=dict)
    host: str | None = None


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
    ctx = _LaunchContext(
        spec=spec,
        reporter=reporter,
        ssh=ssh,
        cfg=cfg,
        state_path=state_path,
        run_id=run_id,
        run_dir=_shared_path(bv, "runs", run_id),
        local_log=Path(f"/tmp/mcode-bluevela-{run_id}.log"),
    )
    reporter.add_phases(PHASES)

    # `_tear_down` is registered BEFORE `_phase_submit` runs so it covers the
    # full window from bsub accept through ready: if `_parse_job_id`,
    # `_require_safe`, `state.update`, or any later phase raises after bsub
    # has succeeded, the orphan LSF job still gets bkill'd. The job-id check
    # makes the cleanup a no-op when bsub never accepted.
    def _tear_down() -> None:
        if not ctx.job_id:
            return
        try:
            ssh.run(f"bkill {_q(ctx.job_id)} >/dev/null 2>&1 || true", timeout=15)
        except Exception:
            pass

    try:
        _phase_submit(ctx)
        _phase_queued(ctx)
        _phase_starting(ctx)
        return _phase_ready(ctx)
    except BaseException:
        _tear_down()
        raise


def _phase_submit(ctx: _LaunchContext) -> None:
    """Pick a queue, upload script + env.json, bsub. On success ctx.job_id and
    ctx.queue_chosen are set, and a pending ServerRecord is persisted so the
    caller can bkill the job if a later phase fails."""
    bv = ctx.cfg.bluevela

    def submit_feed() -> str:
        return f"validating queues: {', '.join(bv.queue_order)}"

    ctx.reporter.start("submit", feed=submit_feed)
    try:
        queue = _pick_queue(ctx.ssh, bv)
        ctx.queue_chosen = queue
    except TransportError as e:
        ctx.reporter.finish(PhaseStatus.FAILED, detail=str(e))
        raise LaunchError(
            what="cannot reach Blue Vela",
            why=str(e),
            next=_hint_for(str(e)),
        ) from e

    # Defense in depth: validate everything we'll interpolate into remote
    # shell commands before we touch the cluster.
    _require_safe("queue", queue, _SAFE_IDENT_RE)
    _require_safe("group", bv.group, _SAFE_IDENT_RE)
    _require_safe("gpu_mode", bv.gpu_mode, _SAFE_IDENT_RE)
    _require_safe("run_dir", ctx.run_dir, _SAFE_POSIX_PATH_RE)
    tp = ctx.spec.profile.tensor_parallel
    if not isinstance(tp, int) or tp <= 0 or tp > 32:
        raise LaunchError(
            what=f"profile.tensor_parallel out of range: {tp!r}",
            why="must be a positive int ≤ 32",
            next="fix the profile in launch/profiles.py",
        )

    env_payload = build_env_json(ctx.spec, bv, run_dir=ctx.run_dir)
    env_payload["QUEUE"] = queue
    ctx.env_payload = env_payload

    staging = Path(f"/tmp/mcode-bv-stage-{ctx.run_id}")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "env.json").write_text(json.dumps(env_payload, indent=2))

    ctx.ssh.run(f"mkdir -p {_q(ctx.run_dir)}", timeout=30)
    ctx.ssh.upload(_SCRIPTS_DIR / "bluevela_vllm.sh", f"{ctx.run_dir}/vllm.sh", timeout=60)
    ctx.ssh.upload(staging / "env.json", f"{ctx.run_dir}/env.json", timeout=60)
    if ctx.spec.profile.chat_template:
        tmpl = _RESOURCES_DIR / ctx.spec.profile.chat_template
        if not tmpl.exists():
            raise LaunchError(
                what=f"chat template {ctx.spec.profile.chat_template!r} missing locally",
                why=f"expected at {tmpl}",
                next="add the template to src/mcode/launch/resources/ and retry",
            )
        remote_tmpl = _shared_path(bv, "templates")
        _require_safe("templates_dir", remote_tmpl, _SAFE_POSIX_PATH_RE)
        ctx.ssh.run(f"mkdir -p {_q(remote_tmpl)}", timeout=30)
        ctx.ssh.upload(tmpl, f"{remote_tmpl}/{ctx.spec.profile.chat_template}", timeout=60)

    bsub_cmd = (
        f"bsub -G {_q(bv.group)} -q {_q(queue)} "
        f"-J {_q(f'mcode-vllm-{ctx.run_id}')} -n 1 -R {_q('span[hosts=1]')} "
        f"-gpu {_q(f'num={tp}:mode={bv.gpu_mode}')} "
        f"-o {_q(f'{ctx.run_dir}/vllm.log')} -e {_q(f'{ctx.run_dir}/vllm.log')} "
        f"bash {_q(f'{ctx.run_dir}/vllm.sh')}"
    )
    r = ctx.ssh.run(bsub_cmd, timeout=60)
    if not r.ok:
        ctx.reporter.finish(PhaseStatus.FAILED, detail=(r.stderr or "bsub failed").strip()[:80])
        raise LaunchError(
            what="bsub rejected the submission",
            why=(r.stderr or r.stdout).strip()[:400],
            next=_hint_for(r.stderr or r.stdout),
            logs=str(ctx.local_log),
        )
    # Set ctx.job_id BEFORE _require_safe so the outer _tear_down has a valid
    # bkill target if the safety check itself rejects the parsed id.
    ctx.job_id = _parse_job_id(r.stdout + r.stderr)
    _require_safe("job_id", ctx.job_id, _SAFE_DIGITS_RE)
    ctx.reporter.finish(PhaseStatus.DONE, detail=f"job {ctx.job_id} in queue {queue}")

    # Persist a pending ServerRecord immediately so the caller can find this
    # job for bkill purposes if any later phase fails.
    pending = ServerRecord(
        id=f"server-{ctx.run_id}",
        target=Target.BLUEVELA,
        endpoint="",
        model=ctx.spec.model,
        config_hash=_config_hash(ctx.spec),
        job_id=ctx.job_id,
        log_path=f"{ctx.run_dir}/vllm.log",
        started_at=_now_iso(),
        status="pending",
        metadata={
            "queue": queue,
            "group": bv.group,
            "run_dir": ctx.run_dir,
            "login": bv.login,
        },
    )
    state.update(ctx.state_path, lambda s: s.upsert_server(pending))


def _phase_queued(ctx: _LaunchContext) -> None:
    """Poll bjobs until LSF transitions to RUN. Terminal LSF states
    (DONE/EXIT/PSUSP/USUSP) before the endpoint exists are treated as fatal
    so we surface them in seconds, not the full host_file deadline."""
    bv = ctx.cfg.bluevela

    def queued_feed() -> str:
        try:
            stat = _bjobs_state(ctx.ssh, ctx.job_id)
        except TransportError as e:
            raise TransportError(str(e)) from e
        return f"LSF state: {stat or '?'}"

    ctx.reporter.start("queued", feed=queued_feed, mode="slow")
    host_deadline = time.monotonic() + _HOST_FILE_DEADLINE_S
    ssh_fail_streak = 0
    while True:
        try:
            stat = _bjobs_state(ctx.ssh, ctx.job_id)
            ssh_fail_streak = 0
        except TransportError as e:
            ssh_fail_streak = _absorb_ssh_blip(ssh_fail_streak, e, phase="queue wait")
            continue
        if stat is None or stat.upper() == "RUN":
            break
        if stat.upper() in ("DONE", "EXIT", "PSUSP", "USUSP"):
            tail = _remote_log_tail(ctx.ssh, ctx.run_dir)
            ctx.reporter.finish(PhaseStatus.FAILED, detail=f"LSF {stat}")
            raise LaunchError(
                what=f"LSF job reached {stat} before running",
                why=tail or f"bjobs state: {stat}",
                next=_hint_for(tail or stat),
                logs=f"ssh {bv.login} tail -n 50 {ctx.run_dir}/vllm.log",
            )
        if time.monotonic() > host_deadline:
            ctx.reporter.finish(PhaseStatus.FAILED, detail="queued too long")
            raise LaunchError(
                what=f"LSF job stayed in {stat} past {_HOST_FILE_DEADLINE_S}s",
                why="queue backlog",
                next="try a different queue (edit [bluevela].queue_order) or retry later",
            )
        time.sleep(_HEALTH_POLL_SLOW_S)
    ctx.reporter.finish(PhaseStatus.DONE, detail=f"job {ctx.job_id} running")


def _phase_starting(ctx: _LaunchContext) -> None:
    """Poll the host file and HTTP /v1/models until the endpoint is healthy
    or the LSF job exits. ctx.host is populated on success."""
    bv = ctx.cfg.bluevela

    def starting_feed() -> str:
        tail = _remote_log_tail(ctx.ssh, ctx.run_dir, lines=1)
        host_inner = _remote_host_file(ctx.ssh, ctx.run_dir)
        where = f"host {host_inner}" if host_inner else "waiting for host file"
        return f"{where} · {tail}" if tail else where

    ctx.reporter.start("starting", feed=starting_feed, mode="slow")
    deadline = time.monotonic() + _STARTUP_ABSOLUTE_DEADLINE_S
    starting_fail_streak = 0
    while True:
        try:
            host = _remote_host_file(ctx.ssh, ctx.run_dir)
            if host:
                ok, _status = _http_health(ctx.ssh, host, _DEFAULT_VLLM_PORT)
                if ok:
                    ctx.host = host
                    break
            starting_fail_streak = 0
        except TransportError as e:
            starting_fail_streak = _absorb_ssh_blip(starting_fail_streak, e, phase="startup")
            continue
        if _bjobs_state(ctx.ssh, ctx.job_id) in (None, "EXIT", "DONE"):
            tail = _remote_log_tail(ctx.ssh, ctx.run_dir)
            ctx.reporter.finish(PhaseStatus.FAILED, detail="job exited early")
            raise LaunchError(
                what="vLLM job exited before endpoint became healthy",
                why=tail or "LSF reports EXIT",
                next=_hint_for(tail),
                logs=f"ssh {bv.login} tail -n 100 {ctx.run_dir}/vllm.log",
            )
        if time.monotonic() > deadline:
            ctx.reporter.finish(PhaseStatus.FAILED, detail="startup deadline exceeded")
            raise LaunchError(
                what=f"server did not become ready within {_STARTUP_ABSOLUTE_DEADLINE_S}s",
                why=_remote_log_tail(ctx.ssh, ctx.run_dir) or "no progress",
                next=(
                    "check log for OOM / chat-template / parser issues; "
                    "reduce max_model_len or TP in the profile"
                ),
                logs=f"ssh {bv.login} tail -n 200 {ctx.run_dir}/vllm.log",
            )
        time.sleep(_HEALTH_POLL_SLOW_S)
    ctx.reporter.finish(PhaseStatus.DONE, detail=f"host {ctx.host}")


def _phase_ready(ctx: _LaunchContext) -> ServerRecord:
    """Construct and persist the final healthy ServerRecord, return it."""
    bv = ctx.cfg.bluevela
    ctx.reporter.start("ready")
    endpoint = f"http://{ctx.host}:{_DEFAULT_VLLM_PORT}/v1"
    server = ServerRecord(
        id=f"server-{ctx.run_id}",
        target=Target.BLUEVELA,
        endpoint=endpoint,
        model=ctx.spec.model,
        config_hash=_config_hash(ctx.spec),
        job_id=ctx.job_id,
        log_path=f"{ctx.run_dir}/vllm.log",
        started_at=_now_iso(),
        status="healthy",
        metadata={
            "queue": ctx.queue_chosen,
            "group": bv.group,
            "run_dir": ctx.run_dir,
            "login": bv.login,
            "env_json": ctx.env_payload,
        },
    )
    state.update(ctx.state_path, lambda s: s.upsert_server(server))
    ctx.reporter.finish(PhaseStatus.DONE, detail=endpoint)
    return server


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

    # Codex pre-merge-review fix: don't `|| true` the bkill — inspect it.
    # If bkill succeeded OR the job is already absent from LSF, drop the
    # record. If bkill failed for any other reason (permission denied,
    # transient LSF hiccup), preserve the record as stop-pending so the
    # user can retry. Otherwise we lose the only local handle to a live job.
    def _mark_pending_stop(s: state.State) -> None:
        entry = s.server(server_id)
        if entry is not None:
            entry.status = "stop-pending"
            s.upsert_server(entry)

    try:
        r = ssh.run(f"bkill {_q(srv.job_id)}", timeout=30)
    except TransportError:
        state.update(state_path, _mark_pending_stop)
        return False

    kill_confirmed = r.ok or any(
        phrase in (r.stdout + r.stderr).lower()
        for phrase in (
            "is being terminated",
            "already finished",
            "job not found",
            "job has already finished",
        )
    )
    if not kill_confirmed:
        # bkill returned non-zero and stderr didn't match "already gone"
        # phrases — assume the job is still alive. Preserve record.
        state.update(state_path, _mark_pending_stop)
        return False

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

    # Wrap any post-preflight TransportError in LaunchError so the CLI
    # renders the formatted ✗/why/next layout instead of a traceback
    # (Codex final-verify-pass hardening).
    def _probe(cmd: str, *, timeout: float = 15.0):
        try:
            return ssh.run(cmd, timeout=timeout)
        except TransportError as e:
            raise LaunchError(
                what="SSH dropped mid-probe",
                why=str(e),
                next=_hint_for(str(e)),
            ) from e

    # --- home + user ------------------------------------------------------
    home = _probe("echo $HOME", timeout=10).stdout.strip() or "$HOME"
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
    bg = _probe("bugroup 2>/dev/null || true")
    groups = _parse_bugroup(bg.stdout, user=user)
    group = groups[0] if groups else ""

    # --- queues -----------------------------------------------------------
    bq = _probe("bqueues -u $USER -o 'QUEUE_NAME PRIO STATUS' 2>/dev/null")
    queue_rows = _parse_bqueues(bq.stdout)

    # `bqueues -u` doesn't surface the ONLY_INTERACTIVE policy. We probe
    # each candidate with `bqueues -l <q>` so an interactive-only queue
    # doesn't land in queue_order and fail every batch launch.
    #
    # Codex final-review fix: tri-state result. True = confirmed batch-OK,
    # False = confirmed interactive-only, None = probe failed (transport,
    # timeout, unknown status). Fail closed if ALL probes are None —
    # silently writing queue_order=["normal"] under those conditions risks
    # shipping a bad config that only fails at submit time.
    def _is_batch_queue(q: str) -> bool | None:
        if not _SAFE_IDENT_RE.match(q):
            return None
        try:
            probe = ssh.run(f"bqueues -l {_q(q)} 2>/dev/null", timeout=60)
        except TransportError:
            # Transport dropping mid-init must NOT raise out of doctor_init;
            # let the fail-closed aggregation path raise a formatted
            # LaunchError instead (Codex pre-merge verification fix).
            return None
        if not probe.ok or not probe.stdout:
            return None
        return "ONLY_INTERACTIVE" not in probe.stdout

    open_queues = [name for name, _, status in queue_rows if status.startswith("Open")]
    confirmed_batch: list[str] = []
    probe_failed: list[str] = []
    for q in open_queues:
        verdict = _is_batch_queue(q)
        if verdict is True:
            confirmed_batch.append(q)
        elif verdict is None:
            probe_failed.append(q)
        # False -> interactive, drop silently
    queue_order = confirmed_batch[:3]
    if not queue_order:
        # Codex pre-merge-review fix: fail closed unless at least one queue
        # was positively confirmed batch-capable. This covers three cases:
        # (a) all probes errored → transient issue, retry
        # (b) all open queues are interactive-only → user needs a group /
        #     project that unlocks a batch queue
        # (c) bqueues -u returned nothing parseable → cluster state issue
        # Previously we silently wrote queue_order=["normal"] which shipped
        # a plausible config that only failed at submit time.
        raise LaunchError(
            what="could not confirm any batch-capable queue",
            why=(
                f"{len(probe_failed)} queue policy probe(s) failed ({probe_failed!r}); "
                f"remaining open queues are interactive-only or none. "
                f"open={open_queues!r}"
            ),
            next=(
                "retry `doctor bluevela --init` if this looks transient; "
                "otherwise check `bqueues -u $USER` on the cluster and set "
                "[bluevela].queue_order manually in your launch.toml"
            ),
        )

    # --- compose config ---------------------------------------------------
    # Note on shared_root: an earlier iteration auto-preferred
    # `/proj/dmfexp/<user>` to escape home-quota failures from per-host
    # podman graphroots. The bluevela_vllm.sh script now uses per-job
    # graphroots in /tmp instead, so shared_root only carries small
    # artifacts (run dirs, templates, log files) — home is fine for that.
    # Users who want HF cache on a quota-free filesystem can set HF_HOME
    # in their hf-env.sh, independent of shared_root.
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
