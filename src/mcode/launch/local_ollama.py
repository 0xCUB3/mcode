"""Local Ollama target.

v1 scope: the `ollama serve` daemon is the user's responsibility (it usually
runs via brew services / systemd / launchctl and stays running). This module
verifies the daemon is reachable, pulls the requested model (with progress),
and records the OpenAI-compatible endpoint.

The "model pull" phase is the real UX win — Ollama's download can take
minutes for large models, and progress is streamable from its API.

Phases:
    check   → ollama daemon reachable
    pull    → model present (skipped if already pulled)
    ready   → /v1/models returns the model

Endpoint: http://<host>:<port>/v1 — Ollama exposes an OpenAI-compatible
surface since 0.1.30+.
"""

from __future__ import annotations

import hashlib
import json
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
    ServerRecord,
    Target,
)
from mcode.launch.progress import _ReporterBase as Reporter

PHASES: list[Phase] = [
    Phase("check", "Ollama daemon reachable"),
    Phase("pull", "Pull model"),
    Phase("ready", "Model loaded"),
]

_PULL_ABSOLUTE_DEADLINE_S = 1800  # 30 min cap on model download


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _base_url(cfg: LaunchConfig) -> str:
    return f"http://{cfg.local_ollama.host}:{cfg.local_ollama.port}"


def _config_hash(spec: LaunchSpec) -> str:
    return hashlib.sha256(json.dumps({"model": spec.model}, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def _get_json(url: str, timeout_s: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
        return None


def _tags(cfg: LaunchConfig) -> list[str]:
    data = _get_json(f"{_base_url(cfg)}/api/tags")
    if not data:
        return []
    return [m.get("name") for m in data.get("models", []) if m.get("name")]


def _normalize_ollama_name(name: str) -> str:
    """Ollama treats `foo` as shorthand for `foo:latest`. Normalize so that
    membership checks against /api/tags don't falsely fail (Codex fix)."""
    return name if ":" in name else f"{name}:latest"


def _model_in_tags(model: str, tags: list[str]) -> bool:
    target = _normalize_ollama_name(model)
    return any(_normalize_ollama_name(t) == target for t in tags)


def _daemon_ok(cfg: LaunchConfig) -> bool:
    return _get_json(f"{_base_url(cfg)}/api/version") is not None


def _v1_models(cfg: LaunchConfig) -> list[str]:
    """GET /v1/models — the actual endpoint the launcher advertises. Used
    in the ready phase so we prove the model is served there, not just that
    `/api/tags` lists it (Codex fix: readiness must match the advertised URL)."""
    data = _get_json(f"{_base_url(cfg)}/v1/models")
    if not data:
        return []
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


def _pull_stream(cfg: LaunchConfig, model: str, deadline: float):
    """Yield (status, completed_bytes, total_bytes) tuples from /api/pull.
    Raises LaunchError on transport or model-not-found errors.
    """
    payload = json.dumps({"name": model, "stream": True}).encode()
    req = urllib.request.Request(
        f"{_base_url(cfg)}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30.0)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise LaunchError(
            what=f"ollama rejected pull of {model!r}",
            why=f"HTTP {e.code}: {detail[:200]}",
            next="check the model name on https://ollama.com/library",
        ) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise LaunchError(
            what="ollama daemon unreachable during pull",
            why=str(e),
            next=f"verify `ollama serve` is running and listening on {_base_url(cfg)}",
        ) from e

    with resp:
        for raw in resp:
            if time.monotonic() > deadline:
                raise LaunchError(
                    what=f"ollama pull exceeded {_PULL_ABSOLUTE_DEADLINE_S}s",
                    why="download is taking too long",
                    next="check your network / ollama logs; retry with a smaller model",
                )
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError as e:
                # Codex fix: don't silently swallow malformed frames — the
                # heartbeat gets stuck on stale progress and the real pull
                # failure is obscured. Surface as LaunchError with context.
                raise LaunchError(
                    what=f"ollama pull produced malformed JSON for {model!r}",
                    why=f"{e}: {line[:200]!r}",
                    next="try `ollama pull` directly to see what the daemon is emitting",
                ) from e
            if err := evt.get("error"):
                raise LaunchError(
                    what=f"ollama pull failed for {model!r}",
                    why=err,
                    next="check the model name on https://ollama.com/library, or `ollama logs`",
                )
            status = evt.get("status", "")
            try:
                completed = int(evt.get("completed", 0) or 0)
                total = int(evt.get("total", 0) or 0)
            except (TypeError, ValueError) as e:
                raise LaunchError(
                    what=f"ollama pull progress frame has non-numeric bytes for {model!r}",
                    why=f"{e}: {evt!r}",
                    next="ollama may have changed its API shape; update mcode",
                ) from e
            yield status, completed, total


def launch(
    spec: LaunchSpec,
    reporter: Reporter,
    *,
    cfg: LaunchConfig | None = None,
    state_path: Path | None = None,
) -> ServerRecord:
    if spec.target != Target.LOCAL_OLLAMA:
        raise LaunchError(
            what="local_ollama.launch called with wrong target",
            why=f"spec.target = {spec.target!r}",
            next="use the bluevela.launch or local_vllm.launch module",
        )
    cfg = cfg or LaunchConfig()

    reporter.add_phases(PHASES)

    # --- check phase -------------------------------------------------------
    with reporter.phase("check", feed=lambda: f"probing {_base_url(cfg)}/api/version"):
        if not _daemon_ok(cfg):
            raise LaunchError(
                what="ollama daemon not reachable",
                why=f"no response from {_base_url(cfg)}/api/version",
                next=(
                    "start it with `ollama serve` (or `brew services start ollama` "
                    "on macOS), then re-run"
                ),
            )

    # --- pull phase --------------------------------------------------------
    # If the model is already present, skip the pull entirely and just show a
    # one-line detail that it was cached. Normalize name to handle `foo` vs
    # `foo:latest` (Codex fix).
    already = _model_in_tags(spec.model, _tags(cfg))
    pull_progress: dict[str, tuple[int, int, str]] = {"last": (0, 0, "queued")}

    def pull_feed() -> str:
        completed, total, status = pull_progress["last"]
        if total > 0:
            pct = (100.0 * completed / total) if total else 0.0
            return (
                f"{status} · {completed // (1024 * 1024)}/{total // (1024 * 1024)} MB ({pct:.0f}%)"
            )
        return status or "working…"

    reporter.start("pull", feed=pull_feed)
    try:
        if already:
            reporter.set_detail("already cached")
        else:
            deadline = time.monotonic() + _PULL_ABSOLUTE_DEADLINE_S
            last_status = "starting"
            for status, completed, total in _pull_stream(cfg, spec.model, deadline):
                pull_progress["last"] = (completed, total, status)
                last_status = status or last_status
            reporter.set_detail(f"done · {last_status}")
    except LaunchError:
        reporter.finish(PhaseStatus.FAILED)
        raise
    else:
        reporter.finish(PhaseStatus.DONE)

    # --- ready phase -------------------------------------------------------
    # Codex fix: prove the model is served at /v1/models (the advertised
    # endpoint), not just that /api/tags lists it. Tags and /v1 can disagree
    # during daemon restarts or manifest shuffles.
    reporter.start("ready")
    served = _v1_models(cfg)
    if not _model_in_tags(spec.model, served):
        reporter.finish(
            PhaseStatus.FAILED,
            detail="model missing from /v1/models",
        )
        raise LaunchError(
            what=f"ollama does not advertise {spec.model!r} at /v1/models after pull",
            why=f"served: {served}",
            next="run `ollama list` to verify the pull, restart `ollama serve`, and retry",
        )
    endpoint = f"{_base_url(cfg)}/v1"
    server = ServerRecord(
        id=f"server-local-ollama-{uuid.uuid4().hex[:8]}",
        target=Target.LOCAL_OLLAMA,
        endpoint=endpoint,
        model=spec.model,
        config_hash=_config_hash(spec),
        job_id=None,  # ollama daemon isn't owned by us
        started_at=_now_iso(),
        status="healthy",
        metadata={"host": cfg.local_ollama.host, "port": cfg.local_ollama.port},
    )
    state.update(state_path, lambda s: s.upsert_server(server))
    reporter.finish(PhaseStatus.DONE, detail=endpoint)
    return server


def doctor(cfg: LaunchConfig | None = None) -> list[Check]:
    cfg = cfg or LaunchConfig()
    checks: list[Check] = []
    ok = _daemon_ok(cfg)
    checks.append(
        Check(
            name=f"ollama daemon at {_base_url(cfg)}",
            ok=ok,
            detail="reachable" if ok else "unreachable",
            next=(
                "" if ok else "start `ollama serve` or set [local_ollama] host/port in your config"
            ),
        )
    )
    return checks


def stop(server_id: str, *, state_path: Path | None = None) -> bool:
    """Remove the server record. We don't own the ollama daemon, so we never
    signal it — just drop our record."""

    def _drop(s: state.State) -> bool:
        if s.server(server_id) is None:
            return False
        s.servers = [x for x in s.servers if x.id != server_id]
        return True

    return bool(state.update(state_path, _drop))


def refresh(record: ServerRecord, cfg: LaunchConfig | None = None) -> ServerRecord:
    cfg = cfg or LaunchConfig()
    if not _daemon_ok(cfg):
        record.status = "stopped"
    elif not _model_in_tags(record.model, _tags(cfg)):
        record.status = "stopped"
    else:
        record.status = "healthy"
    return record
