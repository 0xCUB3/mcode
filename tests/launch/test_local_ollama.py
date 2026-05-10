from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcode.launch import local_ollama, profiles, state
from mcode.launch.config import LaunchConfig
from mcode.launch.models import LaunchError, LaunchSpec, ServerRecord, Target
from mcode.launch.progress import NullReporter


def _spec(model: str = "qwen2.5:0.5b") -> LaunchSpec:
    return LaunchSpec(
        target=Target.LOCAL_OLLAMA,
        model=model,
        profile=profiles.resolve(model),
    )


def _url_response(status: int = 200, body: str = ""):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body.encode()
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: None
    return resp


def test_launch_wrong_target_raises() -> None:
    spec = _spec()
    spec.target = Target.LOCAL_VLLM
    with pytest.raises(LaunchError):
        local_ollama.launch(spec, NullReporter.create(local_ollama.PHASES))


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_daemon_unreachable_gives_actionable_error(mock_open) -> None:
    mock_open.side_effect = urllib.error.URLError("connection refused")
    with pytest.raises(LaunchError) as ei:
        local_ollama.launch(_spec(), NullReporter.create(local_ollama.PHASES))
    assert "not reachable" in ei.value.what
    assert "ollama serve" in ei.value.next


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_launch_skips_pull_when_model_already_cached(mock_open, tmp_path: Path) -> None:
    """If the model is already in /api/tags, don't hit /api/pull."""

    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps({"models": [{"name": "qwen2.5:0.5b"}]}))
        if url.endswith("/v1/models"):
            return _url_response(body=json.dumps({"data": [{"id": "qwen2.5:0.5b"}]}))
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    state_path = tmp_path / "state.json"
    server = local_ollama.launch(
        _spec(),
        NullReporter.create(local_ollama.PHASES),
        cfg=LaunchConfig(),
        state_path=state_path,
    )
    assert server.target == Target.LOCAL_OLLAMA
    # Verify /api/pull was NOT called.
    called_urls = []
    for call in mock_open.call_args_list:
        arg = call.args[0]
        called_urls.append(arg if isinstance(arg, str) else arg.full_url)
    assert not any("/api/pull" in u for u in called_urls)
    # Persisted
    assert state.load(state_path).server(server.id) is not None


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_launch_pulls_model_when_missing(mock_open, tmp_path: Path) -> None:
    """Happy path: daemon up, model missing, pull streams progress, tags now
    lists the model, ServerRecord is persisted."""
    tags_sequence = iter(
        [
            # First /api/tags: empty (pull needed)
            {"models": []},
        ]
    )

    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps(next(tags_sequence)))
        if url.endswith("/v1/models"):
            # Regression: readiness checks /v1/models, not /api/tags
            return _url_response(body=json.dumps({"data": [{"id": "qwen2.5:0.5b"}]}))
        if url.endswith("/api/pull"):
            # Return a stream of 3 ndjson events.
            stream = b"\n".join(
                [
                    json.dumps({"status": "pulling", "completed": 50, "total": 100}).encode(),
                    json.dumps({"status": "pulling", "completed": 100, "total": 100}).encode(),
                    json.dumps({"status": "success"}).encode(),
                ]
            )
            resp = MagicMock()
            resp.__enter__ = lambda self: self
            resp.__exit__ = lambda self, *a: None
            resp.__iter__ = lambda self: iter(stream.splitlines(keepends=True))
            return resp
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    state_path = tmp_path / "state.json"
    server = local_ollama.launch(
        _spec(),
        NullReporter.create(local_ollama.PHASES),
        cfg=LaunchConfig(),
        state_path=state_path,
    )
    assert server.model == "qwen2.5:0.5b"
    assert server.endpoint == "http://127.0.0.1:11434/v1"


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_pull_error_surfaced_with_hint(mock_open) -> None:
    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps({"models": []}))
        if url.endswith("/api/pull"):
            stream = json.dumps({"error": "pull model manifest: not found"}).encode() + b"\n"
            resp = MagicMock()
            resp.__enter__ = lambda self: self
            resp.__exit__ = lambda self, *a: None
            resp.__iter__ = lambda self: iter(stream.splitlines(keepends=True))
            return resp
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    with pytest.raises(LaunchError) as ei:
        local_ollama.launch(_spec(), NullReporter.create(local_ollama.PHASES))
    assert "pull failed" in ei.value.what
    assert "ollama.com/library" in ei.value.next


def test_doctor_reports_daemon_reachability() -> None:
    cfg = LaunchConfig()
    with patch("mcode.launch.local_ollama._daemon_ok", return_value=True):
        checks = local_ollama.doctor(cfg)
    assert checks[0].ok is True

    with patch("mcode.launch.local_ollama._daemon_ok", return_value=False):
        checks = local_ollama.doctor(cfg)
    assert checks[0].ok is False
    assert "ollama serve" in checks[0].next


def test_stop_drops_record(tmp_path: Path) -> None:
    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-abc",
        target=Target.LOCAL_OLLAMA,
        endpoint="x",
        model="m",
        config_hash="h",
    )
    state.update(state_path, lambda s: s.upsert_server(server))
    assert local_ollama.stop("server-abc", state_path=state_path) is True
    assert state.load(state_path).server("server-abc") is None
    # Idempotent
    assert local_ollama.stop("server-abc", state_path=state_path) is False


def test_refresh_flips_to_stopped_when_daemon_down() -> None:
    server = ServerRecord(
        id="s",
        target=Target.LOCAL_OLLAMA,
        endpoint="x",
        model="qwen2.5:0.5b",
        config_hash="h",
        status="healthy",
    )
    with patch("mcode.launch.local_ollama._daemon_ok", return_value=False):
        updated = local_ollama.refresh(server)
    assert updated.status == "stopped"


def test_refresh_flips_to_stopped_when_model_missing() -> None:
    server = ServerRecord(
        id="s",
        target=Target.LOCAL_OLLAMA,
        endpoint="x",
        model="qwen2.5:0.5b",
        config_hash="h",
        status="healthy",
    )
    with (
        patch("mcode.launch.local_ollama._daemon_ok", return_value=True),
        patch("mcode.launch.local_ollama._tags", return_value=["other-model:latest"]),
    ):
        updated = local_ollama.refresh(server)
    assert updated.status == "stopped"


def test_model_tag_normalization() -> None:
    """Regression: `foo` must match `foo:latest` and vice versa."""
    assert local_ollama._model_in_tags("granite4", ["granite4:latest"])
    assert local_ollama._model_in_tags("granite4:latest", ["granite4"])
    assert local_ollama._model_in_tags("granite4:3b", ["granite4:3b"])
    assert not local_ollama._model_in_tags("granite4:3b", ["granite4:latest"])
    assert not local_ollama._model_in_tags("granite4", ["gemma4:latest"])


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_launch_uses_latest_alias_in_cache_check(mock_open, tmp_path: Path) -> None:
    """If user passes `granite4` (no tag) and daemon has `granite4:latest`, we
    must detect the cache hit and skip the pull."""

    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps({"models": [{"name": "granite4:latest"}]}))
        if url.endswith("/v1/models"):
            return _url_response(body=json.dumps({"data": [{"id": "granite4:latest"}]}))
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    server = local_ollama.launch(
        _spec("granite4"),
        NullReporter.create(local_ollama.PHASES),
        cfg=LaunchConfig(),
        state_path=tmp_path / "s.json",
    )
    assert server.model == "granite4"
    called_urls = [
        (c.args[0] if isinstance(c.args[0], str) else c.args[0].full_url)
        for c in mock_open.call_args_list
    ]
    assert not any("/api/pull" in u for u in called_urls)


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_malformed_pull_frame_raises_launch_error(mock_open) -> None:
    """Regression: malformed NDJSON frames must produce actionable LaunchError,
    not silently swallowed or raw ValueError."""

    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps({"models": []}))
        if url.endswith("/api/pull"):
            stream = b'this is not json\n{"status": "ok"}\n'
            resp = MagicMock()
            resp.__enter__ = lambda self: self
            resp.__exit__ = lambda self, *a: None
            resp.__iter__ = lambda self: iter(stream.splitlines(keepends=True))
            return resp
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    with pytest.raises(LaunchError) as ei:
        local_ollama.launch(_spec(), NullReporter.create(local_ollama.PHASES))
    assert "malformed JSON" in ei.value.what


@patch("mcode.launch.local_ollama.urllib.request.urlopen")
def test_ready_phase_fails_when_v1_models_lacks_model(mock_open) -> None:
    """Regression: even if /api/tags lists the model, ready-phase must hit
    /v1/models to prove the advertised endpoint actually serves it."""

    def responder(req, timeout=3.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return _url_response(body=json.dumps({"version": "0.1.40"}))
        if url.endswith("/api/tags"):
            return _url_response(body=json.dumps({"models": [{"name": "qwen2.5:0.5b"}]}))
        if url.endswith("/v1/models"):
            return _url_response(body=json.dumps({"data": []}))  # empty!
        raise AssertionError(f"unexpected url {url}")

    mock_open.side_effect = responder
    with pytest.raises(LaunchError) as ei:
        local_ollama.launch(_spec(), NullReporter.create(local_ollama.PHASES))
    assert "/v1/models" in ei.value.what
