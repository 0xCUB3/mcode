from __future__ import annotations

import os

from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext

from mcode.launch.models import ServerRecord, Target
from mcode.llm.session import LLMSession, _build_sampling_strategy, _strategy_for_requirements


def test_backend_kwargs_for_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    s = LLMSession(model_id="granite4:latest", backend_name="ollama")
    assert s._backend_kwargs() == {"base_url": "http://ollama:11434"}  # noqa: SLF001


def test_backend_kwargs_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    s = LLMSession(model_id="ibm-granite/granite-3.0-8b-instruct", backend_name="openai")
    assert s._backend_kwargs() == {  # noqa: SLF001
        "base_url": "http://vllm:8000/v1",
        "api_key": "dummy",
    }


def test_backend_kwargs_openai_autoresolves_from_launcher(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = ServerRecord(
        id="srv-1",
        target=Target.BLUEVELA,
        endpoint="http://compute-42:8321/v1",
        model="ibm-granite/granite-4.0-h-small",
        config_hash="deadbeef",
        status="healthy",
    )

    class FakeSnap:
        servers = [server]

    monkeypatch.setattr("mcode.launch.state.load", lambda: FakeSnap())
    s = LLMSession(model_id="ibm-granite/granite-4.0-h-small", backend_name="openai")
    assert s._backend_kwargs() == {  # noqa: SLF001
        "base_url": "http://compute-42:8321/v1",
        "api_key": "dummy",
    }
    assert os.environ.get("OPENAI_BASE_URL") is None
    assert os.environ.get("OPENAI_API_KEY") is None


def test_backend_kwargs_openai_no_autoresolve_when_model_mismatch(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = ServerRecord(
        id="srv-1",
        target=Target.BLUEVELA,
        endpoint="http://compute-42:8321/v1",
        model="some-other-model",
        config_hash="deadbeef",
        status="healthy",
    )

    class FakeSnap:
        servers = [server]

    monkeypatch.setattr("mcode.launch.state.load", lambda: FakeSnap())
    s = LLMSession(model_id="ibm-granite/granite-4.0-h-small", backend_name="openai")
    assert s._backend_kwargs() == {}  # noqa: SLF001


def test_model_options_default_max_new_tokens():
    s = LLMSession(model_id="test-model", backend_name="openai")
    opts = s._model_options(system_prompt="system")  # noqa: SLF001
    assert opts[ModelOption.MAX_NEW_TOKENS] == 1024


def test_model_options_respects_env_max_new_tokens(monkeypatch):
    monkeypatch.setenv("MCODE_MAX_NEW_TOKENS", "2048")
    s = LLMSession(model_id="test-model", backend_name="openai")
    opts = s._model_options(system_prompt="system")  # noqa: SLF001
    assert opts[ModelOption.MAX_NEW_TOKENS] == 2048


def test_build_sampling_strategy_uses_native_multiturn(monkeypatch):
    captured: list[int] = []

    class Strategy:
        def __init__(self, *, loop_budget):
            captured.append(loop_budget)

    monkeypatch.setattr("mcode.llm.session.MultiTurnStrategy", Strategy)

    backend = object()
    _build_sampling_strategy(backend=backend, strategy_name="multiturn", sampling_budget=4)

    assert captured == [4]


def test_strategy_for_requirements_skips_sampling_when_disabled():

    strategy = _strategy_for_requirements(
        backend=object(),
        requirements=[object()],
        strategy_name="none",
        sampling_budget=3,
    )

    assert strategy is None



def test_start_session_uses_chat_context(monkeypatch):
    import mellea

    captured: dict[str, object] = {}

    def fake_start_session(*, backend_name, model_id, ctx=None, plugins=None, **kwargs):
        captured["backend_name"] = backend_name
        captured["model_id"] = model_id
        captured["ctx"] = ctx
        captured["plugins"] = plugins
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(mellea, "start_session", fake_start_session)

    session = LLMSession(model_id="test-model", backend_name="openai")
    session._start_session()

    assert captured["backend_name"] == "openai"
    assert captured["model_id"] == "test-model"
    assert isinstance(captured["ctx"], ChatContext)
