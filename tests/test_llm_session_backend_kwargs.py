from __future__ import annotations

from mcode.llm.session import LLMSession


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

