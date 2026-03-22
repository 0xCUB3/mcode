from __future__ import annotations

import os
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

from mellea.backends import ModelOption

from mcode.llm.session import LLMSession


def test_generate_patch_uses_react(tmp_path):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        env=env,
    )

    session = LLMSession(model_id="test", backend_name="ollama")

    mock_mellea = MagicMock()
    session._m = mock_mellea

    mock_result = MagicMock()
    mock_result.value = "done"
    mock_ctx = MagicMock()

    async def mock_react(*args, **kwargs):
        return (mock_result, mock_ctx)

    with patch("mellea.stdlib.frameworks.react.react", mock_react):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    assert isinstance(result, str)


def test_generate_patch_passes_model_options_to_text_react(tmp_path, monkeypatch):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        env=env,
    )

    session = LLMSession(
        model_id="test",
        backend_name="openai",
        temperature=0.25,
        seed=7,
        loop_budget=9,
    )
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")
    monkeypatch.setenv("MCODE_MAX_NEW_TOKENS", "123")

    captured: dict = {}

    async def mock_text_react(*args, **kwargs):
        captured.update(kwargs)
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = mock_text_react

    with patch.dict(sys.modules, {"mellea.agent.text_react": fake_module}):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert isinstance(result, str)
    assert captured["loop_budget"] == 9
    assert captured["model_options"] == {
        ModelOption.SYSTEM_PROMPT: captured["system_prompt"],
        ModelOption.TEMPERATURE: 0.25,
        ModelOption.SEED: 7,
        ModelOption.MAX_NEW_TOKENS: 123,
        ModelOption.STREAM: False,
    }
