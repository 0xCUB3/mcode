from __future__ import annotations

import os
import subprocess
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from mellea.backends import ModelOption

from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.llm.session import LLMSession


def _init_repo(tmp_path):
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


def test_generate_patch_uses_react(tmp_path):
    _init_repo(tmp_path)

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
    _init_repo(tmp_path)

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


def test_generate_patch_exposes_task_default_verification(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    captured: dict = {}

    def fake_make_agent_tools(repo_root, *, test_cmds=None, test_fn=None):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        return []

    mock_result = MagicMock()
    mock_result.value = "done"
    mock_ctx = MagicMock()

    async def mock_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return (mock_result, mock_ctx)

    with patch("mellea.agent.tools.make_agent_tools", fake_make_agent_tools), patch(
        "mellea.stdlib.frameworks.react.react", mock_react
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds={
                "test_cmds": [f'{sys.executable} -c "print(\'default verification\')"']
            },
        )

    assert isinstance(result, str)
    assert captured["test_cmds"] == [
        f'{sys.executable} -c "print(\'default verification\')"'
    ]
    assert captured["test_fn"] is not None
    assert "Start with `run_tests default`" in captured["goal"]
    assert "Keep verification cheap" in captured["goal"]
    output = captured["test_fn"]("default")
    assert "default verification" in output
    assert "PASSED" in output


def test_generate_patch_keeps_shell_verification_without_task_defaults(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    captured: dict = {}

    def fake_make_agent_tools(repo_root, *, test_cmds=None, test_fn=None):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        return []

    mock_result = MagicMock()
    mock_result.value = "done"
    mock_ctx = MagicMock()

    async def mock_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return (mock_result, mock_ctx)

    with patch("mellea.agent.tools.make_agent_tools", fake_make_agent_tools), patch(
        "mellea.stdlib.frameworks.react.react", mock_react
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert isinstance(result, str)
    assert captured["test_cmds"] == []
    assert captured["test_fn"] is not None
    assert "cheapest shell command" in captured["goal"]
    assert "Avoid full-suite runs unless necessary." in captured["goal"]
    output = captured["test_fn"](f'{sys.executable} -c "print(\'shell verification\')"')
    assert "shell verification" in output
    assert "PASSED" in output
    assert "No task-default verification commands available" in captured["test_fn"]("default")


def test_swebench_live_runner_passes_dataset_test_cmds(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process"),
        results_db=db,
    )

    captured: dict = {}

    def fake_generate_patch(**kwargs):
        captured.update(kwargs)
        return ""

    @contextmanager
    def fake_open():
        yield runner.llm

    class FakeTask:
        instance_id = "live-1"
        repo = "test/repo"
        problem_statement = "Fix the bug"
        hints_text = "Hint"
        test_cmds = [f'{sys.executable} -c "print(\'live verification\')"']

    class FakeLiveSandbox:
        @contextmanager
        def repo_context(self, task):
            yield tmp_path

    runner.llm.open = fake_open  # type: ignore[method-assign]
    runner.llm.generate_patch = fake_generate_patch  # type: ignore[method-assign]

    result = runner._run_swebench_live_task(
        FakeTask(),
        live_sandbox=FakeLiveSandbox(),
        run_id=1,
    )

    assert result["passed"] is False
    assert captured["test_cmds"] == FakeTask.test_cmds
    assert captured["repo"] == FakeTask.repo


def test_swebench_lite_runner_does_not_require_metadata(tmp_path):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process"),
        results_db=db,
    )

    captured: dict = {}

    def fake_generate_patch(**kwargs):
        captured.update(kwargs)
        return ""

    @contextmanager
    def fake_open():
        yield runner.llm

    class FakeTask:
        benchmark = "swebench-lite"
        instance_id = "lite-1"
        repo = "test/repo"
        problem_statement = "Fix the bug"
        hints_text = "Hint"
        raw_instance = {}

    class FakeSWEbenchSandbox:
        @contextmanager
        def repo_context(self, task):
            yield tmp_path

    runner.llm.open = fake_open  # type: ignore[method-assign]
    runner.llm.generate_patch = fake_generate_patch  # type: ignore[method-assign]

    result = runner._run_swebench_task(
        FakeTask(),
        swe_sandbox=FakeSWEbenchSandbox(),
        run_id=1,
    )

    assert result["passed"] is False
    assert captured["test_cmds"] == FakeTask.raw_instance
