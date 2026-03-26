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


def _install_fake_runtime_modules():
    runtime_module = types.ModuleType("mellea.agent.runtime")
    memory_module = types.ModuleType("mellea.agent.runtime.memory")
    workspace_module = types.ModuleType("mellea.agent.runtime.workspace")
    runtime_module.__path__ = []

    class FakeSafetyPolicy:
        def __init__(self, mode, network_access=None, writable_roots=()):
            self.mode = mode
            self.network_access = network_access
            self.writable_roots = tuple(writable_roots)

    class FakeSessionMetadata:
        def __init__(self, session_id=None, executor=None, branch=None, metadata=None):
            self.session_id = session_id
            self.executor = executor
            self.branch = branch
            self.metadata = dict(metadata or {})

    class FakeWorkspace:
        def __init__(self, cwd, safety_policy, session, metadata=None):
            self.cwd = cwd
            self.safety_policy = safety_policy
            self.session = session
            self.metadata = dict(metadata or {})

    class FakeEventLog:
        def __init__(self, *, workspace=None, events=None):
            self.workspace = workspace
            self.events = list(events or [])

    class FakeWorkingMemory:
        def __init__(self, summary="", facts=(), hypotheses=(), next_steps=()):
            self.summary = summary
            self.facts = tuple(facts)
            self.hypotheses = tuple(hypotheses)
            self.next_steps = tuple(next_steps)

    class FakeCondensedState:
        def __init__(
            self,
            working_memory=None,
            recent_messages=(),
            omitted_messages=0,
            show_reminder=None,
        ):
            self.working_memory = working_memory or FakeWorkingMemory()
            self.recent_messages = tuple(recent_messages)
            self.omitted_messages = omitted_messages
            self.show_reminder = omitted_messages > 0 if show_reminder is None else show_reminder

    runtime_module.EventLog = FakeEventLog
    runtime_module.SafetyPolicy = FakeSafetyPolicy
    runtime_module.SessionMetadata = FakeSessionMetadata
    runtime_module.Workspace = FakeWorkspace
    memory_module.CondensedState = FakeCondensedState
    memory_module.WorkingMemory = FakeWorkingMemory
    workspace_module.Workspace = FakeWorkspace
    workspace_module.format_workspace_state = (
        lambda workspace: None
        if workspace is None
        else (
            "Runtime state:\n"
            f"Current working directory: {workspace.cwd}\n"
            "Use relative paths from this directory unless a tool explicitly "
            "requires an absolute path."
        )
    )

    return {
        "mellea.agent.runtime": runtime_module,
        "mellea.agent.runtime.memory": memory_module,
        "mellea.agent.runtime.workspace": workspace_module,
    }


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

    with patch.dict(sys.modules, _install_fake_runtime_modules()), patch(
        "mellea.stdlib.frameworks.react.react", mock_react
    ):
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

    with patch.dict(
        sys.modules,
        {
            **_install_fake_runtime_modules(),
            "mellea.agent.text_react": fake_module,
        },
    ):
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


def test_generate_patch_text_nudge_blocks_empty_diff_final_answer(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    captured: dict = {}

    async def mock_text_react(*args, **kwargs):
        captured.update(kwargs)
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = mock_text_react

    with patch.dict(
        sys.modules,
        {
            **_install_fake_runtime_modules(),
            "mellea.agent.text_react": fake_module,
        },
    ):
        session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds={"test_cmds": ['python -m pytest -q tests/test_bug.py']},
        )

    msgs = captured["on_turn"](7, 9, [])
    assert "working tree still has no code changes" in msgs[-1]["content"]
    assert "Do not call `final_answer` yet" in msgs[-1]["content"]


def test_generate_patch_text_nudge_demands_verification_after_edit(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    captured: dict = {}

    async def mock_text_react(*args, **kwargs):
        captured.update(kwargs)
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = mock_text_react

    with patch.dict(
        sys.modules,
        {
            **_install_fake_runtime_modules(),
            "mellea.agent.text_react": fake_module,
        },
    ):
        session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds={"test_cmds": ['python -m pytest -q tests/test_bug.py']},
        )

    (tmp_path / "foo.py").write_text("x = 2\n")

    msgs = captured["on_turn"](7, 9, [])
    assert "you have not run verification yet" in msgs[-1]["content"]
    assert "Use `run_tests default` now" in msgs[-1]["content"]


def test_generate_patch_exposes_task_default_verification(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    captured: dict = {}

    def fake_make_agent_tools(
        repo_root, *, test_cmds=None, test_fn=None, command_fn=None, workspace=None
    ):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        captured["command_fn"] = command_fn
        captured["workspace"] = workspace
        return []

    mock_result = MagicMock()
    mock_result.value = "done"
    mock_ctx = MagicMock()

    async def mock_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return (mock_result, mock_ctx)

    with patch.dict(sys.modules, _install_fake_runtime_modules()), patch(
        "mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools
    ), patch(
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
    assert captured["test_fn"] is None
    assert captured["command_fn"] is None
    assert captured["workspace"].cwd == str(tmp_path)
    assert "Start with `run_tests default`" in captured["goal"]
    assert "Keep verification cheap" in captured["goal"]


def test_generate_patch_keeps_shell_verification_without_task_defaults(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    captured: dict = {}

    def fake_make_agent_tools(
        repo_root, *, test_cmds=None, test_fn=None, command_fn=None, workspace=None
    ):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        captured["command_fn"] = command_fn
        captured["workspace"] = workspace
        return []

    mock_result = MagicMock()
    mock_result.value = "done"
    mock_ctx = MagicMock()

    async def mock_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return (mock_result, mock_ctx)

    with patch.dict(sys.modules, _install_fake_runtime_modules()), patch(
        "mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools
    ), patch(
        "mellea.stdlib.frameworks.react.react", mock_react
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert isinstance(result, str)
    assert captured["test_cmds"] == []
    assert captured["test_fn"] is None
    assert captured["command_fn"] is None
    assert captured["workspace"].cwd == str(tmp_path)
    assert "cheapest shell command" in captured["goal"]
    assert "Avoid full-suite runs unless necessary" in captured["goal"]


def test_swebench_live_runner_passes_task_metadata_to_generate_patch(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process", n_samples=3),
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
        raw_instance = {"test_cmds": test_cmds}

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
    assert captured["test_cmds"] == FakeTask.raw_instance
    assert captured["n_samples"] == 3
    assert captured["repo"] == FakeTask.repo
    assert captured.get("command_fn") is None


def test_swebench_lite_runner_passes_task_metadata_to_generate_patch(tmp_path):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process", n_samples=3),
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
    assert captured["n_samples"] == 3
    assert captured.get("command_fn") is None


def test_swebench_lite_runner_passes_repo_command_executor_to_generate_patch(tmp_path):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process", n_samples=3),
        results_db=db,
    )

    captured: dict = {}

    def fake_generate_patch(**kwargs):
        captured.update(kwargs)
        return ""

    def fake_command(command: str) -> str:
        return f"ran {command}"

    @contextmanager
    def fake_open():
        yield runner.llm

    class FakeRepoContext:
        def __init__(self):
            self.repo_root = tmp_path
            self.command_fn = fake_command
            self.visible_repo_root = "/testbed"

        def __fspath__(self):
            return str(self.repo_root)

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
            yield FakeRepoContext()

    runner.llm.open = fake_open  # type: ignore[method-assign]
    runner.llm.generate_patch = fake_generate_patch  # type: ignore[method-assign]

    runner._run_swebench_task(
        FakeTask(),
        swe_sandbox=FakeSWEbenchSandbox(),
        run_id=1,
    )

    assert captured["command_fn"] is fake_command
    assert captured["visible_repo_root"] == "/testbed"


def test_swebench_lite_runner_reports_repo_context_failures(tmp_path):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", sandbox="process"),
        results_db=db,
    )

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
            raise RuntimeError("repo setup failed")
            yield tmp_path

    result = runner._run_swebench_task(
        FakeTask(),
        swe_sandbox=FakeSWEbenchSandbox(),
        run_id=1,
    )

    assert result["passed"] is False
    assert result["error"] == "RuntimeError: repo setup failed"
    assert "repo setup failed" in result["stderr"]
