from __future__ import annotations

import os
import subprocess
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from mellea.agent.runtime.events import ToolCallEvent, ToolResultEvent
from mellea.backends import ModelOption

from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.execution.sandbox import DockerUnavailableError
from mcode.llm.session import LLMSession


def _install_fake_runtime_modules():
    runtime_module = types.ModuleType("mellea.agent.runtime")
    memory_module = types.ModuleType("mellea.agent.runtime.memory")
    loops_module = types.ModuleType("mellea.agent.runtime.loops")
    workspace_module = types.ModuleType("mellea.agent.runtime.workspace")
    strategy_module = types.ModuleType("mellea.agent.strategy")
    capabilities_module = types.ModuleType("mellea.agent.capabilities")
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

        def emit(self, event):
            self.events.append(event)
            return event

        def to_dicts(self):
            return [
                event.as_dict() if hasattr(event, "as_dict") else dict(event)
                for event in self.events
            ]

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

    class FakeCondensationConfig:
        def __init__(
            self,
            *,
            working_memory,
            max_messages,
            preserve_recent=4,
            preserve_head=2,
        ):
            self.working_memory = working_memory
            self.max_messages = max_messages
            self.preserve_recent = preserve_recent
            self.preserve_head = preserve_head

    class ToolInvocation:
        def __init__(self, name, status="completed"):
            self.name = name
            self.status = status

    class ToolPhaseState:
        def __init__(
            self,
            *,
            turn,
            budget,
            invocations=(),
            malformed_tool_calls=0,
            final_answer_blocks=0,
        ):
            self.turn = turn
            self.budget = budget
            self.invocations = tuple(invocations)
            self.malformed_tool_calls = malformed_tool_calls
            self.final_answer_blocks = final_answer_blocks

        @property
        def has_edit(self):
            return any(call.name == "edit" for call in self.invocations)

        @property
        def progress(self):
            return self.turn / max(1, self.budget)

    def get_available_tools(
        all_tool_names,
        turn,
        budget,
        state=None,
        policy=None,
        phases=(0.4, 0.8, 1.0),
    ):
        del turn, budget, state, policy, phases
        return list(all_tool_names)

    class FakeOrchestratorContract:
        def __init__(self, *, tool_names, default_verification_commands=()):
            self.tool_names = tuple(tool_names)
            self.default_verification_commands = tuple(default_verification_commands)

        @classmethod
        def from_tool_names(cls, tool_names, *, default_verification_commands=(), **kwargs):
            del kwargs
            return cls(
                tool_names=tool_names,
                default_verification_commands=default_verification_commands,
            )

        @property
        def verification_required(self):
            return bool(self.default_verification_commands)

        def route_for_tool(self, tool_name):
            family = {
                "search_code": "repository_exploration",
                "read_file": "repository_exploration",
                "find_file": "repository_exploration",
                "list_dir": "repository_exploration",
                "edit": "editing",
                "run_tests": "verification",
                "bash": "shell",
                "final_answer": "submission",
            }.get(tool_name, "other")
            return types.SimpleNamespace(requested_family=family, mode="bundled_tool_fallback")

        def snapshot(self):
            family_by_tool = {
                name: self.route_for_tool(name).requested_family for name in self.tool_names
            }
            return {
                "phases": ["diagnose", "edit", "verify", "submit"],
                "tool_names": list(self.tool_names),
                "family_by_tool": family_by_tool,
                "adapter_families": [],
                "default_verification_commands": list(self.default_verification_commands),
                "verification_required": self.verification_required,
                "fallback_route": "bundled_tool_fallback",
            }

    runtime_module.EventLog = FakeEventLog
    runtime_module.SafetyPolicy = FakeSafetyPolicy
    runtime_module.SessionMetadata = FakeSessionMetadata
    runtime_module.Workspace = FakeWorkspace
    memory_module.CondensedState = FakeCondensedState
    memory_module.WorkingMemory = FakeWorkingMemory
    loops_module.CondensationConfig = FakeCondensationConfig
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
    strategy_module.ToolInvocation = ToolInvocation
    strategy_module.ToolPhaseState = ToolPhaseState
    strategy_module.get_available_tools = get_available_tools
    capabilities_module.OrchestratorContract = FakeOrchestratorContract

    return {
        "mellea.agent.runtime": runtime_module,
        "mellea.agent.runtime.memory": memory_module,
        "mellea.agent.runtime.loops": loops_module,
        "mellea.agent.runtime.workspace": workspace_module,
        "mellea.agent.strategy": strategy_module,
        "mellea.agent.capabilities": capabilities_module,
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


def test_generate_patch_uses_text_react(tmp_path):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    async def mock_text_react(*args, **kwargs):
        del args, kwargs
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
    assert captured["condensation"].max_messages > 0
    assert captured["model_options"] == {
        ModelOption.SYSTEM_PROMPT: captured["system_prompt"],
        ModelOption.TEMPERATURE: 0.25,
        ModelOption.SEED: 7,
        ModelOption.MAX_NEW_TOKENS: 123,
        ModelOption.STREAM: False,
    }


def test_generate_patch_turn_guidance_pushes_to_first_edit(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

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
            test_cmds={"test_cmds": ["python -m pytest -q tests/test_bug.py"]},
        )

    msgs = captured["on_turn"](7, 9, [])
    assert "Phase: edit now." in msgs[-1]["content"]
    assert "Stop broad search" in msgs[-1]["content"]


def test_generate_patch_turn_guidance_demands_verification_after_edit(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

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
            test_cmds={"test_cmds": ["python -m pytest -q tests/test_bug.py"]},
        )

    (tmp_path / "foo.py").write_text("x = 2\n")

    msgs = captured["on_turn"](7, 9, [])
    assert "Phase: verify." in msgs[-1]["content"]
    assert "run_tests default" in msgs[-1]["content"]


def test_generate_patch_turn_guidance_stays_in_diagnose_phase_on_read_churn(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=24)
    session._m = MagicMock()
    session._m.backend = MagicMock()

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
            test_cmds={"test_cmds": ["python -m pytest -q tests/test_bug.py"]},
        )

    msgs = captured["on_turn"](
        8,
        24,
        [
            {"role": "user", "content": "[read_file] some file"},
            {"role": "user", "content": "[search_code] some query"},
        ],
    )
    assert "Phase: diagnose, then edit." in msgs[-1]["content"]


def test_generate_patch_discards_exhausted_unverified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    async def mock_text_react(*args, **kwargs):
        del args
        (tmp_path / "foo.py").write_text("x = 2\n")
        return ("ignored", False)

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
            test_cmds={"test_cmds": ["python -m pytest -q tests/test_bug.py"]},
        )

    assert result == ""


def test_generate_patch_keeps_exhausted_verified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    async def mock_text_react(*args, **kwargs):
        del args
        event_log = kwargs["event_log"]
        event_log.emit(ToolCallEvent(tool_name="run_tests", arguments={"test_cmd": "default"}))
        event_log.emit(
            ToolResultEvent(
                tool_name="run_tests",
                status="completed",
                output="$ pytest -q tests/test_bug.py\nPASSED\n1 passed",
            )
        )
        (tmp_path / "foo.py").write_text("x = 2\n")
        return ("ignored", False)

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
            test_cmds={"test_cmds": ["python -m pytest -q tests/test_bug.py"]},
        )

    assert "x = 2" in result


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

    async def mock_text_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = mock_text_react

    with (
        patch.dict(
            sys.modules,
            {
                **_install_fake_runtime_modules(),
                "mellea.agent.text_react": fake_module,
            },
        ),
        patch("mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds={"test_cmds": [f"{sys.executable} -c \"print('default verification')\""]},
        )

    assert isinstance(result, str)
    assert captured["test_cmds"] == [f"{sys.executable} -c \"print('default verification')\""]
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

    async def mock_text_react(*args, **kwargs):
        captured["goal"] = kwargs["goal"]
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = mock_text_react

    with (
        patch.dict(
            sys.modules,
            {
                **_install_fake_runtime_modules(),
                "mellea.agent.text_react": fake_module,
            },
        ),
        patch("mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds=None,
        )

    assert isinstance(result, str)
    assert captured["test_cmds"] == []
    assert captured["test_fn"] is None
    assert captured["command_fn"] is None
    assert captured["workspace"].cwd == str(tmp_path)
    assert "cheapest plain command" in captured["goal"]
    assert "Avoid full-suite runs unless necessary" in captured["goal"]


def test_swebench_live_runner_passes_task_metadata_to_generate_patch(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test", n_samples=3),
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
        test_cmds = [f"{sys.executable} -c \"print('live verification')\""]
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
        config=BenchConfig(model_id="test", n_samples=3),
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
        config=BenchConfig(model_id="test", n_samples=3),
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
        config=BenchConfig(model_id="test"),
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


def test_swebench_lite_runner_reraises_docker_unavailable(tmp_path):
    _init_repo(tmp_path)

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test"),
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
            raise DockerUnavailableError("Docker socket is unavailable")
            yield tmp_path

    with pytest.raises(DockerUnavailableError, match="Docker socket is unavailable"):
        runner._run_swebench_task(
            FakeTask(),
            swe_sandbox=FakeSWEbenchSandbox(),
            run_id=1,
        )


def test_generate_patch_records_scaffold_metrics(tmp_path):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    async def mock_text_react(*args, **kwargs):
        del args
        kwargs["on_turn"](2, 9, [])
        kwargs["tool_gate"]("edit", {}, messages=(), event_log=kwargs["event_log"])
        (tmp_path / "foo.py").write_text("x = 2\n")
        kwargs["on_turn"](3, 9, [])
        kwargs["tool_gate"](
            "run_tests",
            {"test_cmd": "default"},
            messages=(),
            event_log=kwargs["event_log"],
        )
        event_log = kwargs["event_log"]
        event_log.emit(ToolCallEvent(tool_name="run_tests", arguments={"test_cmd": "default"}))
        event_log.emit(
            ToolResultEvent(
                tool_name="run_tests",
                status="completed",
                output="$ pytest -q tests/test_bug.py\nPASSED\n1 passed",
            )
        )
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
        patch_text = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            test_cmds={"test_cmds": ["pytest -q tests/test_bug.py"]},
        )

    metrics = session.last_patch_metrics
    assert "x = 2" in patch_text
    assert metrics is not None
    assert metrics["turns_to_first_edit"] == 2
    assert metrics["turns_to_first_verification"] == 3
    assert metrics["zero_edit"] is False
    assert metrics["zero_verification"] is False
    assert metrics["verification_succeeded"] is True
    assert metrics["terminal_reason"] == "submitted"


def test_swebench_lite_aborts_before_start_run_when_docker_unavailable(tmp_path, monkeypatch):
    _init_repo(tmp_path)

    import mcode.execution.swebench as swebench_module
    from mcode.bench import swebench_lite as lite_module

    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test"),
        results_db=db,
    )

    class FakeTask:
        instance_id = "lite-1"
        repo = "test/repo"
        problem_statement = "Fix the bug"
        hints_text = "Hint"
        raw_instance = {}

    class FakeSWEbenchSandbox:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def prepare_images(self, instances):
            raise DockerUnavailableError("Docker socket is unavailable")

    monkeypatch.setattr(lite_module, "load_swebench_lite", lambda *args, **kwargs: [FakeTask()])
    monkeypatch.setattr(swebench_module, "SWEbenchSandbox", FakeSWEbenchSandbox)

    with pytest.raises(DockerUnavailableError, match="Docker socket is unavailable"):
        runner._run_swebench_lite(limit=1)

    assert db.conn.execute("select count(*) from runs").fetchone()[0] == 0
    assert db.conn.execute("select count(*) from task_results").fetchone()[0] == 0


def test_generate_patch_records_blocked_verification_commands(tmp_path):
    _init_repo(tmp_path)

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=9)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    async def mock_text_react(*args, **kwargs):
        del args
        kwargs["on_turn"](4, 9, [])
        kwargs["tool_gate"]("edit", {}, messages=(), event_log=kwargs["event_log"])
        (tmp_path / "foo.py").write_text("x = 2\n")
        event_log = kwargs["event_log"]
        event_log.emit(
            ToolCallEvent(
                tool_name="run_tests",
                arguments={
                    "test_cmd": "python -m pytest -q tests/test_bug.py | head -20",
                },
            )
        )
        event_log.emit(
            ToolResultEvent(
                tool_name="run_tests",
                status="completed",
                output=(
                    "$ python -m pytest -q tests/test_bug.py | head -20\n"
                    "BLOCKED\n"
                    "Run a plain verification command inside `run_tests`. Do not use "
                    "pipes, redirection, or command chaining."
                ),
            )
        )
        return ("done", False)

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
            test_cmds={"test_cmds": ["pytest -q tests/test_bug.py"]},
        )

    metrics = session.last_patch_metrics
    assert result == ""
    assert metrics is not None
    assert metrics["blocked_verification_commands"] == 1
    assert metrics["terminal_reason"] == "unverified_diff_discarded"
