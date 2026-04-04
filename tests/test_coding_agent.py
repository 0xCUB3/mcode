from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from mcode.llm.session import LLMSession


def _install_fake_runtime_modules():
    runtime_module = types.ModuleType("mellea.agent.runtime")
    memory_module = types.ModuleType("mellea.agent.runtime.memory")
    loops_module = types.ModuleType("mellea.agent.runtime.loops")
    capabilities_module = types.ModuleType("mellea.agent.capabilities")

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

    class FakeWorkingMemory:
        def __init__(self, summary="", facts=(), hypotheses=(), next_steps=()):
            self.summary = summary
            self.facts = tuple(facts)
            self.hypotheses = tuple(hypotheses)
            self.next_steps = tuple(next_steps)

        def as_message(self, *, omitted_messages=0):
            prefix = (
                f"Condensed context: {omitted_messages} omitted\n"
                if omitted_messages
                else "Condensed context:\n"
            )
            return {"role": "user", "content": f"{prefix}{self.summary}".strip()}

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
    capabilities_module.OrchestratorContract = FakeOrchestratorContract

    return (
        {
            "mellea.agent.runtime": runtime_module,
            "mellea.agent.runtime.memory": memory_module,
            "mellea.agent.runtime.loops": loops_module,
            "mellea.agent.capabilities": capabilities_module,
        },
        FakeWorkspace,
        FakeEventLog,
        FakeCondensedState,
        FakeCondensationConfig,
    )


def test_build_coding_agent_assembles_prompt_and_tools_without_benchmark_path(
    tmp_path, monkeypatch
):
    from mcode.agent.coding_agent import build_coding_agent

    session = LLMSession(
        model_id="test",
        backend_name="openai",
        temperature=0.25,
        seed=7,
        loop_budget=9,
    )
    session._m = MagicMock()
    session._m.backend = MagicMock()

    captured: dict[str, object] = {}

    def fake_make_agent_tools(
        repo_root, *, test_cmds=None, test_fn=None, command_fn=None, workspace=None
    ):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        captured["command_fn"] = command_fn
        captured["workspace"] = workspace
        return ["tool-a"]

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "0")

    with (
        patch.dict(sys.modules, _install_fake_runtime_modules()[0]),
        patch("mcode.agent.coding_agent.build_repo_map", return_value="repo map"),
        patch("mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools),
    ):
        assembly = build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            hints_text="Hint text",
            repo_root=str(tmp_path),
            test_cmds={"test_cmds": [f"{sys.executable} -c \"print('default verification')\""]},
        )

    assert assembly.goal.startswith("Fix this bug in test/repo")
    assert "Repository structure:\nrepo map" in assembly.goal
    assert "Additional context:\nHint text" in assembly.goal
    assert "Start with `run_tests default`" in assembly.goal
    assert "Keep verification cheap" in assembly.goal
    assert assembly.tools == ["tool-a"]
    assert captured["repo_root"] == str(tmp_path)
    assert captured["test_cmds"] == [f"{sys.executable} -c \"print('default verification')\""]
    assert captured["test_fn"] is None
    assert captured["command_fn"] is None
    assert captured["workspace"] is assembly.workspace


def test_build_coding_agent_attaches_orchestrator_contract_to_workspace(tmp_path, monkeypatch):
    from mcode.agent import coding_agent as coding_agent_module

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    runtime_modules, _, _, _, _ = _install_fake_runtime_modules()
    fake_policy = types.SimpleNamespace(
        system_prompt="system prompt from mcode",
        goal="goal from mcode",
    )
    fake_verification = types.SimpleNamespace(
        test_cmds=["pytest -q tests/test_coding_agent.py"],
        test_fn=None,
        prompt_block="",
    )
    fake_tools = [
        types.SimpleNamespace(name="search_code"),
        types.SimpleNamespace(name="edit"),
        types.SimpleNamespace(name="run_tests"),
    ]

    with (
        patch.dict(sys.modules, runtime_modules),
        patch.object(coding_agent_module, "build_repo_map", return_value="repo map"),
        patch.object(coding_agent_module, "make_agent_tools", return_value=fake_tools),
        patch.object(coding_agent_module, "build_coding_policy", return_value=fake_policy),
        patch.object(
            coding_agent_module, "build_verification_policy", return_value=fake_verification
        ),
    ):
        assembly = coding_agent_module.build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    snapshot = assembly.capability_contract.snapshot()
    assert snapshot["phases"] == ["diagnose", "edit", "verify", "submit"]
    assert snapshot["family_by_tool"] == {
        "search_code": "repository_exploration",
        "edit": "editing",
        "run_tests": "verification",
    }
    assert snapshot["default_verification_commands"] == ["pytest -q tests/test_coding_agent.py"]
    assert assembly.workspace.metadata["orchestrator_contract"] == snapshot
    assert assembly.capability_contract.verification_required is True


def test_build_coding_agent_assembles_mellea_runtime_and_keeps_mcode_policy(tmp_path, monkeypatch):
    from mcode.agent import coding_agent as coding_agent_module

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    runtime_modules, FakeWorkspace, FakeEventLog, FakeCondensedState, FakeCondensationConfig = (
        _install_fake_runtime_modules()
    )
    fake_policy = types.SimpleNamespace(
        system_prompt="system prompt from mcode",
        goal="goal from mcode",
    )
    fake_verification = types.SimpleNamespace(
        test_cmds=["pytest -q tests/test_coding_agent.py"],
        test_fn=lambda test_cmd="default": test_cmd,
        prompt_block="\n\nVerification:\nUse mcode verification policy.",
    )

    with (
        patch.dict(sys.modules, runtime_modules),
        patch.object(coding_agent_module, "build_repo_map", return_value="repo map"),
        patch.object(coding_agent_module, "make_agent_tools", return_value=["tool-a"]),
        patch.object(coding_agent_module, "build_coding_policy", return_value=fake_policy),
        patch.object(
            coding_agent_module, "build_verification_policy", return_value=fake_verification
        ),
    ):
        assembly = coding_agent_module.build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert assembly.coding_policy is fake_policy
    assert assembly.verification_policy is fake_verification
    assert assembly.system_prompt == "system prompt from mcode"
    assert assembly.goal == "goal from mcode"
    assert isinstance(assembly.workspace, FakeWorkspace)
    assert isinstance(assembly.event_log, FakeEventLog)
    assert assembly.event_log.workspace is assembly.workspace
    assert isinstance(assembly.condensed_state, FakeCondensedState)
    assert isinstance(assembly.condensation, FakeCondensationConfig)
    assert assembly.condensation.max_messages > 0


def test_build_coding_agent_can_expose_visible_repo_root_to_runtime_state(tmp_path, monkeypatch):
    from mcode.agent import coding_agent as coding_agent_module

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    runtime_modules, FakeWorkspace, _, _, _ = _install_fake_runtime_modules()
    fake_policy = types.SimpleNamespace(
        system_prompt="system prompt from mcode",
        goal="goal from mcode",
    )
    fake_verification = types.SimpleNamespace(
        test_cmds=[],
        test_fn=lambda test_cmd="default": test_cmd,
        prompt_block="",
    )

    with (
        patch.dict(sys.modules, runtime_modules),
        patch.object(coding_agent_module, "build_repo_map", return_value="repo map"),
        patch.object(coding_agent_module, "make_agent_tools", return_value=["tool-a"]),
        patch.object(coding_agent_module, "build_coding_policy", return_value=fake_policy),
        patch.object(
            coding_agent_module, "build_verification_policy", return_value=fake_verification
        ),
    ):
        assembly = coding_agent_module.build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
            visible_repo_root="/testbed",
        )

    assert isinstance(assembly.workspace, FakeWorkspace)
    assert assembly.workspace.cwd == str(tmp_path)
    assert assembly.workspace.metadata["display_cwd"] == "/testbed"


def test_build_coding_agent_requires_runtime_primitives(tmp_path):
    from mcode.agent import coding_agent as coding_agent_module

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    with patch.object(
        coding_agent_module.importlib,
        "import_module",
        side_effect=ModuleNotFoundError("missing mellea runtime"),
    ):
        try:
            coding_agent_module.build_coding_agent(
                session=session,
                repo="test/repo",
                problem_statement="Fix the bug",
                repo_root=str(tmp_path),
            )
        except RuntimeError as exc:
            assert "mellea runtime primitives" in str(exc)
        else:  # pragma: no cover - this is the failure mode under test
            raise AssertionError(
                "build_coding_agent() should fail when runtime primitives are missing"
            )


def test_build_coding_policy_composes_shell_first_goal():
    from mcode.agent.coding_policy import build_coding_policy

    policy = build_coding_policy(
        repo="test/repo",
        problem_statement="Fix the bug",
        hints_text="Hint text",
        repo_map_text="repo map",
        verification_prompt="\n\nVerification:\nUse the cheapest shell command.",
    )

    assert policy.system_prompt.startswith(
        "You are an expert software engineer fixing a bug in an open-source repository."
    )
    assert "Repository structure:\nrepo map" in policy.goal
    assert "Additional context:\nHint text" in policy.goal
    assert "Use the cheapest shell command." in policy.goal
    assert "PHASES:" in policy.system_prompt
    assert "DIAGNOSE" in policy.system_prompt
    assert "VERIFY" in policy.system_prompt
    assert "SUBMIT" in policy.system_prompt


def test_build_coding_policy_can_include_repo_customization():
    from mcode.agent.coding_policy import build_coding_policy

    policy = build_coding_policy(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_customization_text="Repo note: use `bin/test -q` for narrow checks.",
    )

    assert (
        "Repository-specific guidance:\nRepo note: use `bin/test -q` for narrow checks."
    ) in policy.goal


def test_build_verification_policy_prefers_task_default_checks(tmp_path):
    from mcode.agent.verification import build_verification_policy

    policy = build_verification_policy(
        repo_root=str(tmp_path),
        test_cmds={"verification_cmds": [f"{sys.executable} -c \"print('default verification')\""]},
    )

    assert policy.test_cmds == [f"{sys.executable} -c \"print('default verification')\""]
    assert "Start with `run_tests default`" in policy.prompt_block
    assert "Keep verification cheap" in policy.prompt_block
    assert policy.test_fn is None


def test_build_verification_policy_uses_shell_first_fallback(tmp_path):
    from mcode.agent.verification import build_verification_policy

    policy = build_verification_policy(repo_root=str(tmp_path))

    assert policy.test_cmds == []
    assert "cheapest plain command" in policy.prompt_block
    assert "Avoid full-suite runs unless necessary" in policy.prompt_block
    assert policy.test_fn is None


def test_build_verification_policy_preserves_explicit_test_fn(tmp_path):
    from mcode.agent.verification import build_verification_policy

    def custom_test_fn(command="default"):
        return f"custom:{command}"

    policy = build_verification_policy(
        repo_root=str(tmp_path),
        test_cmds=["pytest -q"],
        test_fn=custom_test_fn,
    )

    assert policy.test_cmds == ["pytest -q"]
    assert policy.test_fn is custom_test_fn


def test_coding_agent_can_be_requested_from_session_generate_patch(tmp_path, monkeypatch):
    from mcode.llm import session as session_module

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    build_agent = MagicMock()
    build_agent.return_value = types.SimpleNamespace(
        goal="assembled",
        tools=[],
        model_options={},
        loop_budget=1,
        timeout_s=1,
        system_prompt="system",
        verification_cmds=[],
        verification_test_fn=lambda cmd="default": "",
    )

    async def fake_text_react(*args, **kwargs):
        del args, kwargs
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = fake_text_react

    with (
        patch.object(session_module, "build_coding_agent", build_agent, create=True),
        patch.dict(sys.modules, {"mellea.agent.text_react": fake_module}),
        patch.object(session_module, "_get_diff", return_value=""),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert result == ""
    assert build_agent.called
    assert build_agent.called


def test_load_repo_customization_prefers_repo_md(tmp_path):
    from mcode.agent.repo_customization import load_repo_customization

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    customization_dir = repo_dir / ".mcode"
    customization_dir.mkdir()
    (customization_dir / "repo.md").write_text("Use `bin/test -q` for quick checks.\n")

    customization = load_repo_customization(str(repo_dir))

    assert customization.text == "Use `bin/test -q` for quick checks."
    assert customization.source_path == customization_dir / "repo.md"


def test_build_coding_agent_includes_repo_customization_text(tmp_path, monkeypatch):
    from mcode.agent import coding_agent as coding_agent_module

    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = MagicMock()
    session._m.backend = MagicMock()

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "1")

    runtime_modules, _, _, _, _ = _install_fake_runtime_modules()
    fake_verification = types.SimpleNamespace(
        test_cmds=[],
        test_fn=None,
        prompt_block="",
    )

    captured: dict[str, object] = {}

    def fake_build_coding_policy(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            system_prompt="system prompt from mcode",
            goal="goal from mcode",
            prompt_inputs=lambda: {
                "system_prompt": "system prompt from mcode",
                "goal": "goal from mcode",
            },
        )

    with (
        patch.dict(sys.modules, runtime_modules),
        patch.object(coding_agent_module, "build_repo_map", return_value="repo map"),
        patch.object(coding_agent_module, "make_agent_tools", return_value=["tool-a"]),
        patch.object(
            coding_agent_module, "build_coding_policy", side_effect=fake_build_coding_policy
        ),
        patch.object(
            coding_agent_module,
            "build_verification_policy",
            return_value=fake_verification,
        ),
        patch.object(
            coding_agent_module,
            "load_repo_customization",
            return_value=types.SimpleNamespace(
                text="Repo note: use `bin/test -q` first.",
                source_path=tmp_path / ".mcode" / "repo.md",
            ),
        ),
    ):
        coding_agent_module.build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert captured["repo_customization_text"] == "Repo note: use `bin/test -q` first."


def test_session_generate_patch_seeds_react_context_from_runtime_state(tmp_path, monkeypatch):
    from mcode.llm import session as session_module

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    condensed_state = types.SimpleNamespace(label="memory")
    build_agent = MagicMock()
    build_agent.return_value = types.SimpleNamespace(
        goal="assembled",
        tools=[],
        model_options={},
        loop_budget=1,
        timeout_s=1,
        system_prompt="system",
        verification_cmds=[],
        verification_test_fn=lambda cmd="default": "",
        condensed_state=condensed_state,
        workspace=types.SimpleNamespace(cwd=str(tmp_path)),
        event_log=types.SimpleNamespace(),
    )

    captured: dict[str, object] = {}

    async def fake_text_react(*args, **kwargs):
        captured.update(kwargs)
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = fake_text_react

    with (
        patch.object(session_module, "build_coding_agent", build_agent, create=True),
        patch.dict(sys.modules, {"mellea.agent.text_react": fake_module}),
        patch.object(session_module, "_get_diff", return_value=""),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert result == ""
    assert captured["condensed_state"] is condensed_state


def test_session_generate_patch_passes_runtime_state_to_text_react(tmp_path, monkeypatch):
    from mcode.llm import session as session_module

    session = LLMSession(model_id="test", backend_name="ollama")
    session._m = MagicMock()
    session._m.backend = MagicMock()

    workspace = types.SimpleNamespace(cwd=str(tmp_path))
    event_log = types.SimpleNamespace(workspace=workspace)
    condensed_state = types.SimpleNamespace(label="memory")
    build_agent = MagicMock()
    build_agent.return_value = types.SimpleNamespace(
        goal="assembled",
        tools=[],
        model_options={},
        loop_budget=3,
        timeout_s=1,
        system_prompt="system",
        verification_cmds=[],
        verification_test_fn=lambda cmd="default": "",
        workspace=workspace,
        event_log=event_log,
        condensed_state=condensed_state,
        max_retries_per_turn=2,
    )

    captured: dict[str, object] = {}

    async def fake_text_react(*args, **kwargs):
        captured.update(kwargs)
        return ("done", True)

    fake_module = types.ModuleType("mellea.agent.text_react")
    fake_module.text_react = fake_text_react

    with (
        patch.object(session_module, "build_coding_agent", build_agent, create=True),
        patch.dict(sys.modules, {"mellea.agent.text_react": fake_module}),
        patch.object(session_module, "_get_diff", return_value=""),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert result == ""
    assert captured["event_log"] is event_log
    assert captured["condensed_state"] is condensed_state
    assert captured["max_retries_per_turn"] == 2
    assert callable(captured["tool_gate"])
