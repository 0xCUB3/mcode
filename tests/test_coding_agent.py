from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from mcode.llm.session import LLMSession


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

    def fake_make_agent_tools(repo_root, *, test_cmds=None, test_fn=None):
        captured["repo_root"] = repo_root
        captured["test_cmds"] = test_cmds
        captured["test_fn"] = test_fn
        return ["tool-a"]

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "0")

    with patch("mcode.agent.coding_agent.build_repo_map", return_value="repo map"), patch(
        "mcode.agent.coding_agent.make_agent_tools", fake_make_agent_tools
    ):
        assembly = build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            hints_text="Hint text",
            repo_root=str(tmp_path),
            test_cmds={
                "test_cmds": [f'{sys.executable} -c "print(\'default verification\')"']
            },
        )

    assert assembly.goal.startswith("Fix this bug in test/repo")
    assert "Repository structure:\nrepo map" in assembly.goal
    assert "Additional context:\nHint text" in assembly.goal
    assert "Start with `run_tests default`" in assembly.goal
    assert "Keep verification cheap" in assembly.goal
    assert assembly.tools == ["tool-a"]
    assert captured["repo_root"] == str(tmp_path)
    assert captured["test_cmds"] == [
        f'{sys.executable} -c "print(\'default verification\')"'
    ]
    assert captured["test_fn"] is not None


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


def test_build_verification_policy_prefers_task_default_checks(tmp_path):
    from mcode.agent.verification import build_verification_policy

    policy = build_verification_policy(
        repo_root=str(tmp_path),
        test_cmds={
            "verification_cmds": [f'{sys.executable} -c "print(\'default verification\')"']
        },
    )

    assert policy.test_cmds == [f'{sys.executable} -c "print(\'default verification\')"']
    assert "Start with `run_tests default`" in policy.prompt_block
    assert "Keep verification cheap" in policy.prompt_block
    output = policy.test_fn("default")
    assert "default verification" in output
    assert "PASSED" in output


def test_build_verification_policy_uses_shell_first_fallback(tmp_path):
    from mcode.agent.verification import build_verification_policy

    policy = build_verification_policy(repo_root=str(tmp_path))

    assert policy.test_cmds == []
    assert "cheapest shell command" in policy.prompt_block
    assert "Avoid full-suite runs unless necessary." in policy.prompt_block
    output = policy.test_fn(f'{sys.executable} -c "print(\'shell verification\')"')
    assert "shell verification" in output
    assert "PASSED" in output


def test_coding_agent_can_be_requested_from_session_generate_patch(
    tmp_path, monkeypatch
):
    from mcode.llm import session as session_module

    monkeypatch.setenv("MELLEA_TEXT_TOOLS", "0")

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
        use_text_tools=False,
        use_budget_warning=False,
        use_mid_nudge=False,
        system_prompt="system",
        verification_cmds=[],
        verification_test_fn=lambda cmd="default": "",
    )

    async def fake_react(*args, **kwargs):
        return (types.SimpleNamespace(value="done"), MagicMock())

    with patch.object(session_module, "build_coding_agent", build_agent, create=True), patch(
        "mellea.agent.tools.make_agent_tools", return_value=[]
    ), patch("mellea.agent.repomap.build_repo_map", return_value="repo map"), patch(
        "mellea.stdlib.frameworks.react.react", fake_react
    ), patch.object(session_module, "_get_diff", return_value="diff"):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )

    assert result == "diff"
    assert build_agent.called
