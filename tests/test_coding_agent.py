from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from mellea.backends import ModelOption

from mcode.agent.coding_agent import build_coding_agent, make_agent_tools
from mcode.agent.tooling import format_tool_result
from mcode.agent.verification import (
    build_run_tests_tool,
    build_verification_policy,
)
from mcode.llm.session import LLMSession


def test_build_coding_agent_assembles_prompt_and_tools(tmp_path):
    session = LLMSession(
        model_id="test",
        backend_name="openai",
        temperature=0.25,
        seed=7,
        loop_budget=9,
    )
    session._m = SimpleNamespace(backend=object())

    with patch("mcode.agent.coding_agent.make_agent_tools", return_value=["tool-a"]):
        assembly = build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            hints_text="Hint text",
            repo_root=str(tmp_path),
            test_cmds={"verification_cmds": ["pytest -q tests/test_bug.py"]},
        )

    assert assembly.tools == ["tool-a"]
    assert "Hint text" in assembly.goal
    assert 'test_cmd="default"' in assembly.goal
    assert "Local workspace context:" not in assembly.goal
    assert "Spend at most two search/read turns" in assembly.system_prompt
    assert assembly.model_options[ModelOption.TEMPERATURE] == 0.25
    assert assembly.model_options[ModelOption.SEED] == 7
    assert assembly.loop_budget == 9


def test_build_verification_policy_normalizes_commands():
    policy = build_verification_policy(
        test_cmds={"verification_cmds": ["pytest -q tests/test_bug.py", ""]},
    )

    assert policy.test_cmds == ["pytest -q tests/test_bug.py"]
    assert 'test_cmd="default"' in policy.prompt_block
    assert "not pass `run_tests default`" in policy.prompt_block


def test_verification_policy_without_default_rejects_default():
    policy = build_verification_policy(command_fn=lambda command: command)

    assert policy.test_cmds == []
    assert policy.allow_default_test_cmd is False
    assert "There is no default test command" in policy.prompt_block
    assert "use `test_cmd=\"default\"`" not in policy.prompt_block


def test_build_run_tests_tool_uses_command_fn_for_default_commands(tmp_path):
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(
        test_cmds=["pytest -q tests/test_bug.py"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == ["pytest -q tests/test_bug.py"]
    assert "$ pytest -q tests/test_bug.py" in result
    assert "PASSED" in result


def test_run_tests_tool_blocks_default_without_declared_commands(tmp_path):
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == []
    assert "$ default" in result
    assert "BLOCKED" in result
    assert "No default test command is declared" in result


def test_run_tests_tool_blocks_evasive_pytest_selection(tmp_path):
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run('pytest tests/test_bug.py -k "not test_join"')

    assert seen == []
    assert "BLOCKED" in result
    assert "without skipping" in result

    result = tool.run("pytest tests/test_bug.py 2>&1 | head -80")
    assert seen == []
    assert "BLOCKED" in result
    assert "masking their exit status" in result


def test_run_tests_tool_appends_failure_report_snippets(tmp_path):
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-example.xml").write_text(
        '<testsuite><testcase name="badCase" classname="ExampleTest">'
        '<failure message="expected 1 but was 2">stack trace</failure>'
        "</testcase></testsuite>"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(command, "FAILED", "There were failing tests.")

    policy = build_verification_policy(
        test_cmds=["./gradlew test --no-daemon -q"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "Failure report snippets:" in result
    assert "build/test-results/test/TEST-example.xml" in result
    assert "expected 1 but was 2" in result


def test_make_agent_tools_appends_run_tests(tmp_path):
    policy = build_verification_policy(test_cmds=["pytest -q"])

    run_tests_tool = SimpleNamespace(name="run_tests")
    with patch("mcode.agent.coding_agent.build_run_tests_tool", return_value=run_tests_tool):
        tools = make_agent_tools(str(tmp_path), verification_policy=policy)

    assert [tool.name for tool in tools] == [
        "search_code",
        "edit",
        "read_file",
        "find_file",
        "list_dir",
        "run_tests",
    ]


def test_make_agent_tools_uses_native_mellea_tools(tmp_path):
    policy = build_verification_policy(test_cmds=["pytest -q"])

    tools = {
        tool.name: tool for tool in make_agent_tools(str(tmp_path), verification_policy=policy)
    }

    assert callable(tools["read_file"].run)
    assert tools["list_dir"].run()
    assert tools["read_file"].as_json_tool["function"]["parameters"]["required"] == ["path"]
    assert tools["list_dir"].as_json_tool["function"]["parameters"].get("required") == []


def test_make_agent_tools_normalizes_visible_repo_paths(tmp_path):
    target = tmp_path / "pkg"
    target.mkdir()
    (target / "mod.py").write_text("value = 1\n")
    policy = build_verification_policy(test_cmds=[])
    tools = {
        tool.name: tool
        for tool in make_agent_tools(
            str(tmp_path),
            verification_policy=policy,
            visible_repo_root="/testbed",
        )
    }

    assert "value = 1" in tools["read_file"].run("/testbed/pkg/mod.py")
    assert "pkg" in tools["list_dir"].run("/testbed")
    result = tools["edit"].run("c:/users/user/tmp/repo/pkg/mod.py", "value = 1", "value = 2")

    assert "APPLIED" in result
    result = tools["edit"].run(str(target / "mod.py"), "value = 2", "value = 3")

    assert "APPLIED" in result
    assert (target / "mod.py").read_text() == "value = 3\n"
