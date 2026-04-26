from __future__ import annotations

import subprocess
from pathlib import Path
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

    with (
        patch("mcode.agent.coding_agent.build_repo_map", return_value="repo map"),
        patch(
            "mcode.agent.coding_agent.build_candidate_files",
            return_value="Likely files to inspect first:\nfoo.py",
        ),
        patch("mcode.agent.coding_agent.make_agent_tools", return_value=["tool-a"]),
        patch(
            "mcode.agent.coding_agent.collect_workspace_context",
            return_value=SimpleNamespace(
                text="Local workspace context:\n- README.md\nUse project docs.",
            ),
        ),
    ):
        assembly = build_coding_agent(
            session=session,
            repo="test/repo",
            problem_statement="Fix the bug",
            hints_text="Hint text",
            repo_root=str(tmp_path),
            test_cmds={"verification_cmds": ["pytest -q tests/test_bug.py"]},
        )

    assert assembly.tools == ["tool-a"]
    assert "repo map" in assembly.goal
    assert "Hint text" in assembly.goal
    assert 'test_cmd="default"' in assembly.goal
    assert "Local workspace context:" in assembly.goal
    assert "Use project docs." in assembly.goal
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


def _init_git_repo(path: Path) -> None:
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path,
        capture_output=True,
        check=True,
        env=env,
    )


def test_run_tests_tool_infers_default_from_changed_test_file(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "tests" / "test_bug.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_old():\n    assert True\n")
    _init_git_repo(tmp_path)
    test_file.write_text("def test_new():\n    assert True\n")
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == ["python -m pytest -q tests/test_bug.py"]
    assert "PASSED" in result


def test_run_tests_tool_infers_default_from_changed_source_test_file(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "pkg" / "mod.py"
    test_file = tmp_path / "pkg" / "tests" / "test_mod.py"
    test_file.parent.mkdir(parents=True)
    source_file.write_text("VALUE = 1\n")
    test_file.write_text("def test_mod():\n    assert True\n")
    _init_git_repo(tmp_path)
    source_file.write_text("VALUE = 2\n")
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == ["python -m pytest -q pkg/tests/test_mod.py"]
    assert "PASSED" in result


def test_run_tests_tool_infers_default_from_changed_source_test_dir(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "pkg" / "core.py"
    test_file = tmp_path / "pkg" / "tests" / "test_sampled.py"
    test_file.parent.mkdir(parents=True)
    source_file.write_text("VALUE = 1\n")
    test_file.write_text("def test_sampled():\n    assert True\n")
    _init_git_repo(tmp_path)
    source_file.write_text("VALUE = 2\n")
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == ["python -m pytest -q pkg/tests"]
    assert "PASSED" in result


def test_run_tests_tool_skips_default_when_no_tests_can_be_inferred(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "pkg" / "core.py"
    source_file.parent.mkdir()
    source_file.write_text("VALUE = 1\n")
    _init_git_repo(tmp_path)
    source_file.write_text("VALUE = 2\n")
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert seen == []
    assert "SKIPPED" in result
    assert "No test commands available" in result


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


def test_make_agent_tools_preserve_optional_tool_defaults(tmp_path):
    policy = build_verification_policy(test_cmds=["pytest -q"])

    tools = {
        tool.name: tool for tool in make_agent_tools(str(tmp_path), verification_policy=policy)
    }

    assert tools["read_file"].as_json_tool["function"]["parameters"]["required"] == ["path"]
    assert tools["list_dir"].as_json_tool["function"]["parameters"]["required"] == []
