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
    assert 'use `test_cmd="default"`' not in policy.prompt_block


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


def test_run_tests_tool_blocks_custom_commands_when_defaults_exist(tmp_path):
    seen: list[str] = []

    def command_fn(command: str) -> str:
        seen.append(command)
        return format_tool_result(command, "PASSED", "ok")

    policy = build_verification_policy(
        test_cmds=["./tests/runtests.py auth_tests.test_validators"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("pytest tests/auth_tests/test_validators.py -v")

    assert seen == []
    assert "BLOCKED" in result
    assert 'test_cmd="default"' in result


def test_run_tests_tool_appends_failure_report_snippets(tmp_path):
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-example.xml").write_text(
        '<testsuite><testcase name="badCase" classname="ExampleTest">'
        '<failure message="expected 1 but was 2">stack trace</failure>'
        "</testcase></testsuite>"
    )
    test_src = tmp_path / "src" / "test" / "java" / "ExampleTest.java"
    test_src.parent.mkdir(parents=True)
    test_src.write_text(
        "class ExampleTest {\n"
        "  @Test\n"
        "  void badCase() {\n"
        "    assertThat(value).isEqualTo(1);\n"
        "  }\n"
        "}\n"
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
    assert "Failing test source snippets:" in result
    assert "src/test/java/ExampleTest.java::badCase" in result
    assert "assertThat(value).isEqualTo(1)" in result


def test_run_tests_tool_adds_failed_test_helper_snippets(tmp_path):
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-example.xml").write_text(
        '<testsuite><testcase name="usesHelpers" classname="ExampleTest">'
        '<failure message="expected 2 but was 0">stack trace</failure>'
        "</testcase></testsuite>"
    )
    test_src = tmp_path / "src" / "test" / "java" / "ExampleTest.java"
    test_src.parent.mkdir(parents=True)
    test_src.write_text(
        "class ExampleTest {\n"
        "  @Test\n"
        "  void usesHelpers() {\n"
        "    Object value = createValue();\n"
        "    assertThat(value).isNotNull();\n"
        "  }\n"
        "  Object createValue() {\n"
        "    return new Object();\n"
        "  }\n"
        "}\n"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(command, "FAILED", "There were failing tests.")

    policy = build_verification_policy(test_cmds=["./gradlew test"], command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "helper createValue" in result
    assert "return new Object()" in result
    assert "helper assertThat" not in result


def test_run_tests_tool_preserves_status_when_truncating(tmp_path):
    def command_fn(command: str) -> str:
        return format_tool_result(command, "FAILED", "start\n" + ("filler\n" * 200) + "final error")

    policy = build_verification_policy(
        test_cmds=["pytest -q"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default", max_output_chars=120)

    assert result.startswith("$ pytest -q\nFAILED")
    assert "[tool output truncated, keeping final diagnostics]" in result
    assert "final error" in result


def test_run_tests_tool_includes_junit_failure_body_details(tmp_path):
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-HangmanTest.xml").write_text(
        '<testsuite><testcase classname="HangmanTest" name="wonGame">'
        '<failure message="Expecting actual:">'
        "but the following elements were unexpected:\n  [a, o]"
        "</failure></testcase></testsuite>"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(command, "FAILED", "1 test completed, 1 failed")

    policy = build_verification_policy(test_cmds=["./gradlew test"], command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "but the following elements were unexpected" in result
    assert "[a, o]" in result


def test_run_tests_tool_adds_rust_integration_test_source_snippets(tmp_path):
    test_src = tmp_path / "tests" / "alloc-attack.rs"
    test_src.parent.mkdir()
    test_src.write_text(
        "#[test]\n"
        "fn alloc_attack() {\n"
        "    let before = GLOBAL_ALLOCATOR.get_bytes_allocated();\n"
        "    assert!(before < 1024 * 1024);\n"
        "}\n"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(
            command,
            "FAILED",
            "thread 'alloc_attack' panicked at tests/alloc-attack.rs:4:5:\n"
            "assertion failed: GLOBAL_ALLOCATOR.get_bytes_allocated() < 1024 * 1024",
        )

    policy = build_verification_policy(test_cmds=["cargo test"], command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "tests/alloc-attack.rs:4" in result
    assert "fn alloc_attack()" in result


def test_run_tests_tool_skips_stale_reports_for_compile_failures(tmp_path):
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-OldTest.xml").write_text(
        '<testsuite><testcase classname="OldTest" name="old">'
        '<failure message="stale" /></testcase></testsuite>'
    )

    def command_fn(command: str) -> str:
        return format_tool_result(command, "FAILED", "Compilation failed; see compiler output")

    policy = build_verification_policy(test_cmds=["./gradlew test"], command_fn=command_fn)
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "Compilation failed" in result
    assert "OldTest" not in result


def test_run_tests_tool_adds_go_table_subtest_source_snippets(tmp_path):
    test_src = tmp_path / "connect_test.go"
    test_src.write_text(
        "package connect\n\n"
        "func TestResultOf(t *testing.T) {\n"
        "\ttests := []struct {\n"
        "\t\tdescription string\n"
        "\t\tboard []string\n"
        "\t}{\n"
        "\t\t{\n"
        '\t\t\tdescription: "illegal diagonal does not make a winner",\n'
        '\t\t\tboard: []string{"X O . .", " O X X X"},\n'
        "\t\t},\n"
        "\t}\n"
        "}\n"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(
            command,
            "FAILED",
            "--- FAIL: TestResultOf (0.00s)\n"
            "    --- FAIL: TestResultOf/illegal_diagonal_does_not_make_a_winner (0.00s)\n"
            "        connect_test.go:26: got X want empty",
        )

    policy = build_verification_policy(
        test_cmds=["go test ./..."],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "connect_test.go::illegal_diagonal_does_not_make_a_winner" in result
    assert 'board: []string{"X O . .", " O X X X"}' in result


def test_run_tests_tool_adds_pytest_failed_test_source_snippets(tmp_path):
    test_src = tmp_path / "connect_test.py"
    test_src.write_text(
        "class ConnectTest:\n"
        "    def test_illegal_diagonal_does_not_make_a_winner(self):\n"
        "        board = ['X O . .', ' O X X X']\n"
        "        assert winner == ''\n"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(
            command,
            "FAILED",
            "connect_test.py:3: in test_illegal_diagonal_does_not_make_a_winner\n"
            "    assert winner == ''\n"
            "E   AssertionError: 'X' != ''",
        )

    policy = build_verification_policy(
        test_cmds=["python -m pytest *_test.py -v --tb=short -q"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "Failing test source snippets:" in result
    assert "connect_test.py:3" in result
    assert "def test_illegal_diagonal_does_not_make_a_winner" in result
    assert "board = ['X O . .', ' O X X X']" in result


def test_run_tests_tool_adds_jest_failed_test_source_snippets(tmp_path):
    spec = tmp_path / "promises.spec.js"
    spec.write_text(
        "describe('promises', () => {\n"
        "  xtest('resolves when given no arguments', () => {\n"
        "    return expect(allSettled()).resolves.toBeUndefined();\n"
        "  });\n"
        "  xtest('resolves when given no arguments', () => {\n"
        "    return expect(race()).resolves.toBeUndefined();\n"
        "  });\n"
        "});\n"
    )

    def command_fn(command: str) -> str:
        return format_tool_result(
            command,
            "FAILED",
            "FAIL ./promises.spec.js\n"
            "    allSettled\n"
            "      ✕ resolves when given no arguments (1 ms)\n"
            "    race\n"
            "      ✕ resolves when given no arguments (1 ms)",
        )

    policy = build_verification_policy(
        test_cmds=["npm test --silent"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)

    assert tool is not None
    result = tool.run("default")

    assert "Failing test source snippets:" in result
    assert "promises.spec.js::resolves when given no arguments" in result
    assert "allSettled()).resolves.toBeUndefined()" in result
    assert "race()).resolves.toBeUndefined()" in result


def test_make_agent_tools_appends_run_tests(tmp_path):
    policy = build_verification_policy(test_cmds=["pytest -q"])

    run_tests_tool = SimpleNamespace(name="run_tests")
    with patch("mcode.agent.coding_agent.build_run_tests_tool", return_value=run_tests_tool):
        tools = make_agent_tools(str(tmp_path), verification_policy=policy)

    assert [tool.name for tool in tools] == [
        "search_code",
        "edit",
        "read_file",
        "run_file",
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
    assert callable(tools["run_file"].run)
    assert tools["list_dir"].run()
    assert tools["read_file"].as_json_tool["function"]["parameters"]["required"] == ["path"]
    assert tools["list_dir"].as_json_tool["function"]["parameters"].get("required") == []


def test_make_agent_tools_restricts_editable_paths(tmp_path):
    allowed = tmp_path / "src" / "allowed.py"
    blocked = tmp_path / "build.gradle"
    allowed.parent.mkdir()
    allowed.write_text("value = 1\n")
    blocked.write_text("value = 1\n")
    policy = build_verification_policy(test_cmds=[])
    tools = {
        tool.name: tool
        for tool in make_agent_tools(
            str(tmp_path),
            verification_policy=policy,
            editable_paths=[str(allowed)],
        )
    }

    blocked_result = tools["edit"].run(str(blocked), "value = 1", "value = 2")
    allowed_result = tools["edit"].run(str(allowed), "value = 1", "value = 2")

    assert "REJECTED" in blocked_result
    assert "APPLIED" in allowed_result
    assert blocked.read_text() == "value = 1\n"
    assert allowed.read_text() == "value = 2\n"


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
