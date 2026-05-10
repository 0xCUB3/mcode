from mcode.agent.tool_policy import (
    ToolPolicyState,
    blocked_shell_command_reason,
    blocked_verification_command_reason,
    check_edit_path,
    check_tool_call,
)


def test_policy_requires_tests_after_edit() -> None:
    state = ToolPolicyState(must_run_tests_now=True)

    decision = check_tool_call("read_file", state)

    assert not decision.allowed
    assert decision.kind == "invalid"
    assert "run_tests is required" in decision.reason


def test_policy_requires_edit_when_localization_budget_is_spent() -> None:
    state = ToolPolicyState(must_edit_now=True)

    decision = check_tool_call("search_code", state)

    assert not decision.allowed
    assert decision.kind == "invalid"
    assert "edit is required" in decision.reason


def test_policy_blocks_final_answer_before_verification() -> None:
    state = ToolPolicyState(
        has_run_tests_tool=True,
        verification_succeeded=False,
        finalizer_tool_name="final_answer",
    )

    decision = check_tool_call("final_answer", state)

    assert not decision.allowed
    assert decision.kind == "blocked_finalizer"
    assert "verification" in decision.reason


def test_policy_allows_final_answer_after_verification() -> None:
    state = ToolPolicyState(
        has_run_tests_tool=True,
        verification_succeeded=True,
        finalizer_tool_name="final_answer",
    )

    decision = check_tool_call("final_answer", state)

    assert decision.allowed


def test_policy_rejects_edits_outside_allowed_paths() -> None:
    decision = check_edit_path("build.gradle", allowed_edit_paths={"src/main.py"})

    assert not decision.allowed
    assert "src/main.py" in decision.reason


def test_policy_allows_edit_when_no_path_allowlist() -> None:
    decision = check_edit_path("build.gradle", allowed_edit_paths=None)

    assert decision.allowed


def test_policy_blocks_dangerous_shell_prefixes() -> None:
    assert blocked_shell_command_reason("sudo rm harmless")
    assert blocked_shell_command_reason("rm -rf /")
    assert blocked_shell_command_reason("pytest -q") is None


def test_policy_blocks_evasive_verification_commands() -> None:
    assert blocked_verification_command_reason("pytest -q || true")
    assert blocked_verification_command_reason("pytest -q --ignore tests/test_bug.py")
    assert blocked_verification_command_reason("pytest tests/test_bug.py") is None
