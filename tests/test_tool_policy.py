from mcode.agent.tool_policy import ToolPolicyState, check_tool_call


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
