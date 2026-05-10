from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecisionKind = Literal["allowed", "invalid", "blocked_finalizer"]


@dataclass(frozen=True)
class ToolPolicyState:
    """Runtime state used to decide whether a model-requested tool may run."""

    must_edit_now: bool = False
    must_run_tests_now: bool = False
    has_run_tests_tool: bool = False
    verification_succeeded: bool = False
    finalizer_tool_name: str = "final_answer"


@dataclass(frozen=True)
class ToolPolicyDecision:
    allowed: bool
    kind: DecisionKind = "allowed"
    reason: str = ""


def check_tool_call(tool_name: str, state: ToolPolicyState) -> ToolPolicyDecision:
    """Return whether a requested tool call is allowed in the current turn.

    This is intentionally small: it only covers turn-level control-loop gates.
    Tool-specific argument validation still lives with the tool implementation.
    """

    if state.must_run_tests_now and tool_name != "run_tests":
        return ToolPolicyDecision(
            allowed=False,
            kind="invalid",
            reason=(
                "run_tests is required now because source files changed "
                "since the last verification"
            ),
        )
    if state.must_edit_now and tool_name != "edit":
        return ToolPolicyDecision(
            allowed=False,
            kind="invalid",
            reason="edit is required now because no source file has been changed yet",
        )
    if (
        tool_name == state.finalizer_tool_name
        and state.has_run_tests_tool
        and not state.verification_succeeded
    ):
        return ToolPolicyDecision(
            allowed=False,
            kind="blocked_finalizer",
            reason="final_answer requires successful verification first",
        )
    return ToolPolicyDecision(allowed=True)
