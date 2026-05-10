from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

DecisionKind = Literal["allowed", "invalid", "blocked_finalizer"]

DANGEROUS_SHELL_PREFIXES = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){",
    "chmod -R 777 /",
    "sudo",
    "reboot",
    "shutdown",
    "kill -9 -1",
    "pkill",
)

EVASIVE_VERIFICATION_MARKERS = (
    " -k not ",
    " -k 'not ",
    ' -k "not ',
    " --ignore=",
    " --ignore ",
    " || true",
    " | head",
    " | tail",
    " | grep",
)


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

    This covers turn-level control-loop gates. Static command and path checks
    are below so tool implementations can share the same policy vocabulary.
    """

    if state.must_run_tests_now and tool_name != "run_tests":
        return ToolPolicyDecision(
            allowed=False,
            kind="invalid",
            reason=(
                "run_tests is required now because source files changed since the last verification"
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


def check_edit_path(
    normalized_edit_path: str,
    *,
    allowed_edit_paths: Collection[str] | None,
) -> ToolPolicyDecision:
    if allowed_edit_paths is None or normalized_edit_path in allowed_edit_paths:
        return ToolPolicyDecision(allowed=True)
    allowed = ", ".join(sorted(allowed_edit_paths))
    return ToolPolicyDecision(
        allowed=False,
        kind="invalid",
        reason=f"edit only the benchmark implementation file(s): {allowed}",
    )


def blocked_shell_command_reason(command: str) -> str | None:
    cmd_lower = command.strip().lower()
    for blocked in DANGEROUS_SHELL_PREFIXES:
        if cmd_lower.startswith(blocked):
            return f"Error: command blocked for safety: {command[:80]}"
    return None


def blocked_verification_command_reason(command: str) -> str | None:
    normalized = f" {command.lower()} "
    if any(marker in normalized for marker in EVASIVE_VERIFICATION_MARKERS):
        return (
            "Verification commands must run the relevant failing tests without skipping, "
            "excluding, or masking their exit status. Run the direct failing test command "
            "instead."
        )
    return None
