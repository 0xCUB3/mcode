from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationPolicy:
    test_cmds: list[str]
    test_fn: Callable[[str], str] | None
    prompt_block: str

    def tool_kwargs(self) -> dict[str, object]:
        return {
            "test_cmds": self.test_cmds,
            "test_fn": self.test_fn,
        }


@dataclass(frozen=True)
class VerificationState:
    used_run_tests: bool = False
    successful_run_tests: bool = False
    used_default_run_tests: bool = False
    successful_default_run_tests: bool = False


def normalize_verification_commands(source: object | None) -> list[str]:
    if source is None:
        return []

    if isinstance(source, dict):
        for key in ("test_cmds", "verification_cmds", "verification", "commands"):
            if key in source:
                return normalize_verification_commands(source[key])
        return []

    if isinstance(source, str):
        text = source.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return [text]
        return normalize_verification_commands(parsed)

    if isinstance(source, (list, tuple, set)):
        cmds: list[str] = []
        for item in source:
            text = str(item).strip()
            if text:
                cmds.append(text)
        return cmds

    text = str(source).strip()
    return [text] if text else []


def build_verification_prompt(test_cmds: list[str]) -> str:
    if test_cmds:
        formatted = ", ".join(f"`{cmd}`" for cmd in test_cmds[:3])
        more = "" if len(test_cmds) <= 3 else f" and {len(test_cmds) - 3} more"
        return (
            "\n\nVerification:\n"
            "You have a `run_tests` tool. Start with `run_tests default` "
            f"to run the task-default checks ({formatted}{more}). Keep "
            "verification cheap, if you need a narrower check, use a "
            "targeted command instead of a broad suite. Do not run tests "
            "through `bash` when `run_tests` is available. Do not call "
            "`final_answer` after editing until you have attempted "
            "verification with `run_tests`."
        )

    return (
        "\n\nVerification:\n"
        "You still have a `run_tests` tool even though there are no "
        "task-default commands. Use it with the cheapest shell command "
        "that exercises your change, such as `pytest -q path/to/test.py -k "
        "name` or `python -m pytest -q path/to/test.py -k name`. Avoid "
        "full-suite runs unless necessary, and do not run tests through "
        "`bash` when `run_tests` is available. Do not call `final_answer` "
        "after editing until you have attempted verification with "
        "`run_tests`."
    )


def build_budget_warning(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    used_run_tests: bool,
) -> str:
    if not has_changes:
        return (
            "WARNING: You have 2 turns left and your working tree "
            "still has no code changes. Make at least one edit now. "
            "Do not call `final_answer` yet."
        )

    if has_run_tests_tool and not used_run_tests:
        return (
            "WARNING: You have 2 turns left and you have not run "
            "verification yet. Use `run_tests default` now, or a "
            "narrower `run_tests <command>` if needed. Do not call "
            "`final_answer` yet."
        )

    return (
        "WARNING: You have 2 turns left. If your edit is already "
        "verified, call `final_answer` now. If verification is still "
        "failing or missing, do not call `final_answer` yet."
    )


def verification_state_from_event_log(event_log: object | None) -> VerificationState:
    if event_log is None:
        return VerificationState()

    to_dicts = getattr(event_log, "to_dicts", None)
    if not callable(to_dicts):
        return VerificationState()

    pending_test_cmd: str | None = None
    state = VerificationState()
    for event in to_dicts():
        if not isinstance(event, dict):
            continue
        if event.get("tool_name") != "run_tests":
            continue

        if event.get("kind") == "tool_call":
            arguments = event.get("arguments", {})
            if isinstance(arguments, dict):
                pending_test_cmd = str(arguments.get("test_cmd", "")).strip().lower()
                if pending_test_cmd == "default":
                    state = VerificationState(
                        used_run_tests=True,
                        successful_run_tests=state.successful_run_tests,
                        used_default_run_tests=True,
                        successful_default_run_tests=state.successful_default_run_tests,
                    )
                else:
                    state = VerificationState(
                        used_run_tests=True,
                        successful_run_tests=state.successful_run_tests,
                        used_default_run_tests=state.used_default_run_tests,
                        successful_default_run_tests=state.successful_default_run_tests,
                    )
            continue

        if event.get("kind") != "tool_result":
            continue

        output = str(event.get("output", ""))
        success = (
            "PASSED" in output
            and "FAILED" not in output
            and "TIMEOUT" not in output
            and "ERROR" not in output
        )
        if not success:
            pending_test_cmd = None
            continue
        if pending_test_cmd == "default":
            state = VerificationState(
                used_run_tests=True,
                successful_run_tests=True,
                used_default_run_tests=True,
                successful_default_run_tests=True,
            )
        else:
            state = VerificationState(
                used_run_tests=True,
                successful_run_tests=True,
                used_default_run_tests=state.used_default_run_tests,
                successful_default_run_tests=state.successful_default_run_tests,
            )
        pending_test_cmd = None

    return state


@dataclass(frozen=True)
class _LocalToolInvocation:
    name: str
    status: str = "completed"


@dataclass(frozen=True)
class _LocalToolPhaseState:
    turn: int
    budget: int
    invocations: tuple[_LocalToolInvocation, ...] = ()

    @property
    def has_edit(self) -> bool:
        return any(call.name == "edit" for call in self.invocations)


def tool_phase_state_from_event_log(
    event_log: object | None,
    *,
    turn: int,
    budget: int,
):
    try:
        from mellea.agent.strategy import ToolInvocation, ToolPhaseState
    except ImportError:
        ToolInvocation = _LocalToolInvocation
        ToolPhaseState = _LocalToolPhaseState

    if event_log is None:
        return ToolPhaseState(turn=turn, budget=budget)

    to_dicts = getattr(event_log, "to_dicts", None)
    if not callable(to_dicts):
        return ToolPhaseState(turn=turn, budget=budget)

    invocations: list[ToolInvocation] = []
    for event in to_dicts():
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "tool_result":
            continue
        tool_name = event.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        status = str(event.get("status", "completed") or "completed")
        invocations.append(ToolInvocation(tool_name, status=status))

    return ToolPhaseState(
        turn=turn,
        budget=budget,
        invocations=tuple(invocations),
    )


def build_phase_guidance(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    phase_state,
    require_default_verification: bool,
) -> str | None:
    if not has_changes and not phase_state.has_edit:
        return (
            "Phase: diagnose, then edit. Stop widening the search. Pick the single most likely "
            "file and make one concrete edit this turn."
        )

    if not has_run_tests_tool or not has_changes:
        return None

    if require_default_verification and not verification_state.used_default_run_tests:
        return (
            "Phase: verify. Run `run_tests default` now. Do not call `final_answer` before the "
            "task-default verification runs."
        )

    if not require_default_verification and not verification_state.used_run_tests:
        return (
            "Phase: verify. Run `run_tests` with the cheapest command that exercises your "
            "change before you keep editing or submit."
        )

    if not verification_state.successful_run_tests:
        return (
            "Phase: repair and re-verify. Your current verification has not passed yet. Keep "
            "iterating until `run_tests` succeeds."
        )

    return None


def build_tool_gate_message(
    tool_name: str,
    *,
    available_tools: list[str],
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    require_default_verification: bool,
) -> str | None:
    if tool_name == "final_answer":
        return build_submit_block_message(
            has_changes=has_changes,
            has_run_tests_tool=has_run_tests_tool,
            verification_state=verification_state,
            require_default_verification=require_default_verification,
        )

    if tool_name in available_tools:
        return None

    if tool_name == "bash":
        return (
            "Bash is an escape hatch here. Use `read_file`, `search_code`, `edit`, and "
            "`run_tests` for the main loop unless you truly need a shell-only command."
        )

    return None


def build_submit_block_message(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    require_default_verification: bool,
) -> str | None:
    if not has_changes:
        return (
            "You still have no code changes. Make at least one edit before calling `final_answer`."
        )

    if not has_run_tests_tool:
        return None

    if require_default_verification:
        if not verification_state.used_default_run_tests:
            return (
                "Before calling `final_answer`, run `run_tests default` and use that "
                "result to verify your patch."
            )
        if not verification_state.successful_default_run_tests:
            return (
                "Do not call `final_answer` yet. `run_tests default` has not passed. "
                "Fix the code until the task-default verification succeeds."
            )
        return None

    if not verification_state.used_run_tests:
        return (
            "Before calling `final_answer`, run `run_tests` with the cheapest command "
            "that exercises your change."
        )
    if not verification_state.successful_run_tests:
        return (
            "Do not call `final_answer` yet. Your verification has not passed. Fix the "
            "code until `run_tests` succeeds."
        )
    return None


def build_verification_policy(
    *,
    repo_root: str,
    test_cmds: object | None = None,
    test_fn: Callable[[str], str] | None = None,
    timeout_s: int | None = None,
) -> VerificationPolicy:
    verification_cmds = normalize_verification_commands(test_cmds)
    del repo_root, timeout_s

    return VerificationPolicy(
        test_cmds=verification_cmds,
        test_fn=test_fn,
        prompt_block=build_verification_prompt(verification_cmds),
    )
