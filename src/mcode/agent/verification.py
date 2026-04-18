from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from mellea.backends.tools import MelleaTool

from mcode.agent.tooling import execute_command, format_tool_result, is_tool_result


@dataclass(frozen=True)
class VerificationPolicy:
    test_cmds: list[str]
    test_fn: Callable[[str], str] | None
    command_fn: Callable[[str], str] | None
    prompt_block: str


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
        return [text for item in source if (text := str(item).strip())]
    text = str(source).strip()
    return [text] if text else []


def build_verification_prompt(test_cmds: list[str]) -> str:
    if test_cmds:
        formatted = ", ".join(f"`{cmd}`" for cmd in test_cmds[:3])
        more = "" if len(test_cmds) <= 3 else f" and {len(test_cmds) - 3} more"
        return (
            "\n\nVerification:\n"
            "Use `run_tests` before `final_answer`. Start with `run_tests default` "
            f"to execute the task checks ({formatted}{more}), then narrow only if needed."
        )
    return (
        "\n\nVerification:\n"
        "Use `run_tests` before `final_answer`. Pick the narrowest real test command that "
        "covers the edited path."
    )


def build_verification_policy(
    *,
    test_cmds: object | None = None,
    test_fn: Callable[[str], str] | None = None,
    command_fn: Callable[[str], str] | None = None,
) -> VerificationPolicy:
    verification_cmds = normalize_verification_commands(test_cmds)
    return VerificationPolicy(
        test_cmds=verification_cmds,
        test_fn=test_fn,
        command_fn=command_fn,
        prompt_block=build_verification_prompt(verification_cmds),
    )


def build_run_tests_tool(
    *,
    repo_root: str,
    verification_policy: VerificationPolicy,
):
    test_cmds = verification_policy.test_cmds
    test_fn = verification_policy.test_fn
    command_fn = verification_policy.command_fn
    if not test_cmds and test_fn is None and command_fn is None:
        return None

    def _run_tests(
        test_cmd: str = "default",
        timeout_s: int = 120,
        max_output_chars: int = 4000,
    ) -> str:
        if test_fn is not None:
            result = test_fn(test_cmd)
            if is_tool_result(result):
                return result
            return format_tool_result(test_cmd, "COMPLETED", result)

        commands = test_cmds if test_cmd.strip().lower() == "default" else [test_cmd]
        outputs: list[str] = []
        for command in commands:
            if not command.strip():
                continue
            if command_fn is not None:
                result = command_fn(command)
                if is_tool_result(result):
                    outputs.append(result)
                else:
                    outputs.append(format_tool_result(command, "COMPLETED", result))
                continue
            status, output = execute_command(
                command,
                repo_root=repo_root,
                timeout=timeout_s,
            )
            if len(output) > max_output_chars:
                output = output[-max_output_chars:]
            outputs.append(format_tool_result(command, status, output))
        if outputs:
            return "\n---\n".join(outputs)
        return format_tool_result(test_cmd, "SKIPPED", "No test commands available.")

    return MelleaTool.from_callable(_run_tests, name="run_tests")
