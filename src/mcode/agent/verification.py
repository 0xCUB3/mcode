from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from mellea.backends.tools import MelleaTool

from mcode.agent.tooling import execute_command, format_tool_result, is_tool_result
from mcode.mellea_compat import import_requirements


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
            f"to execute the task checks ({formatted}{more}), then narrow only if needed. "
            "Keep the `final_answer` text short."
        )
    return (
        "\n\nVerification:\n"
        "Use `run_tests` before `final_answer`. Pick the narrowest real test command that "
        "covers the edited path. Keep the `final_answer` text short."
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


def build_turn_requirements(
    *,
    verification_policy: VerificationPolicy,
    enforce_run_tests: bool,
) -> list[object]:
    if not enforce_run_tests:
        return []

    reqs = import_requirements()
    return [
        reqs.uses_tool("run_tests"),
        reqs.tool_arg_validator(
            "Use `run_tests default` or one of the declared test commands.",
            "run_tests",
            "test_cmd",
            lambda value: _valid_test_command(value, verification_policy.test_cmds),
        ),
    ]


def build_submission_requirements() -> list[object]:
    reqs = import_requirements()
    return [
        reqs.Requirement(
            "Return a concise structured submission.",
            validation_fn=reqs.simple_validate(
                lambda text: _valid_submission_text(text),
                reason="Return a concise structured submission.",
            ),
        )
    ]


def _valid_test_command(value: object, allowed_commands: list[str]) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if text.lower() == "default":
        return True
    if not allowed_commands:
        return True
    return text in allowed_commands


def _valid_submission_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 4000:
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    summary = str(parsed.get("summary", "")).strip()
    tests_ran = parsed.get("tests_ran", [])
    return bool(summary) and isinstance(tests_ran, list)


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

    tool = MelleaTool.from_callable(_run_tests, name="run_tests")
    return MelleaTool(
        name=tool.name,
        tool_call=_run_tests,
        as_json_tool=_patch_run_tests_schema(tool.as_json_tool),
    )


def _patch_run_tests_schema(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    patched = deepcopy(schema)
    function_schema = patched.get("function")
    if not isinstance(function_schema, dict):
        return patched
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return patched
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        if isinstance(properties.get("timeout_s"), dict):
            properties["timeout_s"]["type"] = "integer"
        if isinstance(properties.get("max_output_chars"), dict):
            properties["max_output_chars"]["type"] = "integer"
    parameters["required"] = ["test_cmd"]
    return patched
