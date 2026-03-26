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
        "verified, call `final_answer` now. If not, make your best "
        "fix and call `final_answer` immediately."
    )


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
