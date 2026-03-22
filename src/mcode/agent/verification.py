from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationPolicy:
    test_cmds: list[str]
    test_fn: Callable[[str], str]
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
            "targeted command instead of a broad suite."
        )

    return (
        "\n\nVerification:\n"
        "You still have a `run_tests` tool even though there are no "
        "task-default commands. Use it with the cheapest shell command "
        "that exercises your change, such as `pytest -q path/to/test.py -k "
        "name` or `python -m pytest -q path/to/test.py -k name`. Avoid "
        "full-suite runs unless necessary."
    )


def build_verification_policy(
    *,
    repo_root: str,
    test_cmds: object | None = None,
    test_fn: Callable[[str], str] | None = None,
    timeout_s: int | None = None,
) -> VerificationPolicy:
    verification_cmds = normalize_verification_commands(test_cmds)
    verification_timeout_s = timeout_s if timeout_s is not None else int(
        os.environ.get("MCODE_REACT_TIMEOUT", "120")
    )

    if test_fn is None:

        def verification_test_fn(test_cmd: str = "default") -> str:
            command = test_cmd.strip()
            if command.lower() == "default":
                if not verification_cmds:
                    return (
                        "No task-default verification commands available. "
                        "Pass an explicit shell command, such as "
                        "`pytest -q` or `python -m pytest -q path/to/test.py -k name`."
                    )
                cmds = verification_cmds
            else:
                cmds = [command]

            outputs: list[str] = []
            for cmd in cmds:
                if not cmd.strip():
                    continue
                try:
                    result = subprocess.run(
                        ["bash", "-lc", cmd],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        timeout=verification_timeout_s,
                    )
                    out = result.stdout + result.stderr
                    status = (
                        "PASSED"
                        if result.returncode == 0
                        else f"FAILED (exit {result.returncode})"
                    )
                    outputs.append(f"$ {cmd}\n{status}\n{out}")
                except subprocess.TimeoutExpired:
                    outputs.append(f"$ {cmd}\nTIMEOUT after {verification_timeout_s}s")
                except OSError as e:
                    outputs.append(f"$ {cmd}\nError: {e}")

            return "\n---\n".join(outputs) if outputs else "No verification commands available."

    else:
        verification_test_fn = test_fn

    return VerificationPolicy(
        test_cmds=verification_cmds,
        test_fn=verification_test_fn,
        prompt_block=build_verification_prompt(verification_cmds),
    )
