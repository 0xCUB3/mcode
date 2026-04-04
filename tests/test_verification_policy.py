from __future__ import annotations

from mcode.agent import verification
from mcode.llm import session as session_module


def test_normalize_verification_commands_from_metadata_dict() -> None:
    assert verification.normalize_verification_commands(
        {"test_cmds": [" pytest -q ", "", "python -m pytest -q"]}
    ) == ["pytest -q", "python -m pytest -q"]


def test_normalize_verification_commands_from_json_string() -> None:
    assert verification.normalize_verification_commands(
        {"verification_cmds": '["tox -q", "python -m pytest -q"]'}
    ) == ["tox -q", "python -m pytest -q"]


def test_normalize_verification_commands_handles_empty_metadata() -> None:
    assert verification.normalize_verification_commands(None) == []
    assert verification.normalize_verification_commands({}) == []


def test_build_verification_prompt_mentions_default_checks() -> None:
    prompt = verification.build_verification_prompt(["pytest -q", "python -m pytest -q"])

    assert "Start with `run_tests default`" in prompt
    assert "Do not run tests through `bash`" in prompt
    assert "Do not call `final_answer`" in prompt
    assert "`pytest -q`" in prompt


def test_build_budget_warning_blocks_submit_without_changes() -> None:
    warning = verification.build_budget_warning(
        has_changes=False,
        has_run_tests_tool=True,
        used_run_tests=False,
    )

    assert "working tree still has no code changes" in warning
    assert "Do not call `final_answer` yet" in warning


def test_build_budget_warning_requires_verification_before_submit() -> None:
    warning = verification.build_budget_warning(
        has_changes=True,
        has_run_tests_tool=True,
        used_run_tests=False,
    )

    assert "you have not run verification yet" in warning
    assert "Use `run_tests default` now" in warning
    assert "Do not call `final_answer` yet" in warning


def test_session_does_not_define_verification_policy_helpers() -> None:
    assert not hasattr(session_module, "_normalize_verification_commands")
    assert not hasattr(session_module, "_verification_prompt")


def test_build_tool_gate_message_mentions_capability_family_for_bash() -> None:
    message = verification.build_tool_gate_message(
        "bash",
        available_tools=["read_file", "search_code", "edit", "run_tests"],
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(),
        require_default_verification=True,
        requested_family="verification",
        route_mode="bundled_tool_fallback",
    )

    assert message is not None
    assert "`verification` capability family" in message
    assert "bundled tools" in message
    assert "Bash is an escape hatch" in message


def test_validate_verification_command_accepts_plain_pytest() -> None:
    assert verification.validate_verification_command("pytest -q tests/test_bug.py -k bug") is None


def test_validate_verification_command_blocks_shell_wrappers() -> None:
    message = verification.validate_verification_command(
        "cd /testbed && python -m pytest -q tests/test_bug.py"
    )

    assert message is not None
    assert "test command itself" in message or "plain verification command" in message


def test_validate_verification_command_blocks_pipes() -> None:
    message = verification.validate_verification_command(
        "python -m pytest -q tests/test_bug.py | head -20"
    )

    assert message is not None
    assert "Do not use pipes" in message


class _FakePhaseState:
    def __init__(self, *, progress: float, has_edit: bool) -> None:
        self.progress = progress
        self.has_edit = has_edit


def test_tighten_available_tools_pushes_earlier_edit() -> None:
    tools = verification.tighten_available_tools(
        ["search_code", "read_file", "find_file", "list_dir", "edit", "bash"],
        phase_state=_FakePhaseState(progress=0.55, has_edit=False),
        has_changes=False,
        verification_state=verification.VerificationState(),
        has_run_tests_tool=False,
    )

    assert "edit" in tools
    assert "read_file" in tools
    assert "search_code" not in tools
    assert "find_file" not in tools
    assert "list_dir" not in tools
    assert "bash" not in tools


def test_build_phase_guidance_mentions_plain_verification_after_block() -> None:
    message = verification.build_phase_guidance(
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(
            used_run_tests=True,
            blocked_verification_commands=1,
        ),
        phase_state=_FakePhaseState(progress=0.6, has_edit=True),
        require_default_verification=False,
    )

    assert message is not None
    assert "plain command only" in message
    assert "pipes" in message


def test_build_run_tests_tool_blocks_wrapped_command(tmp_path) -> None:
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return f"ran {command}"

    tool = verification.build_run_tests_tool(
        repo_root=str(tmp_path),
        test_cmds=["pytest -q tests/test_bug.py"],
        command_fn=command_fn,
    )

    result = tool.run("cd /testbed && python -m pytest -q tests/test_bug.py")

    assert calls == []
    assert "BLOCKED" in result
    assert "test command itself" in result or "plain verification command" in result


def test_build_run_tests_tool_runs_plain_command(tmp_path) -> None:
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return f"ran {command}"

    tool = verification.build_run_tests_tool(
        repo_root=str(tmp_path),
        test_cmds=["pytest -q tests/test_bug.py"],
        command_fn=command_fn,
    )

    result = tool.run("pytest -q tests/test_bug.py -k bug")

    assert calls == ["pytest -q tests/test_bug.py -k bug"]
    assert "COMPLETED" in result
