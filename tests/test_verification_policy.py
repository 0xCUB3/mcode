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
    assert "`probe_python`" in prompt
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
    assert "`probe_python`" in message


def test_build_tool_gate_message_pushes_probe_python_to_run_tests_after_edit() -> None:
    message = verification.build_tool_gate_message(
        "probe_python",
        available_tools=["read_file", "edit", "run_tests"],
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(),
        require_default_verification=False,
        requested_family="verification",
        route_mode="bundled_tool_fallback",
    )

    assert message is not None
    assert "no longer available after edits" in message
    assert "Run `run_tests` now" in message


def test_build_tool_gate_message_pushes_blocked_bash_toward_test_discovery() -> None:
    message = verification.build_tool_gate_message(
        "bash",
        available_tools=["read_file", "search_code", "find_file", "list_dir", "edit", "run_tests"],
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(),
        require_default_verification=False,
        requested_family="shell",
        route_mode="bundled_tool_fallback",
    )

    assert message is not None
    assert "`find_file`" in message
    assert "`list_dir`" in message
    assert "`run_tests`" in message


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


def test_classify_verification_command_distinguishes_probe_runner_and_script() -> None:
    assert verification.classify_verification_command("python -c 'print(1)'") == "probe"
    assert verification.classify_verification_command("python -m pytest -q tests/test_bug.py") == (
        "runner"
    )
    assert verification.classify_verification_command("python tests/test_bug.py") == "script"


def test_verification_state_requires_submit_eligible_success_after_edit() -> None:
    event_log = type(
        "FakeEventLog",
        (),
        {
            "to_dicts": lambda self: [
                {"kind": "summary", "metadata": {"kind": "edit_started", "turn": 2}},
                {
                    "kind": "tool_call",
                    "tool_name": "run_tests",
                    "arguments": {"test_cmd": "python -c 'print(1)'"},
                },
                {
                    "kind": "tool_result",
                    "tool_name": "run_tests",
                    "output": "$ python -c 'print(1)'\nPASSED\n1 passed",
                },
            ]
        },
    )()

    state = verification.verification_state_from_event_log(event_log)

    assert state.successful_run_tests is True
    assert state.successful_submit_eligible_run_tests is False
    assert state.successful_submit_eligible_after_edit is False


def test_verification_state_tracks_post_edit_probe_budget_until_run_tests() -> None:
    event_log = type(
        "FakeEventLog",
        (),
        {
            "to_dicts": lambda self: [
                {"kind": "summary", "metadata": {"kind": "edit_started", "turn": 2}},
                {
                    "kind": "tool_call",
                    "tool_name": "probe_python",
                    "arguments": {"code": "print(1)"},
                },
                {"kind": "tool_call", "tool_name": "read_file", "arguments": {"path": "foo.py"}},
                {
                    "kind": "tool_call",
                    "tool_name": "probe_python",
                    "arguments": {"code": "print(2)"},
                },
            ]
        },
    )()

    state = verification.verification_state_from_event_log(event_log)

    assert state.post_edit_probe_calls == 2
    assert state.post_edit_run_tests_calls == 0


def test_build_submit_block_message_requires_runner_style_check_after_edit() -> None:
    message = verification.build_submit_block_message(
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(
            used_run_tests=True,
            successful_run_tests=True,
            successful_submit_eligible_run_tests=False,
            successful_submit_eligible_after_edit=False,
        ),
        require_default_verification=False,
    )

    assert message is not None
    assert "runner-style" in message
    assert "`python -c`" in message


class _FakePhaseState:
    def __init__(
        self,
        *,
        progress: float,
        has_edit: bool,
        last_tool_name: str | None = None,
        repeated_tool_streak: int = 0,
    ) -> None:
        self.progress = progress
        self.has_edit = has_edit
        self.last_tool_name = last_tool_name
        self.repeated_tool_streak = repeated_tool_streak


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


def test_tighten_available_tools_cuts_repeated_search_churn() -> None:
    tools = verification.tighten_available_tools(
        ["search_code", "read_file", "find_file", "list_dir", "edit", "bash"],
        phase_state=_FakePhaseState(
            progress=0.4,
            has_edit=False,
            last_tool_name="search_code",
            repeated_tool_streak=3,
        ),
        has_changes=False,
        verification_state=verification.VerificationState(),
        has_run_tests_tool=False,
    )

    assert "edit" in tools
    assert "read_file" in tools
    assert "search_code" not in tools
    assert "find_file" not in tools
    assert "list_dir" not in tools


def test_tighten_available_tools_forces_run_tests_after_repeated_probe_loop() -> None:
    tools = verification.tighten_available_tools(
        ["read_file", "edit", "probe_python", "run_tests", "bash"],
        phase_state=_FakePhaseState(
            progress=0.7,
            has_edit=True,
            last_tool_name="probe_python",
            repeated_tool_streak=1,
        ),
        has_changes=True,
        verification_state=verification.VerificationState(post_edit_probe_calls=1),
        has_run_tests_tool=True,
    )

    assert "run_tests" in tools
    assert "edit" in tools
    assert "probe_python" not in tools
    assert "bash" not in tools


def test_tighten_available_tools_keeps_probe_before_any_edit() -> None:
    tools = verification.tighten_available_tools(
        ["read_file", "edit", "probe_python", "run_tests", "bash"],
        phase_state=_FakePhaseState(
            progress=0.3,
            has_edit=False,
        ),
        has_changes=False,
        verification_state=verification.VerificationState(),
        has_run_tests_tool=True,
    )

    assert "probe_python" in tools


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


def test_build_phase_guidance_stops_post_edit_probe_loop() -> None:
    message = verification.build_phase_guidance(
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.VerificationState(),
        phase_state=_FakePhaseState(progress=0.6, has_edit=True),
        require_default_verification=False,
    )

    assert message is not None
    assert "already have a patch" in message
    assert "`run_tests` now" in message


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


def test_build_probe_python_tool_runs_code_via_python_c(tmp_path) -> None:
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return "probe output"

    tool = verification.build_probe_python_tool(
        repo_root=str(tmp_path),
        command_fn=command_fn,
    )

    result = tool.run("print('hello')")

    assert len(calls) == 1
    assert calls[0].startswith("python -c ")
    assert "COMPLETED" in result
    assert "probe output" in result
