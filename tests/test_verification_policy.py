from __future__ import annotations

from mellea.agent.runtime.events import ToolCallEvent, ToolResultEvent

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


class _FakeEventLog:
    def __init__(self, events):
        self._events = list(events)

    def to_dicts(self):
        return [event.as_dict() for event in self._events]


def test_verification_state_resets_after_edit() -> None:
    event_log = _FakeEventLog(
        [
            ToolCallEvent(tool_name="run_tests", arguments={"test_cmd": "default"}),
            ToolResultEvent(tool_name="run_tests", status="completed", output="PASSED"),
            ToolResultEvent(tool_name="edit", status="completed", output="APPLIED"),
        ]
    )

    state = verification.verification_state_from_event_log(event_log)

    assert state.used_run_tests is False
    assert state.successful_run_tests is False
    assert state.used_default_run_tests is False
    assert state.successful_default_run_tests is False


def test_submit_block_requires_default_after_targeted_success() -> None:
    event_log = _FakeEventLog(
        [
            ToolCallEvent(
                tool_name="run_tests",
                arguments={
                    "test_cmd": ("python -m pytest -q astropy/timeseries/tests/test_common.py")
                },
            ),
            ToolResultEvent(tool_name="run_tests", status="completed", output="PASSED"),
        ]
    )

    message = verification.build_submit_block_message(
        has_changes=True,
        has_run_tests_tool=True,
        verification_state=verification.verification_state_from_event_log(event_log),
        require_default_verification=True,
    )

    assert message == (
        "Before calling `final_answer`, run `run_tests default` and use that "
        "result to verify your patch."
    )
