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
