from __future__ import annotations

from mcode.llm.session import _normalize_verification_commands


def test_normalize_verification_commands_from_metadata_dict() -> None:
    assert _normalize_verification_commands(
        {"test_cmds": [" pytest -q ", "", "python -m pytest -q"]}
    ) == ["pytest -q", "python -m pytest -q"]


def test_normalize_verification_commands_from_json_string() -> None:
    assert _normalize_verification_commands(
        {"verification_cmds": '["tox -q", "python -m pytest -q"]'}
    ) == ["tox -q", "python -m pytest -q"]


def test_normalize_verification_commands_handles_empty_metadata() -> None:
    assert _normalize_verification_commands(None) == []
    assert _normalize_verification_commands({}) == []
