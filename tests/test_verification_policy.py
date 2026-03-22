from __future__ import annotations

def test_normalize_verification_commands_from_metadata_dict() -> None:
    from mcode.agent.verification import normalize_verification_commands

    assert normalize_verification_commands(
        {"test_cmds": [" pytest -q ", "", "python -m pytest -q"]}
    ) == ["pytest -q", "python -m pytest -q"]


def test_normalize_verification_commands_from_json_string() -> None:
    from mcode.agent.verification import normalize_verification_commands

    assert normalize_verification_commands(
        {"verification_cmds": '["tox -q", "python -m pytest -q"]'}
    ) == ["tox -q", "python -m pytest -q"]


def test_normalize_verification_commands_handles_empty_metadata() -> None:
    from mcode.agent.verification import normalize_verification_commands

    assert normalize_verification_commands(None) == []
    assert normalize_verification_commands({}) == []
