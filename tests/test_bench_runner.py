from types import SimpleNamespace

from mcode.bench.runner import _allowed_polyglot_test_commands, _should_reset_before_retry


def test_allowed_polyglot_commands_include_declared_sequence() -> None:
    prepared = SimpleNamespace(
        test_commands=("npm install --silent --no-audit --no-fund", "npm test --silent"),
        test_paths=(),
        task=SimpleNamespace(language="javascript"),
    )

    assert (
        "npm install --silent --no-audit --no-fund && npm test --silent"
        in _allowed_polyglot_test_commands(prepared)
    )


def test_should_not_reset_repairable_compile_errors() -> None:
    output = "./zebra_puzzle.go:150:6: horseIdx declared and not used"

    assert not _should_reset_before_retry(output)


def test_should_reset_structural_syntax_errors() -> None:
    output = "error: unclosed delimiter"

    assert _should_reset_before_retry(output)
