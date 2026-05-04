from mcode.bench.runner import _should_reset_before_retry


def test_should_not_reset_repairable_compile_errors() -> None:
    output = "./zebra_puzzle.go:150:6: horseIdx declared and not used"

    assert not _should_reset_before_retry(output)


def test_should_reset_structural_syntax_errors() -> None:
    output = "error: unclosed delimiter"

    assert _should_reset_before_retry(output)
