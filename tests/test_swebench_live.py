from __future__ import annotations

import pytest

from mcode.execution.swebench_live import (
    _check_resolution,
    _ms_image_name,
    _parse_pytest_output,
)


def test_parse_pytest_output_basic():
    output = (
        "PASSED test_foo.py::test_one\n"
        "FAILED test_foo.py::test_two\n"
        "ERROR test_foo.py::test_three\n"
    )
    result = _parse_pytest_output(output)
    assert result["test_foo.py::test_one"] == "PASSED"
    assert result["test_foo.py::test_two"] == "FAILED"
    assert result["test_foo.py::test_three"] == "ERROR"


def test_parse_pytest_output_prefix_format():
    output = "PASSED test_bar.py::test_alpha\nFAILED test_bar.py::test_beta\n"
    result = _parse_pytest_output(output)
    assert result["test_bar.py::test_alpha"] == "PASSED"
    assert result["test_bar.py::test_beta"] == "FAILED"


def test_parse_pytest_output_strips_error_message():
    """FAILED lines have ' - ' replaced with ' ', then split on whitespace."""
    output = (
        "PASSED tests/test_base.py::test_set\n"
        "FAILED tests/test_base.py::test_get_item - KeyError: 'DOTENV_INT does not exist'\n"
    )
    result = _parse_pytest_output(output)
    assert result["tests/test_base.py::test_set"] == "PASSED"
    assert result["tests/test_base.py::test_get_item"] == "FAILED"


def test_parse_pytest_output_verbose_lines_ignored():
    """Verbose output (test_id PASSED [NN%]) is ignored; only -rA summary is parsed."""
    output = (
        "tests/foo.py::test_a PASSED [ 19%]\n"
        "tests/foo.py::test_b FAILED [ 20%]\n"
        "PASSED tests/foo.py::test_a\n"
        "FAILED tests/foo.py::test_b\n"
    )
    result = _parse_pytest_output(output)
    assert result["tests/foo.py::test_a"] == "PASSED"
    assert result["tests/foo.py::test_b"] == "FAILED"
    assert len(result) == 2


def test_parse_pytest_output_parametrized_with_spaces():
    """Parametrized test IDs with spaces get truncated at first space (matches official)."""
    output = "PASSED tests/test_foo.py::test_validate[A long name]\n"
    result = _parse_pytest_output(output)
    assert result["tests/test_foo.py::test_validate[A"] == "PASSED"


def test_parse_pytest_output_empty():
    assert _parse_pytest_output("") == {}
    assert _parse_pytest_output("some random output\nno test results here\n") == {}


def test_check_resolution_all_pass():
    test_results = {
        "test_a": "PASSED",
        "test_b": "PASSED",
        "test_c": "PASSED",
    }
    report = _check_resolution(
        test_results,
        fail_to_pass=["test_a"],
        pass_to_pass=["test_b", "test_c"],
    )
    assert report["resolved"] is True
    assert report["fail_to_pass"]["test_a"] == "PASSED"
    assert report["pass_to_pass"]["test_b"] == "PASSED"


def test_check_resolution_fail_still_fails():
    test_results = {
        "test_a": "FAILED",
        "test_b": "PASSED",
    }
    report = _check_resolution(
        test_results,
        fail_to_pass=["test_a"],
        pass_to_pass=["test_b"],
    )
    assert report["resolved"] is False
    assert report["fail_to_pass"]["test_a"] == "FAILED"


def test_check_resolution_p2p_regression_blocks_resolution():
    """P2P failures block resolution (matches official SWE-bench-Live spec)."""
    test_results = {
        "test_a": "PASSED",
        "test_b": "FAILED",
    }
    report = _check_resolution(
        test_results,
        fail_to_pass=["test_a"],
        pass_to_pass=["test_b"],
    )
    assert report["resolved"] is False
    assert report["pass_to_pass"]["test_b"] == "FAILED"
    assert report["p2p_regressions"] == ["test_b"]


def test_check_resolution_missing_p2p_is_ok():
    """MISSING P2P tests don't block resolution (dataset IDs often unmatchable)."""
    test_results = {"test_a": "PASSED"}
    report = _check_resolution(
        test_results,
        fail_to_pass=["test_a"],
        pass_to_pass=["test_missing"],
    )
    assert report["resolved"] is True
    assert report["pass_to_pass"]["test_missing"] == "MISSING"


def test_check_resolution_missing_f2p_blocks():
    """MISSING F2P tests DO block resolution (must actually pass)."""
    test_results = {"test_b": "PASSED"}
    report = _check_resolution(
        test_results,
        fail_to_pass=["test_missing"],
        pass_to_pass=["test_b"],
    )
    assert report["resolved"] is False
    assert report["fail_to_pass"]["test_missing"] == "MISSING"


def test_check_resolution_empty_fail_to_pass():
    report = _check_resolution(
        {"test_a": "PASSED"},
        fail_to_pass=[],
        pass_to_pass=["test_a"],
    )
    assert report["resolved"] is False


def test_ms_image_name():
    assert _ms_image_name("django__django__4.0") == (
        "starryzhang/sweb.eval.x86_64.django_1776_django_1776_4.0"
    )


def test_ms_image_name_double_underscore():
    assert _ms_image_name("sympy__sympy__1.0") == (
        "starryzhang/sweb.eval.x86_64.sympy_1776_sympy_1776_1.0"
    )


def test_ms_image_name_uppercase():
    assert _ms_image_name("Django__Django__4.0") == (
        "starryzhang/sweb.eval.x86_64.django_1776_django_1776_4.0"
    )


def test_ms_image_name_mixed():
    assert _ms_image_name("Repo__Name__v2") == (
        "starryzhang/sweb.eval.x86_64.repo_1776_name_1776_v2"
    )


def test_load_swebench_live_missing_datasets(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from mcode.bench.swebench_live import load_swebench_live

    with pytest.raises(RuntimeError, match="datasets"):
        load_swebench_live(None, split="verified", limit=1)
