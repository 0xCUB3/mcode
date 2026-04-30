from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from mcode.execution.sandbox import DockerUnavailableError
from mcode.execution.swebench_live import (
    SWEbenchLiveSandbox,
    _build_agent_shell_command,
    _check_resolution,
    _exec_in_container,
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
        "docker.io/starryzhang/sweb.eval.x86_64.django_1776_django_1776_4.0"
    )


def test_ms_image_name_double_underscore():
    assert _ms_image_name("sympy__sympy__1.0") == (
        "docker.io/starryzhang/sweb.eval.x86_64.sympy_1776_sympy_1776_1.0"
    )


def test_ms_image_name_uppercase():
    assert _ms_image_name("Django__Django__4.0") == (
        "docker.io/starryzhang/sweb.eval.x86_64.django_1776_django_1776_4.0"
    )


def test_ms_image_name_mixed():
    assert _ms_image_name("Repo__Name__v2") == (
        "docker.io/starryzhang/sweb.eval.x86_64.repo_1776_name_1776_v2"
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


def test_build_agent_shell_command_activates_testbed_and_rewrites_repo_root():
    command = "cd /tmp/mcode-testbed-123/testbed && python -m pytest -q"
    wrapped = _build_agent_shell_command(
        command,
        host_repo_root="/tmp/mcode-testbed-123/testbed",
    )

    assert "source /opt/miniconda3/bin/activate" in wrapped
    assert "conda activate testbed" in wrapped
    assert "git config --global --add safe.directory /testbed" in wrapped
    assert "cd /testbed && python -m pytest -q" in wrapped
    assert "/tmp/mcode-testbed-123/testbed" not in wrapped


def test_exec_in_container_raises_docker_unavailable() -> None:
    class BrokenContainer:
        def exec_run(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("podman.sock connection reset")

    with pytest.raises(DockerUnavailableError, match="SWE-bench Live evaluation"):
        _exec_in_container(BrokenContainer(), "pytest", timeout_s=1)


def test_swebench_live_evaluate_patch_propagates_test_timeout(monkeypatch) -> None:
    task = SimpleNamespace(
        instance_id="django__django__4.0",
        test_patch="",
        test_cmds=["python -m pytest tests/test_fix.py"],
        fail_to_pass=["tests/test_fix.py::test_fix"],
        pass_to_pass=[],
    )

    class FakeImages:
        def get(self, name):
            assert name == _ms_image_name(task.instance_id)

    class FakeExecResult:
        def __init__(self, output: bytes, exit_code: int) -> None:
            self.output = output
            self.exit_code = exit_code

    class FakeContainer:
        def start(self) -> None:
            pass

        def put_archive(self, dest, data) -> None:
            del dest, data

        def exec_run(self, argv, *, workdir):
            del workdir
            command = argv[-1]
            if "git apply" in command:
                return FakeExecResult(b"applied\n", 0)
            time.sleep(0.05)
            return FakeExecResult(b"PASSED tests/test_fix.py::test_fix\n", 0)

        def remove(self, *, force: bool) -> None:
            del force

    class FakeContainers:
        def create(self, **kwargs):
            return FakeContainer()

    fake_client = SimpleNamespace(images=FakeImages(), containers=FakeContainers())
    sandbox = SWEbenchLiveSandbox()
    monkeypatch.setattr(sandbox, "_get_client", lambda: fake_client)

    result = sandbox.evaluate_patch(
        task=task,
        patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
        run_id="test-run",
        timeout_s=0.001,
    )

    assert result.timed_out is True
    assert result.resolved is False
    assert "Command timed out" in result.test_output
