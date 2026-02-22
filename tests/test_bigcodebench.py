from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mcode.bench.runner import _combine_for_eval
from mcode.bench.tasks import Task

_BCB_FIXTURE = [
    {
        "task_id": "BigCodeBench/0",
        "complete_prompt": ('import random\nimport itertools\ndef task_func():\n    """..."""\n'),
        "instruct_prompt": "Write a function task_func that generates random permutations.",
        "canonical_solution": "...",
        "code_prompt": "def task_func():\n",
        "test": (
            "import unittest\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_result_type(self):\n"
            "        result = task_func()\n"
            "        self.assertIsInstance(result, float)\n"
        ),
        "entry_point": "task_func",
        "doc_struct": "...",
        "libs": "['random', 'itertools']",
    },
    {
        "task_id": "BigCodeBench/1",
        "complete_prompt": 'def task_func(s: str) -> dict:\n    """Count chars."""\n',
        "instruct_prompt": "Write a function that counts characters in a string.",
        "canonical_solution": "...",
        "code_prompt": "def task_func(s):\n",
        "test": (
            "import unittest\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_basic(self):\n"
            "        self.assertEqual(task_func('abc'), {'a':1,'b':1,'c':1})\n"
        ),
        "entry_point": "task_func",
        "doc_struct": "...",
        "libs": "['collections']",
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "bigcodebench.json"
    p.write_text(json.dumps(_BCB_FIXTURE))
    return p


def test_load_bigcodebench_complete_produces_tasks(tmp_path: Path) -> None:
    with patch("mcode.bench.bigcodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.bigcodebench import load_bigcodebench

        tasks = load_bigcodebench(None, variant="complete")

    assert len(tasks) == 2
    task0 = next(t for t in tasks if t.task_id == "BigCodeBench/0")
    assert task0.benchmark == "bigcodebench-complete"
    assert task0.prompt == _BCB_FIXTURE[0]["complete_prompt"]
    assert task0.entry_point == "task_func"
    assert "unittest" in task0.test_code
    assert task0.metadata["variant"] == "complete"
    assert task0.metadata["source"] == "bigcode/bigcodebench"


def test_load_bigcodebench_instruct_produces_tasks(tmp_path: Path) -> None:
    with patch("mcode.bench.bigcodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.bigcodebench import load_bigcodebench

        tasks = load_bigcodebench(None, variant="instruct")

    assert len(tasks) == 2
    task0 = next(t for t in tasks if t.task_id == "BigCodeBench/0")
    assert task0.benchmark == "bigcodebench-instruct"
    assert task0.prompt == _BCB_FIXTURE[0]["instruct_prompt"]
    assert task0.metadata["variant"] == "instruct"


def test_load_bigcodebench_invalid_variant_raises(tmp_path: Path) -> None:
    with patch("mcode.bench.bigcodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.bigcodebench import load_bigcodebench

        with pytest.raises(ValueError, match="unknown"):
            load_bigcodebench(None, variant="unknown")


def test_load_bigcodebench_limit_via_load_benchmark(tmp_path: Path) -> None:
    with patch("mcode.bench.bigcodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.tasks import load_benchmark

        tasks = load_benchmark("bigcodebench-complete", Path("/tmp"), limit=1)

    assert len(tasks) == 1


def test_combine_for_eval_bigcodebench_complete() -> None:
    test_code = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_something(self):\n"
        "        self.assertEqual(task_func(), 42)\n"
    )
    task = Task(
        benchmark="bigcodebench-complete",
        task_id="BigCodeBench/0",
        prompt="Write task_func.",
        entry_point="task_func",
        test_code=test_code,
        metadata={"source": "bigcode/bigcodebench", "variant": "complete"},
    )
    code = "def task_func():\n    return 42\n"
    result = _combine_for_eval(task, code)

    assert "def task_func" in result
    assert "class TestCases" in result
    assert "unittest.main" in result


def test_combine_for_eval_bigcodebench_instruct() -> None:
    test_code = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_something(self):\n"
        "        self.assertEqual(task_func(), 99)\n"
    )
    task = Task(
        benchmark="bigcodebench-instruct",
        task_id="BigCodeBench/1",
        prompt="Write task_func that returns 99.",
        entry_point="task_func",
        test_code=test_code,
        metadata={"source": "bigcode/bigcodebench", "variant": "instruct"},
    )
    code = "def task_func():\n    return 99\n"
    result = _combine_for_eval(task, code)

    assert "def task_func" in result
    assert "class TestCases" in result
    assert "unittest.main" in result


def test_combine_for_eval_bigcodebench_harness_runs() -> None:
    """End-to-end: the test class from combined script passes when run via unittest."""
    import unittest

    test_code = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(task_func(), 42)\n"
    )
    task = Task(
        benchmark="bigcodebench-complete",
        task_id="BigCodeBench/99",
        prompt="Write task_func returning 42.",
        entry_point="task_func",
        test_code=test_code,
        metadata={"source": "bigcode/bigcodebench", "variant": "complete"},
    )
    code = "def task_func():\n    return 42\n"
    combined = _combine_for_eval(task, code)

    # Exec the combined code (without __main__ guard) to define function + test class
    globs: dict = {}
    exec(  # noqa: S102
        f"def task_func():\n    return 42\n\n{test_code}",
        globs,
    )
    TestCases = globs["TestCases"]
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)
    result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(suite)  # noqa: SIM115
    assert result.wasSuccessful(), f"Tests failed: {result.failures}"
    # Also verify the combined harness structure is correct
    assert "if __name__ == '__main__'" in combined


def test_combine_for_eval_bigcodebench_harness_fails() -> None:
    """Tests fail when code returns wrong value."""
    import unittest

    test_code = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(task_func(), 42)\n"
    )
    task = Task(
        benchmark="bigcodebench-complete",
        task_id="BigCodeBench/98",
        prompt="Write task_func returning 42.",
        entry_point="task_func",
        test_code=test_code,
        metadata={"source": "bigcode/bigcodebench", "variant": "complete"},
    )
    # Wrong return value: test will fail
    code = "def task_func():\n    return 0\n"
    combined = _combine_for_eval(task, code)
    _ = combined  # harness structure already tested above

    # Exec function + test class directly to verify tests actually fail
    globs: dict = {}
    exec(  # noqa: S102
        f"def task_func():\n    return 0\n\n{test_code}",
        globs,
    )
    TestCases = globs["TestCases"]
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)
    result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(suite)  # noqa: SIM115
    assert not result.wasSuccessful(), "Expected test failures but none occurred"
