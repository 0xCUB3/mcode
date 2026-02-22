from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mcode.bench.runner import _combine_for_eval
from mcode.bench.tasks import Task

_LCB_FIXTURE = [
    {
        "question_id": 101,
        "question_content": "Given two integers, print their sum.",
        "starter_code": "",
        "test_cases": json.dumps({"inputs": ["1 2\n", "3 4\n"], "outputs": ["3\n", "7\n"]}),
        "contest_date": "2024-03-15",
        "difficulty": "easy",
        "question_title": "Sum Two",
    },
    {
        "question_id": 102,
        "question_content": "Print hello world.",
        "starter_code": "# your code here\n",
        "test_cases": json.dumps({"inputs": ["\n"], "outputs": ["hello world\n"]}),
        "contest_date": "2024-07-01",
        "difficulty": "easy",
        "question_title": "Hello World",
    },
    {
        "question_id": 103,
        "question_content": "Reverse a string.",
        "starter_code": "",
        "test_cases": json.dumps({"inputs": ["abc\n"], "outputs": ["cba\n"]}),
        "contest_date": "2024-05-20",
        "difficulty": "medium",
        "question_title": "Reverse String",
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "livecodebench.json"
    p.write_text(json.dumps(_LCB_FIXTURE))
    return p


def test_load_livecodebench_produces_tasks(tmp_path: Path) -> None:
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.livecodebench import load_livecodebench

        tasks = load_livecodebench(None)

    assert len(tasks) == 3
    task0 = next(t for t in tasks if t.task_id == "101")
    assert task0.benchmark == "livecodebench"
    assert task0.task_id == "101"
    assert "Given two integers, print their sum." in task0.prompt
    assert task0.entry_point is None
    parsed = json.loads(task0.test_code)
    assert "inputs" in parsed
    assert "outputs" in parsed
    assert task0.metadata["contest_date"] == "2024-03-15"
    assert task0.metadata["difficulty"] == "easy"
    assert task0.metadata["question_title"] == "Sum Two"
    assert task0.metadata["source"] == "livecodebench"


def test_load_livecodebench_cutoff_filters(tmp_path: Path) -> None:
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.livecodebench import load_livecodebench

        tasks = load_livecodebench(None, cutoff="2024-06-01")

    # Only tasks with contest_date < "2024-06-01": 2024-03-15 and 2024-05-20
    assert len(tasks) == 2
    ids = {t.task_id for t in tasks}
    assert "101" in ids  # 2024-03-15
    assert "103" in ids  # 2024-05-20
    assert "102" not in ids  # 2024-07-01 is after cutoff


def test_load_livecodebench_cutoff_none_returns_all(tmp_path: Path) -> None:
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.livecodebench import load_livecodebench

        tasks = load_livecodebench(None, cutoff=None)

    assert len(tasks) == 3


def test_load_livecodebench_limit(tmp_path: Path) -> None:
    """load_livecodebench returns all tasks; _limit in load_benchmark handles the cap."""
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.livecodebench import load_livecodebench

        tasks = load_livecodebench(None)

    # _limit slicing is applied at the load_benchmark level; here we verify the full set
    assert len(tasks) == 3
    assert len(tasks[:1]) == 1


def test_load_livecodebench_limit_via_load_benchmark(tmp_path: Path) -> None:
    """load_benchmark respects limit for livecodebench."""
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.tasks import load_benchmark

        tasks = load_benchmark("livecodebench", Path("/tmp"), limit=2)

    assert len(tasks) == 2


def test_load_livecodebench_starter_code_in_prompt(tmp_path: Path) -> None:
    with patch("mcode.bench.livecodebench._DATA_FILE", _write_fixture(tmp_path)):
        from mcode.bench.livecodebench import load_livecodebench

        tasks = load_livecodebench(None)

    task_with_starter = next(t for t in tasks if t.task_id == "102")
    # starter_code "# your code here\n" should appear in prompt
    assert "# your code here" in task_with_starter.prompt


def test_combine_for_eval_livecodebench() -> None:
    test_data = json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})
    task = Task(
        benchmark="livecodebench",
        task_id="101",
        prompt="Given two integers, print their sum.",
        entry_point=None,
        test_code=test_data,
        metadata={"source": "livecodebench", "contest_date": "2024-03-15"},
    )
    result = _combine_for_eval(task, "a, b = map(int, input().split()); print(a + b)")

    assert "json" in result
    assert "sys.stdin" in result
    assert "io.StringIO" in result
    assert "a, b = map(int, input().split())" in result
    assert "_failed" in result
    assert "exec(compile(" in result


def test_combine_for_eval_livecodebench_harness_runs() -> None:
    """End-to-end: exec the harness with a simple passing case."""
    test_data = json.dumps({"inputs": ["\n"], "outputs": ["hello\n"]})
    task = Task(
        benchmark="livecodebench",
        task_id="200",
        prompt="Print hello.",
        entry_point=None,
        test_code=test_data,
        metadata={"source": "livecodebench", "contest_date": "2024-01-01"},
    )
    code = "print('hello')"
    combined = _combine_for_eval(task, code)

    # Should execute without raising
    exec(combined, {})  # noqa: S102


def test_combine_for_eval_livecodebench_harness_fails_on_wrong_output() -> None:
    """Harness raises SystemExit when output does not match."""
    test_data = json.dumps({"inputs": ["\n"], "outputs": ["goodbye\n"]})
    task = Task(
        benchmark="livecodebench",
        task_id="201",
        prompt="Print goodbye.",
        entry_point=None,
        test_code=test_data,
        metadata={"source": "livecodebench", "contest_date": "2024-01-01"},
    )
    code = "print('hello')"
    combined = _combine_for_eval(task, code)

    with pytest.raises(SystemExit):
        exec(combined, {})  # noqa: S102
