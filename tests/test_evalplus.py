from __future__ import annotations

import importlib
import types
from unittest.mock import patch

import pytest

from mcode.bench.runner import _combine_for_eval
from mcode.bench.tasks import Task

_HUMANEVAL_FIXTURE = {
    "HumanEval/0": {
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    },
    "HumanEval/1": {
        "prompt": "def mul(a, b):\n",
        "entry_point": "mul",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 6\n",
    },
}

_MBPP_FIXTURE = {
    "Mbpp/1": {
        "prompt": "Write a function to add two numbers.",
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        "test_setup_code": "",
        "test": "...",
    },
    "Mbpp/2": {
        "prompt": "Write a function to multiply.",
        "test_list": ["assert mul(2, 3) == 6"],
        "test_setup_code": "import math",
        "test": "...",
    },
}


def _make_evalplus_modules(humaneval_fixture=None, mbpp_fixture=None):
    evalplus_mod = types.ModuleType("evalplus")
    evalplus_data_mod = types.ModuleType("evalplus.data")
    if humaneval_fixture is not None:
        evalplus_data_mod.get_human_eval_plus = lambda: humaneval_fixture
    if mbpp_fixture is not None:
        evalplus_data_mod.get_mbpp_plus = lambda: mbpp_fixture
    evalplus_mod.data = evalplus_data_mod
    return {"evalplus": evalplus_mod, "evalplus.data": evalplus_data_mod}


def test_load_humaneval_plus_produces_tasks() -> None:
    fake_mods = _make_evalplus_modules(humaneval_fixture=_HUMANEVAL_FIXTURE)
    with patch.dict("sys.modules", fake_mods):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        tasks = ep_mod.load_humaneval_plus(None)

    assert len(tasks) == 2
    task0 = next(t for t in tasks if t.task_id == "HumanEval/0")
    assert task0.benchmark == "humaneval+"
    assert task0.prompt == "def add(a, b):\n"
    assert task0.entry_point == "add"
    assert task0.test_code == "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    assert task0.metadata == {"source": "evalplus/humaneval+"}

    task1 = next(t for t in tasks if t.task_id == "HumanEval/1")
    assert task1.benchmark == "humaneval+"
    assert task1.entry_point == "mul"


def test_load_humaneval_plus_limit() -> None:
    fake_mods = _make_evalplus_modules(humaneval_fixture=_HUMANEVAL_FIXTURE)
    with patch.dict("sys.modules", fake_mods):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        tasks = ep_mod.load_humaneval_plus(None, limit=1)

    assert len(tasks) == 1


def test_load_humaneval_plus_missing_evalplus() -> None:
    with patch.dict("sys.modules", {"evalplus": None, "evalplus.data": None}):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        with pytest.raises(RuntimeError, match="evalplus"):
            ep_mod.load_humaneval_plus(None)


def test_load_mbpp_plus_produces_tasks() -> None:
    fake_mods = _make_evalplus_modules(mbpp_fixture=_MBPP_FIXTURE)
    with patch.dict("sys.modules", fake_mods):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        tasks = ep_mod.load_mbpp_plus(None)

    assert len(tasks) == 2
    task1 = next(t for t in tasks if t.task_id == "Mbpp/1")
    assert task1.benchmark == "mbpp+"
    assert task1.entry_point is None
    assert "Write a function to add two numbers." in task1.prompt
    assert "assert add(1, 2) == 3" in task1.prompt
    assert task1.test_code == "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    assert task1.metadata == {"source": "evalplus/mbpp+"}

    task2 = next(t for t in tasks if t.task_id == "Mbpp/2")
    assert task2.benchmark == "mbpp+"
    assert task2.entry_point is None
    assert "import math" in task2.test_code
    assert "assert mul(2, 3) == 6" in task2.test_code


def test_load_mbpp_plus_limit() -> None:
    fake_mods = _make_evalplus_modules(mbpp_fixture=_MBPP_FIXTURE)
    with patch.dict("sys.modules", fake_mods):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        tasks = ep_mod.load_mbpp_plus(None, limit=1)

    assert len(tasks) == 1


def test_load_mbpp_plus_missing_evalplus() -> None:
    with patch.dict("sys.modules", {"evalplus": None, "evalplus.data": None}):
        import mcode.bench.evalplus as ep_mod

        importlib.reload(ep_mod)
        with pytest.raises(RuntimeError, match="evalplus"):
            ep_mod.load_mbpp_plus(None)


def test_combine_for_eval_humaneval_plus() -> None:
    task = Task(
        benchmark="humaneval+",
        task_id="HumanEval/0",
        prompt="def add(a, b):\n",
        entry_point="add",
        test_code="def check(candidate):\n    assert candidate(1,2)==3\n",
        metadata={},
    )
    result = _combine_for_eval(task, "def add(a,b): return a+b")
    assert "def add(a,b): return a+b" in result
    assert "def check(candidate):" in result
    assert "check(add)" in result
    assert "__mcode_main" in result


def test_combine_for_eval_mbpp_plus() -> None:
    task = Task(
        benchmark="mbpp+",
        task_id="Mbpp/1",
        prompt="Write a function to add two numbers.",
        entry_point=None,
        test_code="assert add(1,2)==3\n",
        metadata={},
    )
    result = _combine_for_eval(task, "def add(a,b): return a+b")
    assert "def add(a,b): return a+b" in result
    assert "# --- mbpp tests ---" in result
    assert "assert add(1,2)==3" in result
