from __future__ import annotations

from pathlib import Path

from mcode.cli import _parse_task_ids


def test_parse_task_ids_accepts_long_inline_comma_list() -> None:
    task_ids = [f"sympy__sympy-{i}" for i in range(1000, 1045)]

    assert _parse_task_ids(",".join(task_ids)) == task_ids


def test_parse_task_ids_accepts_text_file(tmp_path: Path) -> None:
    task_ids = ["sympy__sympy-1", "sympy__sympy-2", "sympy__sympy-3"]
    task_file = tmp_path / "task_ids.txt"
    task_file.write_text("\n".join(task_ids) + "\n")

    assert _parse_task_ids(str(task_file)) == task_ids
