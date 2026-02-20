from __future__ import annotations

from mcode.bench.tasks import Task


def load_humaneval_plus(cache_dir, *, limit: int | None = None) -> list[Task]:
    _ = cache_dir
    try:
        from evalplus.data import get_human_eval_plus
    except Exception as e:
        raise RuntimeError(
            "EvalPlus benchmarks require the `evalplus` extra. "
            "Install with `uv pip install -e '.[evalplus]'`.\n"
            "If you installed `mcode` via `uv tool install ...`, install the extra there too:\n"
            "  `uv tool install -e '.[evalplus]'`"
        ) from e

    problems = get_human_eval_plus()
    tasks: list[Task] = []
    for key, problem in problems.items():
        if limit is not None and len(tasks) >= limit:
            break
        tasks.append(
            Task(
                benchmark="humaneval+",
                task_id=key,
                prompt=problem["prompt"],
                entry_point=problem["entry_point"],
                test_code=problem["test"],
                metadata={"source": "evalplus/humaneval+"},
            )
        )
    return tasks


def load_mbpp_plus(cache_dir, *, limit: int | None = None) -> list[Task]:
    _ = cache_dir
    try:
        from evalplus.data import get_mbpp_plus
    except Exception as e:
        raise RuntimeError(
            "EvalPlus benchmarks require the `evalplus` extra. "
            "Install with `uv pip install -e '.[evalplus]'`.\n"
            "If you installed `mcode` via `uv tool install ...`, install the extra there too:\n"
            "  `uv tool install -e '.[evalplus]'`"
        ) from e

    problems = get_mbpp_plus()
    tasks: list[Task] = []
    for key, problem in problems.items():
        if limit is not None and len(tasks) >= limit:
            break
        prompt = _prompt_from_problem(problem)
        test_code = _test_code_from_problem(problem)
        tasks.append(
            Task(
                benchmark="mbpp+",
                task_id=key,
                prompt=prompt,
                entry_point=None,
                test_code=test_code,
                metadata={"source": "evalplus/mbpp+"},
            )
        )
    return tasks


def _prompt_from_problem(problem: dict) -> str:
    tests = "\n".join(problem.get("test_list", []))
    setup = problem.get("test_setup_code", "").strip()
    setup_block = f"\n\n# Test setup\n{setup}\n" if setup else ""
    return (
        "Write Python code that solves the following problem.\n"
        "Return only Python code.\n\n"
        f"Problem:\n{problem['prompt'].strip()}\n"
        f"{setup_block}\n"
        f"# Tests\n{tests}\n"
    )


def _test_code_from_problem(problem: dict) -> str:
    setup = problem.get("test_setup_code", "").strip()
    tests = problem.get("test_list", [])
    lines: list[str] = []
    if setup:
        lines.append(setup)
    lines.extend(tests)
    return "\n".join(lines) + "\n"
