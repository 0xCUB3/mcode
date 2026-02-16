from __future__ import annotations

from pathlib import Path

from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.bench.tasks import Task


def test_run_task_llm_exception_is_recorded(tmp_path: Path) -> None:
    db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="dummy", sandbox="process"),
        results_db=db,
    )

    # MBPP task shape; it won't actually execute because we force LLM failure.
    task = Task(
        benchmark="mbpp",
        task_id="mbpp-1",
        prompt="Return 1.",
        entry_point=None,
        test_code="assert True\n",
        metadata={},
    )

    def boom(*args, **kwargs):
        raise RuntimeError("mellea blew up")

    # Avoid needing a real Mellea session in unit tests.
    runner.llm.generate_code = boom  # type: ignore[method-assign]

    result = runner.run_task(task)
    assert result["passed"] is False
    assert result["attempts_used"] == 0
    assert "mellea blew up" in (result.get("error") or "")

