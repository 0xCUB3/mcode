from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.results import ResultsDB
from mcode.cli import app


def _write_run(db_path: Path, *, task_id: str, passed: bool) -> None:
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 15,
                "timeout_s": 300,
                "cache_dir": str(db_path.parent / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": task_id,
                "passed": passed,
                "attempts_used": 1,
                "time_ms": 10,
                "exit_code": 0 if passed else 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None if passed else "failed",
                "code_sha256": "abc",
            },
        )


def test_compare_command_reports_diff(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    _write_run(baseline_dir / "baseline.db", task_id="task-1", passed=False)
    _write_run(candidate_dir / "candidate.db", task_id="task-1", passed=True)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compare",
            "--baseline-dir",
            str(baseline_dir),
            "--candidate-dir",
            str(candidate_dir),
        ],
        color=False,
    )

    assert res.exit_code == 0
    assert "Baseline: 0/1 passed" in res.stdout
    assert "Candidate: 1/1 passed" in res.stdout
    assert "Net change: +1" in res.stdout
