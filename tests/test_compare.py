from __future__ import annotations

from pathlib import Path

from mcode.bench.compare import compare_runs
from mcode.bench.results import ResultsDB


def _write_compare_db(
    db_path: Path,
    *,
    suite_entry_name: str,
    results: dict[str, bool],
) -> None:
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": suite_entry_name,
            },
        )
        for task_id, passed in results.items():
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
                    "code_sha256": task_id,
                },
            )


def test_compare_runs_filters_suite_entry(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    _write_compare_db(
        baseline_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": False},
    )
    _write_compare_db(
        baseline_dir / "b.db",
        suite_entry_name="polyglot-go",
        results={"go/alphametics": True},
    )
    _write_compare_db(
        candidate_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": True},
    )
    _write_compare_db(
        candidate_dir / "b.db",
        suite_entry_name="polyglot-go",
        results={"go/alphametics": False},
    )

    report = compare_runs(
        str(baseline_dir),
        str(candidate_dir),
        suite_name="tiny-polyglot-suite",
        suite_entry_name="polyglot-python",
    )

    assert report["gained"] == ["python/affine-cipher"]
    assert report["lost"] == []
    assert report["baseline_total"] == 1
    assert report["candidate_total"] == 1
