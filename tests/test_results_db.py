from __future__ import annotations

import json
from pathlib import Path

from mcode.bench.results import ResultsDB


def test_results_db_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    rdb = ResultsDB(db_path)
    run_id = rdb.start_run(
        "humaneval",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "samples": 3,
            "retrieval": False,
            "max_debug_iterations": 0,
            "timeout_s": 60,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    rdb.save_task_result(
        run_id,
        {
            "task_id": "HumanEval/0",
            "passed": True,
            "samples_generated": 1,
            "debug_iterations_used": 0,
            "time_ms": 10,
            "exit_code": 0,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "error": None,
            "code_sha256": "abc",
        },
    )
    run_id_2 = rdb.start_run(
        "humaneval",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "samples": 3,
            "retrieval": False,
            "max_debug_iterations": 0,
            "timeout_s": 60,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    rdb.save_task_result(
        run_id_2,
        {
            "task_id": "HumanEval/1",
            "passed": False,
            "samples_generated": 1,
            "debug_iterations_used": 0,
            "time_ms": 10,
            "exit_code": 1,
            "timed_out": False,
            "stdout": "",
            "stderr": "fail",
            "error": "Execution failed",
            "code_sha256": "def",
        },
    )

    per_run = rdb.pass_rates_grouped(
        benchmark="humaneval",
        model_id="test-model",
        backend_name="ollama",
        max_debug_iterations=0,
        timeout_s=60,
        group_by=(),
    )
    assert len(per_run) == 2
    assert {r["run_id"] for r in per_run} == {run_id, run_id_2}
    assert all(r["samples"] == 3 for r in per_run)
    cfg = json.loads(per_run[0]["config_json"])
    assert cfg["samples"] == 3

    grouped = rdb.pass_rates_grouped(
        benchmark="humaneval",
        model_id="test-model",
        backend_name="ollama",
        max_debug_iterations=0,
        timeout_s=60,
        group_by=("samples",),
    )
    assert len(grouped) == 1
    assert grouped[0]["samples"] == 3
    assert grouped[0]["total"] == 2
    assert grouped[0]["passed"] == 1
