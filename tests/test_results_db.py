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


def test_run_metrics_grouped_includes_time_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "humaneval",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "samples": 1,
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
                "time_ms": 1000,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 3000,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "fail",
                "error": "Execution failed",
                "code_sha256": "def",
            },
        )

        rows = rdb.run_metrics_grouped(
            benchmark="humaneval",
            model_id="test-model",
            backend_name="ollama",
            max_debug_iterations=0,
            timeout_s=60,
            group_by=(),
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["total"] == 2
        assert r["passed"] == 1
        assert r["timed_out"] == 0
        assert r["timeout_rate"] == 0.0
        assert r["time_ms_total"] == 4000
        assert r["time_ms_avg"] == 2000.0
        assert r["time_ms_p50"] == 2000.0
        assert r["time_ms_p95"] == 2900.0
        assert r["sec_per_solve"] == 4.0
        assert r["solves_per_hour"] == 900.0


def test_run_metrics_grouped_aggregates_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id_1 = rdb.start_run(
            "humaneval",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "samples": 1,
                "retrieval": False,
                "max_debug_iterations": 0,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id_1,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 1000,
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
                "samples": 1,
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
                "time_ms": 3000,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "fail",
                "error": "Execution failed",
                "code_sha256": "def",
            },
        )

        rows = rdb.run_metrics_grouped(
            benchmark="humaneval",
            model_id="test-model",
            backend_name="ollama",
            max_debug_iterations=0,
            timeout_s=60,
            group_by=("samples",),
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["runs"] == 2
        assert r["total"] == 2
        assert r["passed"] == 1
        assert r["timed_out"] == 0
        assert r["timeout_rate"] == 0.0
        assert r["time_ms_total"] == 4000
        assert r["time_ms_p50"] == 2000.0
        assert r["time_ms_p95"] == 2900.0


def test_run_metrics_grouped_counts_timeouts(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "humaneval",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "samples": 1,
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
                "passed": False,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 500,
                "exit_code": 1,
                "timed_out": True,
                "stdout": "",
                "stderr": "timeout",
                "error": "Execution timed out",
                "code_sha256": "abc",
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": True,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 400,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "def",
            },
        )

        rows = rdb.run_metrics_grouped(
            benchmark="humaneval",
            model_id="test-model",
            backend_name="ollama",
            max_debug_iterations=0,
            timeout_s=60,
            group_by=("samples",),
        )

        assert len(rows) == 1
        r = rows[0]
        assert r["total"] == 2
        assert r["passed"] == 1
        assert r["timed_out"] == 1
        assert r["timeout_rate"] == 0.5


def test_merge_from_combines_dbs(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    merged_db = tmp_path / "merged.db"

    with ResultsDB(db_a) as rdb_a:
        run_id = rdb_a.start_run(
            "humaneval",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "samples": 1,
                "retrieval": False,
                "max_debug_iterations": 0,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb_a.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 1000,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
            },
        )

    with ResultsDB(db_b) as rdb_b:
        run_id = rdb_b.start_run(
            "humaneval",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "samples": 1,
                "retrieval": False,
                "max_debug_iterations": 0,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb_b.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "samples_generated": 1,
                "debug_iterations_used": 0,
                "time_ms": 3000,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "fail",
                "error": "Execution failed",
                "code_sha256": "def",
            },
        )

    with ResultsDB(merged_db) as rdb:
        rdb.merge_from([db_a, db_b])
        rows = rdb.run_metrics_grouped(
            benchmark="humaneval",
            model_id="test-model",
            backend_name="ollama",
            max_debug_iterations=0,
            timeout_s=60,
            group_by=("samples",),
        )
        assert len(rows) == 1
        assert rows[0]["runs"] == 2
        assert rows[0]["total"] == 2
