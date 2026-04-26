from __future__ import annotations

import json
import sqlite3
from csv import DictReader
from pathlib import Path

from mcode.bench.results import ResultsDB, export_csv, merge_shard_dbs


def test_start_run_supports_legacy_runs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-results.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE runs (
              id INTEGER PRIMARY KEY,
              timestamp TEXT NOT NULL,
              benchmark TEXT NOT NULL,
              model_id TEXT NOT NULL,
              loop_budget INTEGER NOT NULL,
              timeout_s INTEGER NOT NULL,
              retrieval INTEGER NOT NULL,
              config_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 3,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        row = rdb.conn.execute(
            "SELECT backend_name, retrieval FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    assert row is not None
    assert row["backend_name"] == "ollama"
    assert row["retrieval"] == 0


def test_results_db_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    rdb = ResultsDB(db_path)
    run_id = rdb.start_run(
        "swebench-live",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "loop_budget": 3,
            "timeout_s": 60,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    rdb.save_task_result(
        run_id,
        {
            "task_id": "HumanEval/0",
            "passed": True,
            "attempts_used": 1,
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
        "swebench-live",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "loop_budget": 3,
            "timeout_s": 60,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    rdb.save_task_result(
        run_id_2,
        {
            "task_id": "HumanEval/1",
            "passed": False,
            "attempts_used": 1,
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
        benchmark="swebench-live",
        model_id="test-model",
        backend_name="ollama",
        timeout_s=60,
        group_by=(),
    )
    assert len(per_run) == 2
    assert {r["run_id"] for r in per_run} == {run_id, run_id_2}
    assert all(r["loop_budget"] == 3 for r in per_run)
    cfg = json.loads(per_run[0]["config_json"])
    assert cfg["loop_budget"] == 3

    grouped = rdb.pass_rates_grouped(
        benchmark="swebench-live",
        model_id="test-model",
        backend_name="ollama",
        timeout_s=60,
        group_by=("loop_budget",),
    )
    assert len(grouped) == 1
    assert grouped[0]["loop_budget"] == 3
    assert grouped[0]["total"] == 2
    assert grouped[0]["passed"] == 1


def test_run_metrics_grouped_includes_time_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "attempts_used": 1,
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
                "attempts_used": 1,
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
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="ollama",
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
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id_1,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "attempts_used": 1,
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
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id_2,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "attempts_used": 1,
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
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="ollama",
            timeout_s=60,
            group_by=("loop_budget",),
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
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": False,
                "attempts_used": 1,
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
                "attempts_used": 1,
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
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="ollama",
            timeout_s=60,
            group_by=("loop_budget",),
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
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb_a.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "attempts_used": 1,
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
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb_b.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "attempts_used": 1,
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
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="ollama",
            timeout_s=60,
            group_by=("loop_budget",),
        )
        assert len(rows) == 1
        assert rows[0]["runs"] == 2
        assert rows[0]["total"] == 2


def test_merge_shard_dbs_normalizes_shard_config(tmp_path: Path) -> None:
    shard_a = tmp_path / "results-shard-0.db"
    shard_b = tmp_path / "results-shard-1.db"
    merged = tmp_path / "merged.db"

    for shard_index, shard_db in enumerate((shard_a, shard_b)):
        with ResultsDB(shard_db) as rdb:
            run_id = rdb.start_run(
                "swebench-lite",
                {
                    "backend_name": "ollama",
                    "model_id": "test-model",
                    "loop_budget": 15,
                    "timeout_s": 300,
                    "task_shard_count": 2,
                    "task_shard_index": shard_index,
                    "planned_task_count": 1,
                    "cache_dir": str(tmp_path / "cache"),
                },
            )
            rdb.save_task_result(
                run_id,
                {
                    "task_id": f"task-{shard_index}",
                    "passed": shard_index == 0,
                    "attempts_used": 1,
                    "time_ms": 1000,
                    "exit_code": 0,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "code_sha256": f"sha-{shard_index}",
                },
            )

    report = merge_shard_dbs(out_path=merged, shard_paths=[shard_a, shard_b], force=True)

    with ResultsDB(merged) as rdb:
        config_json = rdb.conn.execute(
            "SELECT config_json FROM runs WHERE id = ?",
            (report["run_id"],),
        ).fetchone()[0]
    config = json.loads(config_json)

    assert config["task_shard_count"] is None
    assert config["task_shard_index"] is None
    assert config["planned_task_count"] == 2
    assert config["merged_shards"] == 2


def test_run_metrics_grouped_includes_scaffold_quality_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 3,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 1000,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "fail",
                "error": "Execution failed",
                "code_sha256": "abc",
                "terminal_reason": "wrong_patch_after_verification",
                "turns_to_first_edit": 2,
                "turns_to_first_verification": 3,
                "zero_edit": False,
                "zero_verification": False,
                "verification_succeeded": True,
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 500,
                "exit_code": None,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": None,
                "terminal_reason": "budget_exhausted",
                "turns_to_first_edit": None,
                "turns_to_first_verification": None,
                "zero_edit": True,
                "zero_verification": True,
                "verification_succeeded": False,
            },
        )

        rows = rdb.run_metrics_grouped(
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="ollama",
            timeout_s=60,
            group_by=(),
        )

    assert len(rows) == 1
    r = rows[0]
    assert r["zero_edit"] == 1
    assert r["zero_edit_rate"] == 0.5
    assert r["zero_verification"] == 1
    assert r["zero_verification_rate"] == 0.5
    assert r["verification_succeeded"] == 1
    assert r["verification_success_rate"] == 0.5
    assert r["turns_to_first_edit_avg"] == 2.0
    assert r["turns_to_first_verification_avg"] == 3.0
    assert r["budget_exhausted"] == 1
    assert r["wrong_patch_after_verification"] == 1
    assert r["submitted"] == 0


def test_run_metrics_grouped_includes_usage_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 2,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 1000,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
                "prompt_snapshot": "prompt one",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "provider": "openai",
                "response_model": "test-model",
                "submission_json": '{"summary":"done"}',
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/1",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 1200,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": "Execution failed",
                "code_sha256": "def",
                "prompt_snapshot": "prompt two",
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
                "provider": "openai",
                "response_model": "test-model",
                "submission_json": '{"summary":"failed"}',
            },
        )

        rows = rdb.run_metrics_grouped(
            benchmark="swebench-live",
            model_id="test-model",
            backend_name="openai",
            timeout_s=60,
            group_by=(),
        )

    assert len(rows) == 1
    r = rows[0]
    assert r["prompts_recorded"] == 2
    assert r["prompt_record_rate"] == 1.0
    assert r["usage_recorded"] == 2
    assert r["usage_record_rate"] == 1.0
    assert r["prompt_tokens_total"] == 30
    assert r["completion_tokens_total"] == 13
    assert r["total_tokens_total"] == 43
    assert r["prompt_tokens_avg"] == 15.0
    assert r["completion_tokens_avg"] == 6.5
    assert r["total_tokens_avg"] == 21.5


def test_export_csv_includes_generation_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "HumanEval/0",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 500,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "ok",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
                "prompt_snapshot": "prompt body",
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "provider": "openai",
                "response_model": "test-model",
                "submission_json": '{"summary":"ok"}',
            },
        )

    out = export_csv(inputs=[db_path], out_dir=tmp_path / "csv", include_logs=True)
    with Path(out["task_results_csv"]).open(newline="", encoding="utf-8") as f:
        rows = list(DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "openai"
    assert row["response_model"] == "test-model"
    assert row["prompt_tokens"] == "11"
    assert row["completion_tokens"] == "7"
    assert row["total_tokens"] == "18"
    assert row["submission_json"] == '{"summary":"ok"}'
    assert row["prompt_snapshot"] == "prompt body"



def test_results_db_persists_merges_and_exports_diagnostic_events(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    merged_db = tmp_path / "merged.db"

    with ResultsDB(db_a) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "diagnostic_traces": True,
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "task-1",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 100,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": "failed",
                "code_sha256": "abc",
                "diagnostic_events": [
                    {
                        "turn": 1,
                        "event_type": "turn_start",
                        "payload": {"turn": 1},
                    },
                    {
                        "turn": None,
                        "event_type": "terminal",
                        "payload": {"official_eval_passed": False},
                    },
                ],
            },
        )

    with ResultsDB(merged_db) as merged:
        merged.merge_from([db_a])
        row = merged.conn.execute(
            """
            SELECT de.run_id, de.task_id, de.event_index, de.event_type, de.payload_json
            FROM diagnostic_events de
            JOIN runs r ON r.id = de.run_id
            ORDER BY de.event_index
            """
        ).fetchone()
        assert row is not None
        assert int(row["run_id"]) == 1
        assert row["task_id"] == "task-1"
        assert row["event_type"] == "turn_start"
        assert json.loads(row["payload_json"]) == {"turn": 1}

    out = export_csv(inputs=[merged_db], out_dir=tmp_path / "csv")
    assert out["diagnostic_events"] == 2
    with Path(out["diagnostic_events_csv"]).open(newline="", encoding="utf-8") as f:
        rows = list(DictReader(f))
    assert [row["event_type"] for row in rows] == ["turn_start", "terminal"]


def test_merge_shard_dbs_preserves_diagnostic_events(tmp_path: Path) -> None:
    shard = tmp_path / "results-shard-0.db"
    merged = tmp_path / "merged.db"

    with ResultsDB(shard) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "task_shard_count": 1,
                "task_shard_index": 0,
                "diagnostic_traces": True,
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "task-1",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 100,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
                "diagnostic_events": [
                    {
                        "turn": 1,
                        "event_type": "generation",
                        "payload": {"tool_call_count": 0},
                    }
                ],
            },
        )

    merge_shard_dbs(out_path=merged, shard_paths=[shard], force=True)

    with ResultsDB(merged) as rdb:
        count = rdb.conn.execute("SELECT COUNT(*) FROM diagnostic_events").fetchone()[0]
        payload = rdb.conn.execute(
            "SELECT payload_json FROM diagnostic_events"
        ).fetchone()[0]
    assert count == 1
    assert json.loads(payload) == {"tool_call_count": 0}