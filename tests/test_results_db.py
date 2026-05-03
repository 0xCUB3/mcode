from __future__ import annotations

import json
import sqlite3
from csv import DictReader
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from mcode.bench.artifacts import (
    SCHEMA_VERSION,
    TaskArtifactManifest,
    TaskArtifactStore,
    VerificationEvidence,
    digest_json,
    digest_text,
    iso_utc_now,
    make_task_digest,
    read_task_manifest,
 )
from mcode.bench.results import ResultsDB, export_csv, merge_shard_dbs


def _sample_task_manifest(
    tmp_path: Path,
    *,
    benchmark: str = "swebench-lite",
    task_id: str = "task-1",
) -> tuple[TaskArtifactManifest, Path]:
    artifact_dir = tmp_path / "artifacts"
    store = TaskArtifactStore.from_task(
        artifact_dir=artifact_dir,
        benchmark=benchmark,
        task_id=task_id,
    )
    repo_id = "example/repo"
    task_metadata = {"base_commit": "abc123", "problem_statement": "Fix the bug"}
    task_ref = store.build_task_ref(
        repo_id=repo_id,
        task_digest=make_task_digest(
            benchmark=benchmark,
            task_id=task_id,
            repo_id=repo_id,
            metadata=task_metadata,
        ),
        metadata=task_metadata,
    )
    evidence = VerificationEvidence(
        verifier_name="run_tests",
        command_label="default",
        command_digest=digest_text("pytest -q"),
        status="PASSED",
        counted_as_verification=True,
        output_digest=digest_text("1 passed"),
        output_preview_path=None,
        metadata={"output_preview": "$ pytest -q\nPASSED\n1 passed"},
    )
    candidate = store.write_candidate(
        candidate_index=0,
        patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
        terminal_reason="submitted",
        selected=True,
        submission_json='{"summary":"done"}',
        generation_time_ms=250,
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        provider="openai",
        response_model="test-model",
        validation_passed_count=1,
        validation_failed_count=0,
        zero_edit=False,
        zero_verification=False,
        verification_succeeded=True,
        trace_events=[{"turn": 1, "event_type": "generation", "payload": {"ok": True}}],
        verification_evidence=[evidence],
        failure_counters={"blocked_finalizer_count": 0},
        metadata={"candidate_label": "baseline"},
    )
    evaluation = store.write_evaluation(
        source_candidate_index=0,
        evaluator_name="official",
        passed=True,
        timed_out=False,
        exit_code=0,
        report={"resolved": True},
        stdout="tests passed",
        stderr=None,
        error_class=None,
        runtime_ms=333,
        metadata={"adapter": benchmark},
    )
    manifest = TaskArtifactManifest(
        schema_version=SCHEMA_VERSION,
        phase="run",
        generated_at=iso_utc_now(),
        run_config_digest=digest_json({"benchmark": benchmark, "phase": "run"}),
        code_sha="deadbeef",
        model_id="test-model",
        backend_name="openai",
        task=task_ref,
        candidates=(candidate,),
        evaluations=(evaluation,),
        metadata={"suite": "smoke"},
    )
    manifest_path = store.write_manifest(manifest)
    return manifest, manifest_path


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


def test_resume_helpers_find_latest_exact_config_and_task_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    config = {
        "backend_name": "ollama",
        "model_id": "test-model",
        "loop_budget": 3,
        "timeout_s": 60,
        "cache_dir": tmp_path / "cache",
    }
    with ResultsDB(db_path) as rdb:
        first_run = rdb.start_run("swebench-lite", config)
        second_run = rdb.start_run("swebench-lite", dict(reversed(config.items())))
        other_run = rdb.start_run("swebench-live", config)

        rdb.save_task_result(
            second_run,
            {
                "task_id": "task-ok",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 10,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
                "terminal_reason": "submitted",
            },
        )
        rdb.save_task_result(
            second_run,
            {
                "task_id": "task-infra",
                "passed": False,
                "attempts_used": 2,
                "time_ms": 20,
                "exit_code": None,
                "timed_out": False,
                "stdout": None,
                "stderr": "trace",
                "error": "DockerUnavailableError: podman socket closed",
                "code_sha256": None,
                "terminal_reason": "infra_failure",
            },
        )

        assert rdb.find_latest_run_by_config("swebench-lite", config) == second_run
        assert rdb.find_latest_run_by_config("swebench-live", config) == other_run
        assert (
            rdb.find_latest_run_by_config(
                "swebench-lite",
                {**config, "timeout_s": 120},
            )
            is None
        )

        rows = rdb.task_terminal_rows(second_run)
        assert rows["task-ok"]["passed"] is True
        assert rows["task-ok"]["terminal_reason"] == "submitted"
        assert rows["task-infra"]["passed"] is False
        assert rows["task-infra"]["terminal_reason"] == "infra_failure"
        assert rows["task-infra"]["error"] == "DockerUnavailableError: podman socket closed"
        assert rdb.run_summary(second_run).total == 2
        assert rdb.run_summary(second_run).passed == 1
        assert rdb.run_summary(first_run).total == 0


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


def test_results_db_derives_diagnostic_counter_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-live",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 4,
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
                "time_ms": 250,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": "failed",
                "code_sha256": "abc",
                "terminal_reason": "wrong_patch_after_verification",
                "turns_to_first_edit": 2,
                "turns_to_first_verification": 5,
                "zero_edit": False,
                "zero_verification": False,
                "verification_succeeded": True,
                "diagnostic_events": [
                    {
                        "turn": 1,
                        "event_type": "tool_arg_compat",
                        "payload": {"raw_arg_call_count": 1, "recoverable_call_count": 1},
                    },
                    {
                        "turn": 1,
                        "event_type": "tool_call_filter",
                        "payload": {"invalid_call_count": 2, "blocked_finalizer_count": 1},
                    },
                    {
                        "turn": 2,
                        "event_type": "edit_result",
                        "payload": {"status": "APPLIED"},
                    },
                    {
                        "turn": 3,
                        "event_type": "read_search_target",
                        "payload": {"tool_name": "search_code", "query": "needle"},
                    },
                    {
                        "turn": 4,
                        "event_type": "run_tests",
                        "payload": {"repeated_failed_run_suppressed": True},
                    },
                    {
                        "turn": 5,
                        "event_type": "final_answer",
                        "payload": {"action": "autofilled"},
                    },
                ],
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
    row = rows[0]
    assert row["malformed_tool_call_recoveries"] == 2
    assert row["invalid_tool_call_count"] == 2
    assert row["blocked_finalizer_count"] == 1
    assert row["repeated_failed_run_test_count"] == 1
    assert row["post_edit_exploration_count"] == 1
    assert row["turns_after_first_edit_before_first_verification_avg"] == 3.0

    out = export_csv(inputs=[db_path], out_dir=tmp_path / "csv")
    with Path(out["task_results_csv"]).open(newline="", encoding="utf-8") as f:
        csv_rows = list(DictReader(f))

    assert csv_rows[0]["malformed_tool_call_recoveries"] == "2"
    assert csv_rows[0]["turns_after_first_edit_before_first_verification"] == "3"


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
        payload = rdb.conn.execute("SELECT payload_json FROM diagnostic_events").fetchone()[0]
    assert count == 1
    assert json.loads(payload) == {"tool_call_count": 0}


def test_results_db_roundtrip_persists_task_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    manifest, manifest_path = _sample_task_manifest(tmp_path)
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 3,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)
        rows = rdb.task_artifact_rows(run_id)
        assert rows[manifest.task.task_id]["phase"] == "run"
        assert rows[manifest.task.task_id]["candidate_count"] == 1
        assert rows[manifest.task.task_id]["evaluation_count"] == 1
        candidate_row = rdb.conn.execute(
            "SELECT patch_sha256, trace_path FROM artifact_candidates "
            "WHERE run_id = ? AND task_id = ?",
            (run_id, manifest.task.task_id),
        ).fetchone()
        evidence_row = rdb.conn.execute(
            "SELECT verifier_name, output_preview_path FROM artifact_verification_evidence "
            "WHERE run_id = ? AND task_id = ?",
            (run_id, manifest.task.task_id),
        ).fetchone()
        evaluation_row = rdb.conn.execute(
            "SELECT evaluator_name, report_path FROM artifact_evaluations "
            "WHERE run_id = ? AND task_id = ?",
            (run_id, manifest.task.task_id),
        ).fetchone()
    assert candidate_row is not None
    assert candidate_row["patch_sha256"] == manifest.candidates[0].patch_stats.sha256
    assert candidate_row["trace_path"] == manifest.candidates[0].trace_path
    assert evidence_row is not None
    assert evidence_row["verifier_name"] == "run_tests"
    assert evidence_row["output_preview_path"] == "candidate-0/verification-0.txt"
    assert evaluation_row is not None
    assert evaluation_row["evaluator_name"] == "official"
    assert evaluation_row["report_path"] == "candidate-0/evaluation-report.json"
    assert read_task_manifest(manifest_path) == manifest


def test_run_metrics_grouped_surfaces_generation_only_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    manifest, manifest_path = _sample_task_manifest(tmp_path)
    generate_manifest = replace(manifest, phase="generate", evaluations=())
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 15,
                "timeout_s": 300,
                "cache_dir": str(tmp_path / "cache"),
                "phase": "generate",
            },
        )
        rdb.save_task_artifact_manifest(
            run_id,
            generate_manifest,
            manifest_path=manifest_path,
        )

        rows = rdb.run_metrics_grouped(
            benchmark="swebench-lite",
            model_id="test-model",
            backend_name="openai",
            timeout_s=300,
            group_by=(),
        )
        grouped_rows = rdb.run_metrics_grouped(
            benchmark="swebench-lite",
            model_id="test-model",
            backend_name="openai",
            timeout_s=300,
            group_by=("loop_budget",),
        )
        task_result_count = rdb.conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]

    assert task_result_count == 0
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["total"] == 0
    assert row["passed"] == 0
    assert row["pass_rate"] == 0.0
    assert row["artifact_generated_tasks"] == 1
    assert row["artifact_evaluated_tasks"] == 0
    assert row["artifact_candidate_count"] == 1
    assert row["artifact_selected_candidate_count"] == 1
    assert row["artifact_selected_patch_byte_count_total"] == (
        manifest.candidates[0].patch_stats.byte_count
    )
    assert row["artifact_selected_touched_file_count_total"] == len(
        manifest.candidates[0].patch_stats.touched_files
    )
    assert row["artifact_selected_added_lines_total"] == (
        manifest.candidates[0].patch_stats.added_lines
    )
    assert row["artifact_selected_deleted_lines_total"] == (
        manifest.candidates[0].patch_stats.deleted_lines
    )
    assert len(grouped_rows) == 1
    assert grouped_rows[0]["total"] == 0
    assert grouped_rows[0]["artifact_generated_tasks"] == 1
    assert grouped_rows[0]["artifact_candidate_count"] == 1


def test_export_csv_includes_generation_only_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    manifest, manifest_path = _sample_task_manifest(tmp_path)
    generate_manifest = replace(manifest, phase="generate", evaluations=())
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 15,
                "timeout_s": 300,
                "cache_dir": str(tmp_path / "cache"),
                "phase": "generate",
            },
        )
        rdb.save_task_artifact_manifest(
            run_id,
            generate_manifest,
            manifest_path=manifest_path,
        )

    out = export_csv(inputs=[db_path], out_dir=tmp_path / "csv")
    with Path(out["task_results_csv"]).open(newline="", encoding="utf-8") as f:
        task_rows = list(DictReader(f))
    with Path(out["artifact_tasks_csv"]).open(newline="", encoding="utf-8") as f:
        artifact_task_rows = list(DictReader(f))
    with Path(out["artifact_candidates_csv"]).open(newline="", encoding="utf-8") as f:
        artifact_candidate_rows = list(DictReader(f))

    assert out["task_results"] == 0
    assert out["artifact_tasks"] == 1
    assert out["artifact_candidates"] == 1
    assert task_rows == []
    assert artifact_task_rows[0]["run_id"] == str(run_id)
    assert artifact_task_rows[0]["task_id"] == manifest.task.task_id
    assert artifact_task_rows[0]["phase"] == "generate"
    assert artifact_task_rows[0]["manifest_path"] == str(manifest_path)
    assert artifact_task_rows[0]["task_digest"] == manifest.task.task_digest
    assert artifact_task_rows[0]["candidate_count"] == "1"
    assert artifact_task_rows[0]["evaluation_count"] == "0"
    assert artifact_candidate_rows[0]["task_id"] == manifest.task.task_id
    assert artifact_candidate_rows[0]["selected"] == "1"
    assert artifact_candidate_rows[0]["patch_path"] == manifest.candidates[0].patch_path
    assert artifact_candidate_rows[0]["patch_sha256"] == (
        manifest.candidates[0].patch_stats.sha256
    )
    assert artifact_candidate_rows[0]["trace_path"] == manifest.candidates[0].trace_path




def test_merge_from_preserves_artifact_rows(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    merged_db = tmp_path / "merged.db"
    manifest_a, manifest_path_a = _sample_task_manifest(tmp_path / "a", task_id="task-a")
    manifest_b, manifest_path_b = _sample_task_manifest(tmp_path / "b", task_id="task-b")
    with ResultsDB(db_a) as rdb_a:
        run_id = rdb_a.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache-a"),
            },
        )
        rdb_a.save_task_artifact_manifest(run_id, manifest_a, manifest_path=manifest_path_a)
    with ResultsDB(db_b) as rdb_b:
        run_id = rdb_b.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 1,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache-b"),
            },
        )
        rdb_b.save_task_artifact_manifest(run_id, manifest_b, manifest_path=manifest_path_b)
    with ResultsDB(merged_db) as rdb:
        rdb.merge_from([db_a, db_b])
        task_count = rdb.conn.execute("SELECT COUNT(*) FROM artifact_tasks").fetchone()[0]
        candidate_count = rdb.conn.execute("SELECT COUNT(*) FROM artifact_candidates").fetchone()[0]
        evaluation_count = rdb.conn.execute("SELECT COUNT(*) FROM artifact_evaluations").fetchone()[
            0
        ]
        evidence_count = rdb.conn.execute(
            "SELECT COUNT(*) FROM artifact_verification_evidence"
        ).fetchone()[0]
    assert task_count == 2
    assert candidate_count == 2
    assert evaluation_count == 2
    assert evidence_count == 2


def test_merge_shard_dbs_preserves_generation_only_artifacts(tmp_path: Path) -> None:
    shard_a = tmp_path / "results-shard-0.db"
    shard_b = tmp_path / "results-shard-1.db"
    merged = tmp_path / "merged.db"
    for shard_index, shard_db in enumerate((shard_a, shard_b)):
        manifest, manifest_path = _sample_task_manifest(
            tmp_path / f"artifacts-{shard_index}",
            task_id=f"task-{shard_index}",
        )
        with ResultsDB(shard_db) as rdb:
            run_id = rdb.start_run(
                "swebench-lite",
                {
                    "backend_name": "openai",
                    "model_id": "test-model",
                    "loop_budget": 15,
                    "timeout_s": 300,
                    "task_shard_count": 2,
                    "task_shard_index": shard_index,
                    "planned_task_count": 1,
                    "cache_dir": str(tmp_path / "cache"),
                    "phase": "generate",
                },
            )
            rdb.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)
    report = merge_shard_dbs(out_path=merged, shard_paths=[shard_a, shard_b], force=True)
    with ResultsDB(merged) as rdb:
        task_count = rdb.conn.execute("SELECT COUNT(*) FROM artifact_tasks").fetchone()[0]
        candidate_count = rdb.conn.execute("SELECT COUNT(*) FROM artifact_candidates").fetchone()[0]
        total = rdb.run_summary(report["run_id"]).total
    assert task_count == 2
    assert candidate_count == 2
    assert total == 0


def test_artifact_manifest_writes_non_json_metadata_with_default_str(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    store = TaskArtifactStore.from_task(
        artifact_dir=artifact_dir,
        benchmark="swebench-live",
        task_id="task-1",
    )
    task_ref = store.build_task_ref(
        repo_id="example/repo",
        task_digest="digest",
        metadata={"raw_instance": {"created_at": datetime(2026, 1, 1, tzinfo=UTC)}},
    )
    manifest = TaskArtifactManifest(
        schema_version=SCHEMA_VERSION,
        phase="generate",
        generated_at=iso_utc_now(),
        run_config_digest="digest",
        code_sha=None,
        model_id="test-model",
        backend_name="openai",
        task=task_ref,
    )
    path = store.write_manifest(manifest)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["task"]["metadata"]["raw_instance"]["created_at"] == "2026-01-01 00:00:00+00:00"

def test_run_metrics_grouped_separates_suite_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        for entry_name, task_id, passed in (
            ("polyglot-python", "python/affine-cipher", True),
            ("polyglot-go", "go/alphametics", False),
        ):
            run_id = rdb.start_run(
                "aider-polyglot",
                {
                    "backend_name": "openai",
                    "model_id": "test-model",
                    "loop_budget": 23,
                    "timeout_s": 300,
                    "suite_name": "tiny-polyglot-suite",
                    "suite_entry_name": entry_name,
                },
            )
            rdb.save_task_result(
                run_id,
                {
                    "task_id": task_id,
                    "passed": passed,
                    "attempts_used": 1,
                    "time_ms": 100,
                    "exit_code": 0 if passed else 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None if passed else "failed",
                    "code_sha256": task_id,
                },
            )
        rows = rdb.run_metrics_grouped(
            benchmark="aider-polyglot",
            model_id="test-model",
            backend_name="openai",
            timeout_s=300,
            group_by=("suite_name", "suite_entry_name", "loop_budget"),
        )
    assert len(rows) == 2
    by_entry = {row["suite_entry_name"]: row for row in rows}
    assert by_entry["polyglot-python"]["passed"] == 1
    assert by_entry["polyglot-go"]["passed"] == 0


def test_pass_rates_grouped_exposes_suite_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": "polyglot-python",
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "python/affine-cipher",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 100,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
            },
        )
        rows = rdb.pass_rates_grouped(
            benchmark="aider-polyglot",
            model_id="test-model",
            backend_name="openai",
            timeout_s=300,
            group_by=(),
        )
    assert rows[0]["suite_name"] == "tiny-polyglot-suite"
    assert rows[0]["suite_entry_name"] == "polyglot-python"


def test_export_csv_includes_suite_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": "polyglot-python",
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "python/affine-cipher",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 100,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "abc",
            },
        )

    out = export_csv(inputs=[db_path], out_dir=tmp_path / "csv")
    with Path(out["runs_csv"]).open(newline="", encoding="utf-8") as f:
        run_rows = list(DictReader(f))
    with Path(out["task_results_csv"]).open(newline="", encoding="utf-8") as f:
        task_rows = list(DictReader(f))

    assert run_rows[0]["suite_name"] == "tiny-polyglot-suite"
    assert run_rows[0]["suite_entry_name"] == "polyglot-python"
    assert task_rows[0]["suite_name"] == "tiny-polyglot-suite"
    assert task_rows[0]["suite_entry_name"] == "polyglot-python"


def test_export_csv_includes_suite_columns_on_artifact_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    manifest, manifest_path = _sample_task_manifest(tmp_path, benchmark="aider-polyglot")
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": "polyglot-python",
            },
        )
        rdb.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)

    out = export_csv(inputs=[db_path], out_dir=tmp_path / "csv")
    with Path(out["artifact_tasks_csv"]).open(newline="", encoding="utf-8") as f:
        artifact_rows = list(DictReader(f))

    assert artifact_rows[0]["suite_name"] == "tiny-polyglot-suite"
    assert artifact_rows[0]["suite_entry_name"] == "polyglot-python"