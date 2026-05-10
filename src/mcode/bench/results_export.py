from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


def export_csv(
    *,
    inputs: list[Path],
    out_dir: Path,
    prefix: str = "mcode",
    include_logs: bool = False,
) -> dict:
    """
    Export one or more results DBs to CSV.

    - Inputs may be .db files and/or directories (dirs: exports top-level *.db).
    - Shard DBs are excluded by default (they are intermediate artifacts).
    - Large text fields (stdout/stderr/error) are excluded by default; set include_logs=True to
      include.
    """

    db_paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            db_paths.extend(sorted(p.glob("*.db")))
        else:
            db_paths.append(p)
    db_paths = [p for p in db_paths if p.exists() and p.suffix == ".db" and "shard-" not in p.name]
    db_paths = sorted(set(db_paths))
    if not db_paths:
        raise FileNotFoundError("No .db files found (pass --input <db|dir> ...).")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / f"{prefix}.runs.csv"
    tasks_csv = out_dir / f"{prefix}.task_results.csv"
    diagnostic_csv = out_dir / f"{prefix}.diagnostic_events.csv"
    artifact_tasks_csv = out_dir / f"{prefix}.artifact_tasks.csv"
    artifact_candidates_csv = out_dir / f"{prefix}.artifact_candidates.csv"
    artifact_evaluations_csv = out_dir / f"{prefix}.artifact_evaluations.csv"
    artifact_evidence_csv = out_dir / f"{prefix}.artifact_verification_evidence.csv"

    runs_fields = [
        "source_db",
        "run_id",
        "timestamp",
        "benchmark",
        "backend_name",
        "model_id",
        "suite_name",
        "suite_entry_name",
        "loop_budget",
        "timeout_s",
        "total",
        "passed",
        "pass_rate",
        "config_json",
    ]

    task_fields = [
        "source_db",
        "run_id",
        "timestamp",
        "benchmark",
        "backend_name",
        "model_id",
        "suite_name",
        "suite_entry_name",
        "loop_budget",
        "timeout_s",
        "task_id",
        "passed",
        "attempts_used",
        "time_ms",
        "exit_code",
        "timed_out",
        "code_sha256",
        "provider",
        "response_model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "terminal_reason",
        "turns_to_first_edit",
        "turns_to_first_verification",
        "turns_after_first_edit_before_first_verification",
        "zero_edit",
        "zero_verification",
        "verification_succeeded",
        "malformed_tool_call_recoveries",
        "invalid_tool_call_count",
        "blocked_finalizer_count",
        "repeated_failed_run_test_count",
        "post_edit_exploration_count",
        "submission_json",
        "config_json",
    ]
    if include_logs:
        task_fields.extend(["stdout", "stderr", "error", "prompt_snapshot"])

    run_rows = 0
    task_rows = 0
    diagnostic_rows = 0
    artifact_task_rows = 0
    artifact_candidate_rows = 0
    artifact_evaluation_rows = 0
    artifact_evidence_rows = 0

    with (
        runs_csv.open("w", newline="", encoding="utf-8") as rf,
        tasks_csv.open("w", newline="", encoding="utf-8") as tf,
    ):
        runs_writer = csv.DictWriter(rf, fieldnames=runs_fields)
        tasks_writer = csv.DictWriter(tf, fieldnames=task_fields)
        runs_writer.writeheader()
        tasks_writer.writeheader()

        for db_path in db_paths:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                runs = conn.execute(
                    """
                    SELECT
                      r.*,
                      COUNT(tr.id) AS total,
                      SUM(tr.passed) AS passed
                    FROM runs r
                    LEFT JOIN task_results tr ON tr.run_id = r.id
                    GROUP BY r.id
                    ORDER BY r.timestamp ASC
                    """
                ).fetchall()

                for r in runs:
                    total = int(r["total"] or 0)
                    passed = int(r["passed"] or 0)
                    pass_rate = (passed / total) if total else 0.0

                    config_json = str(r["config_json"] or "")
                    runs_writer.writerow(
                        {
                            "source_db": str(db_path),
                            "run_id": int(r["id"]),
                            "timestamp": str(r["timestamp"]),
                            "benchmark": str(r["benchmark"]),
                            "backend_name": str(r["backend_name"]),
                            "model_id": str(r["model_id"]),
                            "suite_name": _row_value(r, "suite_name"),
                            "suite_entry_name": _row_value(r, "suite_entry_name"),
                            "loop_budget": int(r["loop_budget"]),
                            "timeout_s": int(r["timeout_s"]),
                            "total": total,
                            "passed": passed,
                            "pass_rate": f"{pass_rate:.6f}",
                            "config_json": config_json,
                        }
                    )
                    run_rows += 1

                    tasks = conn.execute(
                        """
                        SELECT tr.* FROM task_results tr
                        WHERE tr.run_id = ?
                        ORDER BY tr.task_id ASC
                        """,
                        (int(r["id"]),),
                    ).fetchall()

                    for tr in tasks:
                        row = {
                            "source_db": str(db_path),
                            "run_id": int(r["id"]),
                            "timestamp": str(r["timestamp"]),
                            "benchmark": str(r["benchmark"]),
                            "backend_name": str(r["backend_name"]),
                            "model_id": str(r["model_id"]),
                            "suite_name": _row_value(r, "suite_name"),
                            "suite_entry_name": _row_value(r, "suite_entry_name"),
                            "loop_budget": int(r["loop_budget"]),
                            "timeout_s": int(r["timeout_s"]),
                            "task_id": str(tr["task_id"]),
                            "passed": int(tr["passed"]),
                            "attempts_used": int(tr["attempts_used"]),
                            "time_ms": int(tr["time_ms"]),
                            "exit_code": _row_value(tr, "exit_code"),
                            "timed_out": int(_row_value(tr, "timed_out", 0) or 0),
                            "code_sha256": _row_value(tr, "code_sha256"),
                            "provider": _row_value(tr, "provider"),
                            "response_model": _row_value(tr, "response_model"),
                            "prompt_tokens": _row_value(tr, "prompt_tokens"),
                            "completion_tokens": _row_value(tr, "completion_tokens"),
                            "total_tokens": _row_value(tr, "total_tokens"),
                            "terminal_reason": _row_value(tr, "terminal_reason"),
                            "turns_to_first_edit": _row_value(tr, "turns_to_first_edit"),
                            "turns_to_first_verification": _row_value(
                                tr, "turns_to_first_verification"
                            ),
                            "turns_after_first_edit_before_first_verification": _row_value(
                                tr, "turns_after_first_edit_before_first_verification"
                            ),
                            "zero_edit": int(_row_value(tr, "zero_edit", 1) or 0),
                            "zero_verification": int(_row_value(tr, "zero_verification", 1) or 0),
                            "verification_succeeded": int(
                                _row_value(tr, "verification_succeeded", 0) or 0
                            ),
                            "malformed_tool_call_recoveries": int(
                                _row_value(tr, "malformed_tool_call_recoveries", 0) or 0
                            ),
                            "invalid_tool_call_count": int(
                                _row_value(tr, "invalid_tool_call_count", 0) or 0
                            ),
                            "blocked_finalizer_count": int(
                                _row_value(tr, "blocked_finalizer_count", 0) or 0
                            ),
                            "repeated_failed_run_test_count": int(
                                _row_value(tr, "repeated_failed_run_test_count", 0) or 0
                            ),
                            "post_edit_exploration_count": int(
                                _row_value(tr, "post_edit_exploration_count", 0) or 0
                            ),
                            "submission_json": _row_value(tr, "submission_json"),
                            "config_json": config_json,
                        }
                        if include_logs:
                            row.update(
                                {
                                    "stdout": _row_value(tr, "stdout"),
                                    "stderr": _row_value(tr, "stderr"),
                                    "error": _row_value(tr, "error"),
                                    "prompt_snapshot": _row_value(tr, "prompt_snapshot"),
                                }
                            )
                        tasks_writer.writerow(row)
                        task_rows += 1
            finally:
                conn.close()

    diagnostic_fields = [
        "source_db",
        "run_id",
        "task_id",
        "event_index",
        "turn",
        "event_type",
        "payload_json",
    ]
    diagnostic_handle = None
    try:
        diagnostic_writer = None
        for db_path in db_paths:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                if not _sqlite_table_exists(conn, "diagnostic_events"):
                    continue
                events = conn.execute(
                    """
                    SELECT run_id, task_id, event_index, turn, event_type, payload_json
                    FROM diagnostic_events
                    ORDER BY run_id, task_id, event_index
                    """
                ).fetchall()
                if not events:
                    continue
                if diagnostic_handle is None:
                    diagnostic_handle = diagnostic_csv.open("w", newline="", encoding="utf-8")
                    diagnostic_writer = csv.DictWriter(
                        diagnostic_handle, fieldnames=diagnostic_fields
                    )
                    diagnostic_writer.writeheader()
                assert diagnostic_writer is not None
                for event in events:
                    diagnostic_writer.writerow(
                        {
                            "source_db": str(db_path),
                            "run_id": int(event["run_id"]),
                            "task_id": str(event["task_id"]),
                            "event_index": int(event["event_index"]),
                            "turn": _row_value(event, "turn"),
                            "event_type": str(event["event_type"]),
                            "payload_json": str(event["payload_json"]),
                        }
                    )
                    diagnostic_rows += 1
            finally:
                conn.close()
    finally:
        if diagnostic_handle is not None:
            diagnostic_handle.close()

    artifact_specs = [
        (
            "artifact_tasks",
            artifact_tasks_csv,
            [
                "run_id",
                "task_id",
                "benchmark",
                "suite_name",
                "suite_entry_name",
                "phase",
                "artifact_root",
                "manifest_path",
                "schema_version",
                "repo_id",
                "task_digest",
                "candidate_count",
                "evaluation_count",
                "metadata_json",
            ],
            "run_id, task_id",
        ),
        (
            "artifact_candidates",
            artifact_candidates_csv,
            [
                "run_id",
                "task_id",
                "suite_name",
                "suite_entry_name",
                "candidate_index",
                "selected",
                "patch_path",
                "patch_sha256",
                "patch_byte_count",
                "touched_file_count",
                "added_lines",
                "deleted_lines",
                "terminal_reason",
                "submission_json",
                "generation_time_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "provider",
                "response_model",
                "validation_passed_count",
                "validation_failed_count",
                "zero_edit",
                "zero_verification",
                "verification_succeeded",
                "trace_path",
                "failure_counters_json",
                "metadata_json",
            ],
            "run_id, task_id, candidate_index",
        ),
        (
            "artifact_evaluations",
            artifact_evaluations_csv,
            [
                "run_id",
                "task_id",
                "suite_name",
                "suite_entry_name",
                "evaluation_index",
                "source_candidate_index",
                "evaluator_name",
                "passed",
                "timed_out",
                "exit_code",
                "report_path",
                "stdout_preview_path",
                "stderr_preview_path",
                "error_class",
                "runtime_ms",
                "metadata_json",
            ],
            "run_id, task_id, evaluation_index",
        ),
        (
            "artifact_verification_evidence",
            artifact_evidence_csv,
            [
                "run_id",
                "task_id",
                "suite_name",
                "suite_entry_name",
                "candidate_index",
                "evidence_index",
                "verifier_name",
                "command_label",
                "command_digest",
                "status",
                "counted_as_verification",
                "output_digest",
                "output_preview_path",
                "execution_time_ms",
                "started_at",
                "ended_at",
                "timed_out",
                "metadata_json",
            ],
            "run_id, task_id, candidate_index, evidence_index",
        ),
    ]
    artifact_counts: dict[str, int] = {}
    for table, csv_path, fields, order_by in artifact_specs:
        row_count = 0
        with csv_path.open("w", newline="", encoding="utf-8") as af:
            writer = csv.DictWriter(af, fieldnames=["source_db", *fields])
            writer.writeheader()
            for db_path in db_paths:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        f"""
                        SELECT t.*,
                               r.suite_name AS suite_name,
                               r.suite_entry_name AS suite_entry_name
                        FROM {table} t
                        JOIN runs r ON r.id = t.run_id
                        ORDER BY {order_by}
                        """
                    ).fetchall()
                    for row in rows:
                        writer.writerow(
                            {
                                "source_db": str(db_path),
                                **{field: _row_value(row, field) for field in fields},
                            }
                        )
                        row_count += 1
                finally:
                    conn.close()
        artifact_counts[table] = row_count

    artifact_task_rows = artifact_counts.get("artifact_tasks", 0)
    artifact_candidate_rows = artifact_counts.get("artifact_candidates", 0)
    artifact_evaluation_rows = artifact_counts.get("artifact_evaluations", 0)
    artifact_evidence_rows = artifact_counts.get("artifact_verification_evidence", 0)

    report = {
        "dbs": len(db_paths),
        "runs": run_rows,
        "task_results": task_rows,
        "diagnostic_events": diagnostic_rows,
        "artifact_tasks": artifact_task_rows,
        "artifact_candidates": artifact_candidate_rows,
        "artifact_evaluations": artifact_evaluation_rows,
        "artifact_verification_evidence": artifact_evidence_rows,
        "runs_csv": runs_csv,
        "task_results_csv": tasks_csv,
        "artifact_tasks_csv": artifact_tasks_csv,
        "artifact_candidates_csv": artifact_candidates_csv,
        "artifact_evaluations_csv": artifact_evaluations_csv,
        "artifact_verification_evidence_csv": artifact_evidence_csv,
    }
    if diagnostic_rows:
        report["diagnostic_events_csv"] = diagnostic_csv
    return report


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _row_value(row: sqlite3.Row, key: str, default=None):
    keys = row.keys() if hasattr(row, "keys") else ()
    if key in keys:
        return row[key]
    return default
