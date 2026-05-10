from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from mcode.bench.results_sqlite import row_value as _row_value
from mcode.bench.results_sqlite import sqlite_table_exists as _sqlite_table_exists

RUN_FIELDS = [
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

BASE_TASK_FIELDS = [
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

LOG_TASK_FIELDS = ["stdout", "stderr", "error", "prompt_snapshot"]

DIAGNOSTIC_FIELDS = [
    "source_db",
    "run_id",
    "task_id",
    "event_index",
    "turn",
    "event_type",
    "payload_json",
]

ARTIFACT_SPECS = [
    (
        "artifact_tasks",
        "artifact_tasks",
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
        "artifact_candidates",
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
        "artifact_evaluations",
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
        "artifact_verification_evidence",
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

    db_paths = _discover_db_paths(inputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = out_dir / f"{prefix}.runs.csv"
    tasks_csv = out_dir / f"{prefix}.task_results.csv"
    diagnostic_csv = out_dir / f"{prefix}.diagnostic_events.csv"

    run_rows, task_rows = _export_runs_and_tasks(
        db_paths=db_paths,
        runs_csv=runs_csv,
        tasks_csv=tasks_csv,
        include_logs=include_logs,
    )
    diagnostic_rows = _export_diagnostics(db_paths=db_paths, diagnostic_csv=diagnostic_csv)
    artifact_counts, artifact_csvs = _export_artifacts(
        db_paths=db_paths,
        out_dir=out_dir,
        prefix=prefix,
    )

    report = {
        "dbs": len(db_paths),
        "runs": run_rows,
        "task_results": task_rows,
        "diagnostic_events": diagnostic_rows,
        "artifact_tasks": artifact_counts.get("artifact_tasks", 0),
        "artifact_candidates": artifact_counts.get("artifact_candidates", 0),
        "artifact_evaluations": artifact_counts.get("artifact_evaluations", 0),
        "artifact_verification_evidence": artifact_counts.get("artifact_verification_evidence", 0),
        "runs_csv": runs_csv,
        "task_results_csv": tasks_csv,
        "artifact_tasks_csv": artifact_csvs["artifact_tasks"],
        "artifact_candidates_csv": artifact_csvs["artifact_candidates"],
        "artifact_evaluations_csv": artifact_csvs["artifact_evaluations"],
        "artifact_verification_evidence_csv": artifact_csvs["artifact_verification_evidence"],
    }
    if diagnostic_rows:
        report["diagnostic_events_csv"] = diagnostic_csv
    return report


def _discover_db_paths(inputs: list[Path]) -> list[Path]:
    db_paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            db_paths.extend(sorted(path.glob("*.db")))
        else:
            db_paths.append(path)
    db_paths = [
        path
        for path in db_paths
        if path.exists() and path.suffix == ".db" and "shard-" not in path.name
    ]
    db_paths = sorted(set(db_paths))
    if not db_paths:
        raise FileNotFoundError("No .db files found (pass --input <db|dir> ...).")
    return db_paths


def _export_runs_and_tasks(
    *,
    db_paths: list[Path],
    runs_csv: Path,
    tasks_csv: Path,
    include_logs: bool,
) -> tuple[int, int]:
    task_fields = [*BASE_TASK_FIELDS, *(LOG_TASK_FIELDS if include_logs else [])]
    run_rows = 0
    task_rows = 0

    with (
        runs_csv.open("w", newline="", encoding="utf-8") as runs_handle,
        tasks_csv.open("w", newline="", encoding="utf-8") as tasks_handle,
    ):
        runs_writer = csv.DictWriter(runs_handle, fieldnames=RUN_FIELDS)
        tasks_writer = csv.DictWriter(tasks_handle, fieldnames=task_fields)
        runs_writer.writeheader()
        tasks_writer.writeheader()

        for db_path in db_paths:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                for run in _run_rows(conn):
                    config_json = str(run["config_json"] or "")
                    runs_writer.writerow(_run_csv_row(db_path, run, config_json))
                    run_rows += 1

                    for task in _task_rows(conn, int(run["id"])):
                        tasks_writer.writerow(
                            _task_csv_row(
                                db_path,
                                run,
                                task,
                                config_json,
                                include_logs=include_logs,
                            )
                        )
                        task_rows += 1
            finally:
                conn.close()
    return run_rows, task_rows


def _run_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
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


def _task_rows(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT tr.* FROM task_results tr
        WHERE tr.run_id = ?
        ORDER BY tr.task_id ASC
        """,
        (run_id,),
    ).fetchall()


def _run_csv_row(db_path: Path, run: sqlite3.Row, config_json: str) -> dict[str, object]:
    total = int(run["total"] or 0)
    passed = int(run["passed"] or 0)
    pass_rate = (passed / total) if total else 0.0
    return {
        "source_db": str(db_path),
        "run_id": int(run["id"]),
        "timestamp": str(run["timestamp"]),
        "benchmark": str(run["benchmark"]),
        "backend_name": str(run["backend_name"]),
        "model_id": str(run["model_id"]),
        "suite_name": _row_value(run, "suite_name"),
        "suite_entry_name": _row_value(run, "suite_entry_name"),
        "loop_budget": int(run["loop_budget"]),
        "timeout_s": int(run["timeout_s"]),
        "total": total,
        "passed": passed,
        "pass_rate": f"{pass_rate:.6f}",
        "config_json": config_json,
    }


def _task_csv_row(
    db_path: Path,
    run: sqlite3.Row,
    task: sqlite3.Row,
    config_json: str,
    *,
    include_logs: bool,
) -> dict[str, object]:
    row = {
        "source_db": str(db_path),
        "run_id": int(run["id"]),
        "timestamp": str(run["timestamp"]),
        "benchmark": str(run["benchmark"]),
        "backend_name": str(run["backend_name"]),
        "model_id": str(run["model_id"]),
        "suite_name": _row_value(run, "suite_name"),
        "suite_entry_name": _row_value(run, "suite_entry_name"),
        "loop_budget": int(run["loop_budget"]),
        "timeout_s": int(run["timeout_s"]),
        "task_id": str(task["task_id"]),
        "passed": int(task["passed"]),
        "attempts_used": int(task["attempts_used"]),
        "time_ms": int(task["time_ms"]),
        "exit_code": _row_value(task, "exit_code"),
        "timed_out": int(_row_value(task, "timed_out", 0) or 0),
        "code_sha256": _row_value(task, "code_sha256"),
        "provider": _row_value(task, "provider"),
        "response_model": _row_value(task, "response_model"),
        "prompt_tokens": _row_value(task, "prompt_tokens"),
        "completion_tokens": _row_value(task, "completion_tokens"),
        "total_tokens": _row_value(task, "total_tokens"),
        "terminal_reason": _row_value(task, "terminal_reason"),
        "turns_to_first_edit": _row_value(task, "turns_to_first_edit"),
        "turns_to_first_verification": _row_value(task, "turns_to_first_verification"),
        "turns_after_first_edit_before_first_verification": _row_value(
            task, "turns_after_first_edit_before_first_verification"
        ),
        "zero_edit": int(_row_value(task, "zero_edit", 1) or 0),
        "zero_verification": int(_row_value(task, "zero_verification", 1) or 0),
        "verification_succeeded": int(_row_value(task, "verification_succeeded", 0) or 0),
        "malformed_tool_call_recoveries": int(
            _row_value(task, "malformed_tool_call_recoveries", 0) or 0
        ),
        "invalid_tool_call_count": int(_row_value(task, "invalid_tool_call_count", 0) or 0),
        "blocked_finalizer_count": int(_row_value(task, "blocked_finalizer_count", 0) or 0),
        "repeated_failed_run_test_count": int(
            _row_value(task, "repeated_failed_run_test_count", 0) or 0
        ),
        "post_edit_exploration_count": int(_row_value(task, "post_edit_exploration_count", 0) or 0),
        "submission_json": _row_value(task, "submission_json"),
        "config_json": config_json,
    }
    if include_logs:
        row.update({field: _row_value(task, field) for field in LOG_TASK_FIELDS})
    return row


def _export_diagnostics(*, db_paths: list[Path], diagnostic_csv: Path) -> int:
    row_count = 0
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
                        diagnostic_handle, fieldnames=DIAGNOSTIC_FIELDS
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
                    row_count += 1
            finally:
                conn.close()
    finally:
        if diagnostic_handle is not None:
            diagnostic_handle.close()
    return row_count


def _export_artifacts(*, db_paths: list[Path], out_dir: Path, prefix: str) -> tuple[dict, dict]:
    counts: dict[str, int] = {}
    csvs = {
        table: out_dir / f"{prefix}.{csv_name}.csv"
        for table, csv_name, _fields, _order_by in ARTIFACT_SPECS
    }
    for table, _csv_name, fields, order_by in ARTIFACT_SPECS:
        csv_path = csvs[table]
        row_count = 0
        with csv_path.open("w", newline="", encoding="utf-8") as artifact_handle:
            writer = csv.DictWriter(artifact_handle, fieldnames=["source_db", *fields])
            writer.writeheader()
            for db_path in db_paths:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                try:
                    for row in _artifact_rows(conn, table=table, order_by=order_by):
                        writer.writerow(
                            {
                                "source_db": str(db_path),
                                **{field: _row_value(row, field) for field in fields},
                            }
                        )
                        row_count += 1
                finally:
                    conn.close()
        counts[table] = row_count
    return counts, csvs


def _artifact_rows(conn: sqlite3.Connection, *, table: str, order_by: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT t.*,
               r.suite_name AS suite_name,
               r.suite_entry_name AS suite_entry_name
        FROM {table} t
        JOIN runs r ON r.id = t.run_id
        ORDER BY {order_by}
        """
    ).fetchall()
