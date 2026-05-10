from __future__ import annotations

import sqlite3


def copy_artifact_task_from_conn(
    *,
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    src_run_id: int,
    dst_run_id: int,
    task_id: str,
) -> None:
    """Copy one task's artifact rows from a shard DB into the merged DB."""

    if not _table_exists(src_conn, "artifact_tasks"):
        return
    _delete_existing_artifact_rows(dst_conn, dst_run_id=dst_run_id, task_id=task_id)

    task_row = src_conn.execute(
        """
        SELECT * FROM artifact_tasks
        WHERE run_id = ? AND task_id = ?
        LIMIT 1
        """,
        (src_run_id, task_id),
    ).fetchone()
    if task_row is None:
        return

    _copy_task_row(dst_conn, dst_run_id=dst_run_id, task_row=task_row)
    _copy_candidate_rows(
        src_conn,
        dst_conn,
        src_run_id=src_run_id,
        dst_run_id=dst_run_id,
        task_id=task_id,
    )
    _copy_evidence_rows(
        src_conn,
        dst_conn,
        src_run_id=src_run_id,
        dst_run_id=dst_run_id,
        task_id=task_id,
    )
    _copy_evaluation_rows(
        src_conn,
        dst_conn,
        src_run_id=src_run_id,
        dst_run_id=dst_run_id,
        task_id=task_id,
    )


def _delete_existing_artifact_rows(
    dst_conn: sqlite3.Connection,
    *,
    dst_run_id: int,
    task_id: str,
) -> None:
    for table in (
        "artifact_tasks",
        "artifact_candidates",
        "artifact_verification_evidence",
        "artifact_evaluations",
    ):
        dst_conn.execute(
            f"DELETE FROM {table} WHERE run_id = ? AND task_id = ?",
            (dst_run_id, task_id),
        )


def _copy_task_row(
    dst_conn: sqlite3.Connection,
    *,
    dst_run_id: int,
    task_row: sqlite3.Row,
) -> None:
    dst_conn.execute(
        """
        INSERT OR REPLACE INTO artifact_tasks
        (run_id, task_id, benchmark, phase, artifact_root, manifest_path, schema_version,
         repo_id, task_digest, candidate_count, evaluation_count, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dst_run_id,
            str(task_row["task_id"]),
            str(task_row["benchmark"]),
            str(task_row["phase"]),
            str(task_row["artifact_root"]),
            str(task_row["manifest_path"]),
            int(task_row["schema_version"]),
            _row_value(task_row, "repo_id"),
            _row_value(task_row, "task_digest"),
            int(_row_value(task_row, "candidate_count", 0) or 0),
            int(_row_value(task_row, "evaluation_count", 0) or 0),
            str(_row_value(task_row, "metadata_json", "{}") or "{}"),
        ),
    )


def _copy_candidate_rows(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    *,
    src_run_id: int,
    dst_run_id: int,
    task_id: str,
) -> None:
    candidate_rows = src_conn.execute(
        """
        SELECT * FROM artifact_candidates
        WHERE run_id = ? AND task_id = ?
        ORDER BY candidate_index
        """,
        (src_run_id, task_id),
    ).fetchall()
    dst_conn.executemany(
        """
        INSERT OR REPLACE INTO artifact_candidates
        (
          run_id, task_id, candidate_index, selected, patch_path, patch_sha256,
          patch_byte_count, touched_file_count, added_lines, deleted_lines,
          terminal_reason, submission_json, generation_time_ms, prompt_tokens,
          completion_tokens, total_tokens, provider, response_model,
          validation_passed_count, validation_failed_count, zero_edit,
          zero_verification, verification_succeeded, trace_path,
          failure_counters_json, metadata_json
        )
        VALUES (
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            (
                dst_run_id,
                str(row["task_id"]),
                int(row["candidate_index"]),
                int(_row_value(row, "selected", 0) or 0),
                str(row["patch_path"]),
                _row_value(row, "patch_sha256"),
                _row_value(row, "patch_byte_count"),
                int(_row_value(row, "touched_file_count", 0) or 0),
                int(_row_value(row, "added_lines", 0) or 0),
                int(_row_value(row, "deleted_lines", 0) or 0),
                _row_value(row, "terminal_reason"),
                _row_value(row, "submission_json"),
                _row_value(row, "generation_time_ms"),
                _row_value(row, "prompt_tokens"),
                _row_value(row, "completion_tokens"),
                _row_value(row, "total_tokens"),
                _row_value(row, "provider"),
                _row_value(row, "response_model"),
                _row_value(row, "validation_passed_count"),
                _row_value(row, "validation_failed_count"),
                int(_row_value(row, "zero_edit", 1) or 0),
                int(_row_value(row, "zero_verification", 1) or 0),
                int(_row_value(row, "verification_succeeded", 0) or 0),
                _row_value(row, "trace_path"),
                str(_row_value(row, "failure_counters_json", "{}") or "{}"),
                str(_row_value(row, "metadata_json", "{}") or "{}"),
            )
            for row in candidate_rows
        ],
    )


def _copy_evidence_rows(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    *,
    src_run_id: int,
    dst_run_id: int,
    task_id: str,
) -> None:
    if not _table_exists(src_conn, "artifact_verification_evidence"):
        return
    evidence_rows = src_conn.execute(
        """
        SELECT * FROM artifact_verification_evidence
        WHERE run_id = ? AND task_id = ?
        ORDER BY candidate_index, evidence_index
        """,
        (src_run_id, task_id),
    ).fetchall()
    dst_conn.executemany(
        """
        INSERT OR REPLACE INTO artifact_verification_evidence
        (run_id, task_id, candidate_index, evidence_index, verifier_name, command_label,
         command_digest, status, counted_as_verification, output_digest, output_preview_path,
         execution_time_ms, started_at, ended_at, timed_out, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dst_run_id,
                str(row["task_id"]),
                int(row["candidate_index"]),
                int(row["evidence_index"]),
                str(row["verifier_name"]),
                str(row["command_label"]),
                str(row["command_digest"]),
                str(row["status"]),
                int(_row_value(row, "counted_as_verification", 0) or 0),
                str(row["output_digest"]),
                _row_value(row, "output_preview_path"),
                _row_value(row, "execution_time_ms"),
                _row_value(row, "started_at"),
                _row_value(row, "ended_at"),
                int(_row_value(row, "timed_out", 0) or 0),
                str(_row_value(row, "metadata_json", "{}") or "{}"),
            )
            for row in evidence_rows
        ],
    )


def _copy_evaluation_rows(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    *,
    src_run_id: int,
    dst_run_id: int,
    task_id: str,
) -> None:
    if not _table_exists(src_conn, "artifact_evaluations"):
        return
    evaluation_rows = src_conn.execute(
        """
        SELECT * FROM artifact_evaluations
        WHERE run_id = ? AND task_id = ?
        ORDER BY evaluation_index
        """,
        (src_run_id, task_id),
    ).fetchall()
    dst_conn.executemany(
        """
        INSERT OR REPLACE INTO artifact_evaluations
        (run_id, task_id, evaluation_index, source_candidate_index, evaluator_name, passed,
         timed_out, exit_code, report_path, stdout_preview_path, stderr_preview_path,
         error_class, runtime_ms, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dst_run_id,
                str(row["task_id"]),
                int(row["evaluation_index"]),
                int(row["source_candidate_index"]),
                str(row["evaluator_name"]),
                int(_row_value(row, "passed", 0) or 0),
                int(_row_value(row, "timed_out", 0) or 0),
                _row_value(row, "exit_code"),
                _row_value(row, "report_path"),
                _row_value(row, "stdout_preview_path"),
                _row_value(row, "stderr_preview_path"),
                _row_value(row, "error_class"),
                _row_value(row, "runtime_ms"),
                str(_row_value(row, "metadata_json", "{}") or "{}"),
            )
            for row in evaluation_rows
        ],
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
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
