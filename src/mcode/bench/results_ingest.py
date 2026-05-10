from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path


def ingest_one(
    *,
    dst_conn: sqlite3.Connection,
    dst_path: Path,
    input_db: Path,
    insert_run: Callable[..., sqlite3.Cursor],
) -> None:
    if dst_path.resolve() == input_db.resolve():
        raise ValueError("Refusing to merge a DB into itself.")

    src = sqlite3.connect(input_db)
    src.row_factory = sqlite3.Row
    try:
        runs = src.execute(
            """
            SELECT
              id,
              timestamp,
              benchmark,
              backend_name,
              model_id,
              loop_budget,
              timeout_s,
              config_json
            FROM runs
            ORDER BY id
            """
        ).fetchall()
        for run in runs:
            cur = insert_run(
                timestamp=str(run["timestamp"]),
                benchmark=str(run["benchmark"]),
                backend_name=str(run["backend_name"]),
                model_id=str(run["model_id"]),
                loop_budget=int(run["loop_budget"]),
                timeout_s=int(run["timeout_s"]),
                config_json=str(run["config_json"]),
            )
            new_run_id = int(cur.lastrowid)
            old_run_id = int(run["id"])

            task_rows = src.execute(
                """
                SELECT * FROM task_results
                WHERE run_id = ?
                ORDER BY id
                """,
                (old_run_id,),
            ).fetchall()

            dst_conn.executemany(
                """
                INSERT OR REPLACE INTO task_results
                (run_id, task_id, passed, attempts_used, time_ms,
                 exit_code, timed_out, stdout, stderr, error, code_sha256,
                 terminal_reason, turns_to_first_edit, turns_to_first_verification,
                 turns_after_first_edit_before_first_verification, zero_edit,
                 zero_verification, verification_succeeded,
                 malformed_tool_call_recoveries, invalid_tool_call_count,
                 blocked_finalizer_count, repeated_failed_run_test_count,
                 post_edit_exploration_count, prompt_snapshot, prompt_tokens,
                 completion_tokens, total_tokens, provider, response_model,
                 submission_json)
                VALUES (
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    (
                        new_run_id,
                        str(tr["task_id"]),
                        int(tr["passed"]),
                        int(tr["attempts_used"]),
                        int(tr["time_ms"]),
                        _row_value(tr, "exit_code"),
                        int(_row_value(tr, "timed_out", 0) or 0),
                        _row_value(tr, "stdout"),
                        _row_value(tr, "stderr"),
                        _row_value(tr, "error"),
                        _row_value(tr, "code_sha256"),
                        _row_value(tr, "terminal_reason"),
                        _row_value(tr, "turns_to_first_edit"),
                        _row_value(tr, "turns_to_first_verification"),
                        _row_value(tr, "turns_after_first_edit_before_first_verification"),
                        int(_row_value(tr, "zero_edit", 1) or 0),
                        int(_row_value(tr, "zero_verification", 1) or 0),
                        int(_row_value(tr, "verification_succeeded", 0) or 0),
                        int(_row_value(tr, "malformed_tool_call_recoveries", 0) or 0),
                        int(_row_value(tr, "invalid_tool_call_count", 0) or 0),
                        int(_row_value(tr, "blocked_finalizer_count", 0) or 0),
                        int(_row_value(tr, "repeated_failed_run_test_count", 0) or 0),
                        int(_row_value(tr, "post_edit_exploration_count", 0) or 0),
                        _row_value(tr, "prompt_snapshot"),
                        _row_value(tr, "prompt_tokens"),
                        _row_value(tr, "completion_tokens"),
                        _row_value(tr, "total_tokens"),
                        _row_value(tr, "provider"),
                        _row_value(tr, "response_model"),
                        _row_value(tr, "submission_json"),
                    )
                    for tr in task_rows
                ],
            )
            if _sqlite_table_exists(src, "diagnostic_events"):
                event_rows = src.execute(
                    """
                    SELECT task_id, event_index, turn, event_type, payload_json
                    FROM diagnostic_events
                    WHERE run_id = ?
                    ORDER BY id
                    """,
                    (old_run_id,),
                ).fetchall()
                dst_conn.executemany(
                    """
                    INSERT INTO diagnostic_events
                    (run_id, task_id, event_index, turn, event_type, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            new_run_id,
                            str(event["task_id"]),
                            int(event["event_index"]),
                            _row_value(event, "turn"),
                            str(event["event_type"]),
                            str(event["payload_json"]),
                        )
                        for event in event_rows
                    ],
                )
            if _sqlite_table_exists(src, "artifact_tasks"):
                artifact_rows = src.execute(
                    """
                    SELECT * FROM artifact_tasks
                    WHERE run_id = ?
                    ORDER BY task_id
                    """,
                    (old_run_id,),
                ).fetchall()
                dst_conn.executemany(
                    """
                    INSERT OR REPLACE INTO artifact_tasks
                    (
                      run_id, task_id, benchmark, phase, artifact_root, manifest_path,
                      schema_version, repo_id, task_digest, candidate_count,
                      evaluation_count, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            new_run_id,
                            str(ar["task_id"]),
                            str(ar["benchmark"]),
                            str(ar["phase"]),
                            str(ar["artifact_root"]),
                            str(ar["manifest_path"]),
                            int(ar["schema_version"]),
                            _row_value(ar, "repo_id"),
                            _row_value(ar, "task_digest"),
                            int(_row_value(ar, "candidate_count", 0) or 0),
                            int(_row_value(ar, "evaluation_count", 0) or 0),
                            str(_row_value(ar, "metadata_json", "{}") or "{}"),
                        )
                        for ar in artifact_rows
                    ],
                )
                candidate_rows = src.execute(
                    """
                    SELECT * FROM artifact_candidates
                    WHERE run_id = ?
                    ORDER BY task_id, candidate_index
                    """,
                    (old_run_id,),
                ).fetchall()
                dst_conn.executemany(
                    """
                    INSERT OR REPLACE INTO artifact_candidates
                    (
                      run_id, task_id, candidate_index, selected, patch_path,
                      patch_sha256, patch_byte_count, touched_file_count, added_lines,
                      deleted_lines, terminal_reason, submission_json,
                      generation_time_ms, prompt_tokens, completion_tokens, total_tokens,
                      provider, response_model, validation_passed_count,
                      validation_failed_count, zero_edit, zero_verification,
                      verification_succeeded, trace_path, failure_counters_json,
                      metadata_json
                    )
                    VALUES (
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        (
                            new_run_id,
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
                if _sqlite_table_exists(src, "artifact_verification_evidence"):
                    evidence_rows = src.execute(
                        """
                        SELECT * FROM artifact_verification_evidence
                        WHERE run_id = ?
                        ORDER BY task_id, candidate_index, evidence_index
                        """,
                        (old_run_id,),
                    ).fetchall()
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO artifact_verification_evidence
                        (
                          run_id, task_id, candidate_index, evidence_index,
                          verifier_name, command_label, command_digest, status,
                          counted_as_verification, output_digest, output_preview_path,
                          execution_time_ms, started_at, ended_at, timed_out,
                          metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                new_run_id,
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
                if _sqlite_table_exists(src, "artifact_evaluations"):
                    evaluation_rows = src.execute(
                        """
                        SELECT * FROM artifact_evaluations
                        WHERE run_id = ?
                        ORDER BY task_id, evaluation_index
                        """,
                        (old_run_id,),
                    ).fetchall()
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO artifact_evaluations
                        (
                          run_id, task_id, evaluation_index, source_candidate_index,
                          evaluator_name, passed, timed_out, exit_code, report_path,
                          stdout_preview_path, stderr_preview_path, error_class,
                          runtime_ms, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                new_run_id,
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
    finally:
        src.close()


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
