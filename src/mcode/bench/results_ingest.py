from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from mcode.bench.results_artifacts_copy import copy_artifact_task_from_conn
from mcode.bench.results_sqlite import row_value as _row_value
from mcode.bench.results_sqlite import sqlite_table_exists as _sqlite_table_exists


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
                artifact_tasks = src.execute(
                    """
                    SELECT task_id FROM artifact_tasks
                    WHERE run_id = ?
                    ORDER BY task_id
                    """,
                    (old_run_id,),
                ).fetchall()
                for artifact_task in artifact_tasks:
                    copy_artifact_task_from_conn(
                        src_conn=src,
                        dst_conn=dst_conn,
                        src_run_id=old_run_id,
                        dst_run_id=new_run_id,
                        task_id=str(artifact_task["task_id"]),
                    )
    finally:
        src.close()
