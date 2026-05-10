from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from mcode.bench.results_artifacts_copy import copy_artifact_task_from_conn


def merge_shard_dbs(
    *,
    out_path: Path,
    shard_paths: list[Path],
    force: bool = False,
    results_db_factory: Callable[[Path], object],
) -> dict:
    """
    Merge shard SQLite DBs (from sharded runs) into a single run DB.

    If an indexed job retries a shard, multiple DBs for the same shard index may exist.
    We pick the shard DB with the most task_results rows (tie-breaker: newest mtime).
    """

    if not shard_paths:
        raise ValueError("No shard DBs provided")

    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard DB(s): {', '.join(str(p) for p in missing)}")

    pat = re.compile(r"^(?P<bench>.+)-shard-(?P<idx>\d+)\.db$")
    groups: dict[str, list[Path]] = {}
    for p in shard_paths:
        m = pat.match(p.name)
        key = f"{m.group('bench')}-shard-{m.group('idx')}" if m else p.name
        groups.setdefault(key, []).append(p)

    chosen: list[Path] = []
    ignored: list[Path] = []
    for _, paths in sorted(groups.items()):
        if len(paths) == 1:
            chosen.append(paths[0])
            continue

        best: Path | None = None
        best_count = -1
        best_mtime = -1.0
        for p in paths:
            try:
                conn = sqlite3.connect(p)
                try:
                    task_result_count = int(
                        conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
                    )
                    artifact_count = 0
                    if _sqlite_table_exists(conn, "artifact_tasks"):
                        artifact_count = int(
                            conn.execute("SELECT COUNT(*) FROM artifact_tasks").fetchone()[0]
                        )
                    count = max(task_result_count, artifact_count)
                finally:
                    conn.close()
            except Exception:
                count = 0
            mtime = p.stat().st_mtime
            if (count > best_count) or (count == best_count and mtime > best_mtime):
                best = p
                best_count = count
                best_mtime = mtime
        assert best is not None
        chosen.append(best)
        ignored.extend([p for p in paths if p != best])

    if out_path.exists():
        if not force:
            raise FileExistsError(f"Output DB already exists: {out_path} (use --force)")
        out_path.unlink()

    # Read config from the first shard.
    first = sqlite3.connect(chosen[0])
    first.row_factory = sqlite3.Row
    try:
        row = first.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("No runs found in shard DB")
        benchmark = str(row["benchmark"])
        config = json.loads(row["config_json"])
    finally:
        first.close()

    out_db = results_db_factory(out_path)
    run_id = out_db.start_run(benchmark, dict(config))

    seen: set[str] = set()
    written = 0
    for shard in chosen:
        conn = sqlite3.connect(shard)
        conn.row_factory = sqlite3.Row
        try:
            result_rows = conn.execute(
                """
                SELECT * FROM task_results
                """
            ).fetchall()
            result_by_task = {str(row["task_id"]): row for row in result_rows}
            artifact_by_task: dict[str, sqlite3.Row] = {}
            if _sqlite_table_exists(conn, "artifact_tasks"):
                artifact_rows = conn.execute(
                    """
                    SELECT * FROM artifact_tasks
                    """
                ).fetchall()
                artifact_by_task = {str(row["task_id"]): row for row in artifact_rows}
            task_ids = sorted(set(result_by_task) | set(artifact_by_task))
            for task_id in task_ids:
                if task_id in seen:
                    continue
                seen.add(task_id)
                result_row = result_by_task.get(task_id)
                if result_row is not None:
                    result = {
                        "task_id": task_id,
                        "passed": bool(result_row["passed"]),
                        "attempts_used": int(result_row["attempts_used"]),
                        "time_ms": int(result_row["time_ms"]),
                        "exit_code": _row_value(result_row, "exit_code"),
                        "timed_out": bool(_row_value(result_row, "timed_out", 0)),
                        "stdout": _row_value(result_row, "stdout"),
                        "stderr": _row_value(result_row, "stderr"),
                        "error": _row_value(result_row, "error"),
                        "code_sha256": _row_value(result_row, "code_sha256"),
                        "terminal_reason": _row_value(result_row, "terminal_reason"),
                        "turns_to_first_edit": _row_value(result_row, "turns_to_first_edit"),
                        "turns_to_first_verification": _row_value(
                            result_row, "turns_to_first_verification"
                        ),
                        "turns_after_first_edit_before_first_verification": _row_value(
                            result_row, "turns_after_first_edit_before_first_verification"
                        ),
                        "zero_edit": bool(_row_value(result_row, "zero_edit", 1)),
                        "zero_verification": bool(_row_value(result_row, "zero_verification", 1)),
                        "verification_succeeded": bool(
                            _row_value(result_row, "verification_succeeded", 0)
                        ),
                        "malformed_tool_call_recoveries": _row_value(
                            result_row, "malformed_tool_call_recoveries", 0
                        ),
                        "invalid_tool_call_count": _row_value(
                            result_row, "invalid_tool_call_count", 0
                        ),
                        "blocked_finalizer_count": _row_value(
                            result_row, "blocked_finalizer_count", 0
                        ),
                        "repeated_failed_run_test_count": _row_value(
                            result_row, "repeated_failed_run_test_count", 0
                        ),
                        "post_edit_exploration_count": _row_value(
                            result_row, "post_edit_exploration_count", 0
                        ),
                        "prompt_snapshot": _row_value(result_row, "prompt_snapshot"),
                        "prompt_tokens": _row_value(result_row, "prompt_tokens"),
                        "completion_tokens": _row_value(result_row, "completion_tokens"),
                        "total_tokens": _row_value(result_row, "total_tokens"),
                        "provider": _row_value(result_row, "provider"),
                        "response_model": _row_value(result_row, "response_model"),
                        "submission_json": _row_value(result_row, "submission_json"),
                    }
                    diagnostic_events = _diagnostic_events_for_task(
                        conn, int(result_row["run_id"]), task_id
                    )
                    if diagnostic_events:
                        result["diagnostic_events"] = diagnostic_events
                    out_db.save_task_result(run_id, result)
                artifact_task = artifact_by_task.get(task_id)
                if artifact_task is not None:
                    copy_artifact_task_from_conn(
                        src_conn=conn,
                        dst_conn=out_db.conn,
                        src_run_id=int(artifact_task["run_id"]),
                        dst_run_id=run_id,
                        task_id=task_id,
                    )
                written += 1
        finally:
            conn.close()

    merged_config = dict(config)
    merged_config["task_shard_count"] = None
    merged_config["task_shard_index"] = None
    merged_config["planned_task_count"] = written
    merged_config["merged_shards"] = len(chosen)
    out_db.conn.execute(
        "UPDATE runs SET config_json = ? WHERE id = ?",
        (_config_json(merged_config), run_id),
    )
    out_db.conn.commit()

    return {
        "out_path": out_path,
        "benchmark": benchmark,
        "run_id": run_id,
        "tasks_written": written,
        "shards_used": len(chosen),
        "shards_ignored": len(ignored),
    }



def _diagnostic_events_for_task(
    conn: sqlite3.Connection,
    run_id: int,
    task_id: str,
) -> list[dict[str, object]]:
    if not _sqlite_table_exists(conn, "diagnostic_events"):
        return []
    rows = conn.execute(
        """
        SELECT turn, event_type, payload_json
        FROM diagnostic_events
        WHERE run_id = ? AND task_id = ?
        ORDER BY event_index
        """,
        (run_id, task_id),
    ).fetchall()
    events: list[dict[str, object]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            payload = {"raw": str(row["payload_json"])}
        events.append(
            {
                "turn": _row_value(row, "turn"),
                "event_type": str(row["event_type"]),
                "payload": payload,
            }
        )
    return events


def _config_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


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
