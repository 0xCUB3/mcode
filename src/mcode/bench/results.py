from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcode.bench import results_ingest, results_metrics
from mcode.bench.artifacts import TaskArtifactManifest
from mcode.bench.results_artifacts_copy import copy_artifact_task_from_conn
from mcode.bench.results_schema import init_results_schema, table_columns


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class ResultsDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> ResultsDB:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        init_results_schema(self.conn)

    def _table_columns(self, table: str) -> set[str]:
        return table_columns(self.conn, table)

    def _insert_run(
        self,
        *,
        timestamp: str,
        benchmark: str,
        backend_name: str,
        model_id: str,
        loop_budget: int,
        timeout_s: int,
        config_json: str,
    ) -> sqlite3.Cursor:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            config = {}
        suite_name = config.get("suite_name")
        suite_entry_name = config.get("suite_entry_name")
        if "retrieval" in self._table_columns("runs"):
            return self.conn.execute(
                """
                INSERT INTO runs
                (
                  timestamp,
                  benchmark,
                  backend_name,
                  model_id,
                  loop_budget,
                  timeout_s,
                  retrieval,
                  config_json,
                  suite_name,
                  suite_entry_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    benchmark,
                    backend_name,
                    model_id,
                    loop_budget,
                    timeout_s,
                    0,
                    config_json,
                    suite_name,
                    suite_entry_name,
                ),
            )
        return self.conn.execute(
            """
            INSERT INTO runs
            (
              timestamp,
              benchmark,
              backend_name,
              model_id,
              loop_budget,
              timeout_s,
              config_json,
              suite_name,
              suite_entry_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                benchmark,
                backend_name,
                model_id,
                loop_budget,
                timeout_s,
                config_json,
                suite_name,
                suite_entry_name,
            ),
        )

    def start_run(self, benchmark: str, config: dict) -> int:
        now = datetime.now(UTC).isoformat()
        cursor = self._insert_run(
            timestamp=now,
            benchmark=benchmark,
            backend_name=config.get("backend_name", "ollama"),
            model_id=config["model_id"],
            loop_budget=config.get("loop_budget", 3),
            timeout_s=config["timeout_s"],
            config_json=_config_json(config),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def find_latest_run_by_config(self, benchmark: str, config: dict) -> int | None:
        row = self.conn.execute(
            """
            SELECT id FROM runs
            WHERE benchmark = ? AND config_json = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (benchmark, _config_json(config)),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def task_terminal_rows(self, run_id: int) -> dict[str, dict[str, object]]:
        rows = self.conn.execute(
            """
            SELECT
              task_id,
              passed,
              error,
              terminal_reason,
              timed_out,
              exit_code
            FROM task_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return {
            str(row["task_id"]): {
                "passed": bool(row["passed"]),
                "error": row["error"],
                "terminal_reason": row["terminal_reason"],
                "timed_out": bool(row["timed_out"]),
                "exit_code": row["exit_code"],
            }
            for row in rows
        }

    def run_summary(self, run_id: int) -> RunSummary:
        row = self.conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              COALESCE(SUM(CASE WHEN passed THEN 1 ELSE 0 END), 0) AS passed
            FROM task_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return RunSummary(
            run_id=run_id,
            total=int(row["total"]),
            passed=int(row["passed"]),
        )

    def save_task_result(self, run_id: int, result: dict) -> None:
        enriched = _enrich_result_with_diagnostic_metrics(result)
        task_id = str(enriched["task_id"])
        self.conn.execute(
            """
            INSERT OR REPLACE INTO task_results
            (run_id, task_id, passed, attempts_used, time_ms, exit_code,
             timed_out, stdout, stderr, error, code_sha256, terminal_reason,
             turns_to_first_edit, turns_to_first_verification,
             turns_after_first_edit_before_first_verification, zero_edit,
             zero_verification, verification_succeeded, malformed_tool_call_recoveries,
             invalid_tool_call_count, blocked_finalizer_count,
             repeated_failed_run_test_count, post_edit_exploration_count,
             prompt_snapshot, prompt_tokens, completion_tokens, total_tokens,
             provider, response_model, submission_json)
            VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?
            )
            """,
            (
                run_id,
                task_id,
                1 if enriched["passed"] else 0,
                enriched.get("attempts_used", 1),
                enriched["time_ms"],
                enriched.get("exit_code"),
                1 if enriched.get("timed_out", False) else 0,
                enriched.get("stdout"),
                enriched.get("stderr"),
                enriched.get("error"),
                enriched.get("code_sha256"),
                enriched.get("terminal_reason"),
                enriched.get("turns_to_first_edit"),
                enriched.get("turns_to_first_verification"),
                enriched.get("turns_after_first_edit_before_first_verification"),
                1 if enriched.get("zero_edit", True) else 0,
                1 if enriched.get("zero_verification", True) else 0,
                1 if enriched.get("verification_succeeded", False) else 0,
                enriched.get("malformed_tool_call_recoveries", 0),
                enriched.get("invalid_tool_call_count", 0),
                enriched.get("blocked_finalizer_count", 0),
                enriched.get("repeated_failed_run_test_count", 0),
                enriched.get("post_edit_exploration_count", 0),
                enriched.get("prompt_snapshot"),
                enriched.get("prompt_tokens"),
                enriched.get("completion_tokens"),
                enriched.get("total_tokens"),
                enriched.get("provider"),
                enriched.get("response_model"),
                enriched.get("submission_json"),
            ),
        )
        self.conn.execute(
            "DELETE FROM diagnostic_events WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        events = enriched.get("diagnostic_events")
        if isinstance(events, list) and events:
            self.conn.executemany(
                """
                INSERT INTO diagnostic_events
                (run_id, task_id, event_index, turn, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    _coerce_diagnostic_event(run_id, task_id, index, event)
                    for index, event in enumerate(events)
                    if isinstance(event, dict)
                ],
            )
        self.conn.commit()

    def save_task_artifact_manifest(
        self,
        run_id: int,
        manifest: TaskArtifactManifest,
        *,
        manifest_path: Path | None = None,
    ) -> None:
        task = manifest.task
        task_id = str(task.task_id)
        resolved_manifest_path = str(manifest_path or task.artifact_root)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO artifact_tasks
            (run_id, task_id, benchmark, phase, artifact_root, manifest_path, schema_version,
             repo_id, task_digest, candidate_count, evaluation_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                task_id,
                task.benchmark,
                manifest.phase,
                task.artifact_root,
                resolved_manifest_path,
                int(manifest.schema_version),
                task.repo_id,
                task.task_digest,
                len(manifest.candidates),
                len(manifest.evaluations),
                json.dumps(manifest.metadata, sort_keys=True, default=str),
            ),
        )
        self.conn.execute(
            "DELETE FROM artifact_candidates WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        self.conn.execute(
            "DELETE FROM artifact_verification_evidence WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        self.conn.execute(
            "DELETE FROM artifact_evaluations WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        for candidate in manifest.candidates:
            self.conn.execute(
                """
                INSERT INTO artifact_candidates
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
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    task_id,
                    candidate.candidate_index,
                    1 if candidate.selected else 0,
                    candidate.patch_path,
                    candidate.patch_stats.sha256,
                    candidate.patch_stats.byte_count,
                    len(candidate.patch_stats.touched_files),
                    candidate.patch_stats.added_lines,
                    candidate.patch_stats.deleted_lines,
                    candidate.terminal_reason,
                    candidate.submission_json,
                    candidate.generation_time_ms,
                    candidate.prompt_tokens,
                    candidate.completion_tokens,
                    candidate.total_tokens,
                    candidate.provider,
                    candidate.response_model,
                    candidate.validation_passed_count,
                    candidate.validation_failed_count,
                    1 if candidate.zero_edit else 0,
                    1 if candidate.zero_verification else 0,
                    1 if candidate.verification_succeeded else 0,
                    candidate.trace_path,
                    json.dumps(candidate.failure_counters, sort_keys=True, default=str),
                    json.dumps(candidate.metadata, sort_keys=True, default=str),
                ),
            )
            for evidence_index, evidence in enumerate(candidate.verification_evidence):
                self.conn.execute(
                    """
                    INSERT INTO artifact_verification_evidence
                    (
                      run_id, task_id, candidate_index, evidence_index, verifier_name,
                      command_label, command_digest, status, counted_as_verification,
                      output_digest, output_preview_path, execution_time_ms, started_at,
                      ended_at, timed_out, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        candidate.candidate_index,
                        evidence_index,
                        evidence.verifier_name,
                        evidence.command_label,
                        evidence.command_digest,
                        evidence.status,
                        1 if evidence.counted_as_verification else 0,
                        evidence.output_digest,
                        evidence.output_preview_path,
                        evidence.execution_time_ms,
                        evidence.started_at,
                        evidence.ended_at,
                        1 if evidence.timed_out else 0,
                        json.dumps(evidence.metadata, sort_keys=True, default=str),
                    ),
                )
        for evaluation_index, evaluation in enumerate(manifest.evaluations):
            self.conn.execute(
                """
                INSERT INTO artifact_evaluations
                (run_id, task_id, evaluation_index, source_candidate_index, evaluator_name, passed,
                 timed_out, exit_code, report_path, stdout_preview_path, stderr_preview_path,
                 error_class, runtime_ms, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    evaluation_index,
                    evaluation.source_candidate_index,
                    evaluation.evaluator_name,
                    1 if evaluation.passed else 0,
                    1 if evaluation.timed_out else 0,
                    evaluation.exit_code,
                    evaluation.report_path,
                    evaluation.stdout_preview_path,
                    evaluation.stderr_preview_path,
                    evaluation.error_class,
                    evaluation.runtime_ms,
                    json.dumps(evaluation.metadata, sort_keys=True, default=str),
                ),
            )
        self.conn.commit()

    def delete_task_result(self, run_id: int, task_id: str) -> None:
        self.conn.execute(
            "DELETE FROM diagnostic_events WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        self.conn.execute(
            "DELETE FROM task_results WHERE run_id = ? AND task_id = ?",
            (run_id, task_id),
        )
        self.conn.commit()

    def task_artifact_rows(self, run_id: int) -> dict[str, dict[str, object]]:
        if not _sqlite_table_exists(self.conn, "artifact_tasks"):
            return {}
        rows = self.conn.execute(
            """
            SELECT
              at.task_id,
              at.phase,
              at.artifact_root,
              at.manifest_path,
              at.schema_version,
              at.candidate_count,
              at.evaluation_count,
              at.repo_id,
              at.task_digest,
              at.metadata_json,
              ac.candidate_index AS selected_candidate_index,
              ac.patch_byte_count AS selected_patch_byte_count,
              ac.verification_succeeded AS selected_verification_succeeded
            FROM artifact_tasks at
            LEFT JOIN artifact_candidates ac
              ON ac.run_id = at.run_id AND ac.task_id = at.task_id AND ac.selected = 1
            WHERE at.run_id = ?
            """,
            (run_id,),
        ).fetchall()
        out: dict[str, dict[str, object]] = {}
        for row in rows:
            metadata_json = _row_value(row, "metadata_json")
            try:
                metadata = json.loads(str(metadata_json)) if metadata_json else {}
            except json.JSONDecodeError:
                metadata = {"raw": str(metadata_json)}
            out[str(row["task_id"])] = {
                "phase": str(row["phase"]),
                "artifact_root": str(row["artifact_root"]),
                "manifest_path": str(row["manifest_path"]),
                "schema_version": int(row["schema_version"]),
                "candidate_count": int(row["candidate_count"] or 0),
                "evaluation_count": int(row["evaluation_count"] or 0),
                "repo_id": _row_value(row, "repo_id"),
                "task_digest": _row_value(row, "task_digest"),
                "selected_candidate_index": _row_value(row, "selected_candidate_index"),
                "selected_patch_byte_count": _row_value(row, "selected_patch_byte_count"),
                "selected_verification_succeeded": bool(
                    _row_value(row, "selected_verification_succeeded", 0) or 0
                ),
                "metadata": metadata,
            }
        return out

    def pass_rates_grouped(
        self,
        *,
        benchmark: str | None,
        model_id: str | None,
        backend_name: str | None = None,
        timeout_s: int | None = None,
        suite_name: str | None = None,
        suite_entry_name: str | None = None,
        group_by: Sequence[str],
        loop_budget: int | None = None,
    ) -> list[dict]:
        return results_metrics.pass_rates_grouped(
            self.conn,
            benchmark=benchmark,
            model_id=model_id,
            backend_name=backend_name,
            timeout_s=timeout_s,
            suite_name=suite_name,
            suite_entry_name=suite_entry_name,
            group_by=group_by,
            loop_budget=loop_budget,
        )

    def run_metrics_grouped(
        self,
        *,
        benchmark: str | None,
        model_id: str | None,
        backend_name: str | None = None,
        timeout_s: int | None = None,
        suite_name: str | None = None,
        suite_entry_name: str | None = None,
        group_by: Sequence[str],
        loop_budget: int | None = None,
        include_percentiles: bool = True,
    ) -> list[dict]:
        return results_metrics.run_metrics_grouped(
            self.conn,
            benchmark=benchmark,
            model_id=model_id,
            backend_name=backend_name,
            timeout_s=timeout_s,
            suite_name=suite_name,
            suite_entry_name=suite_entry_name,
            group_by=group_by,
            loop_budget=loop_budget,
            include_percentiles=include_percentiles,
        )

    def merge_from(self, inputs: Sequence[Path]) -> None:
        if not inputs:
            return

        self.conn.execute("BEGIN")
        try:
            for p in inputs:
                self._ingest_one(p)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _ingest_one(self, input_db: Path) -> None:
        results_ingest.ingest_one(
            dst_conn=self.conn,
            dst_path=self.path,
            input_db=input_db,
            insert_run=self._insert_run,
        )


def _coerce_diagnostic_event(
    run_id: int,
    task_id: str,
    event_index: int,
    event: dict[str, object],
) -> tuple[object, ...]:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {"value": payload}
    turn = event.get("turn")
    try:
        normalized_turn = int(turn) if turn is not None else None
    except (TypeError, ValueError):
        normalized_turn = None
    return (
        run_id,
        task_id,
        event_index,
        normalized_turn,
        str(event.get("event_type", "unknown")),
        json.dumps(payload, sort_keys=True, default=str),
    )


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


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


def _enrich_result_with_diagnostic_metrics(result: dict) -> dict:
    enriched = dict(result)
    events = enriched.get("diagnostic_events")
    derived = _derive_diagnostic_counters(events if isinstance(events, list) else [])
    for key, value in derived.items():
        enriched.setdefault(key, value)
    enriched.setdefault(
        "turns_after_first_edit_before_first_verification",
        _turn_gap(
            enriched.get("turns_to_first_edit"),
            enriched.get("turns_to_first_verification"),
        ),
    )
    return enriched


def _derive_diagnostic_counters(events: list[object]) -> dict[str, int]:
    counts = {
        "malformed_tool_call_recoveries": 0,
        "invalid_tool_call_count": 0,
        "blocked_finalizer_count": 0,
        "repeated_failed_run_test_count": 0,
        "post_edit_exploration_count": 0,
    }
    first_edit_seen = False
    first_verification_seen = False
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", ""))
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if event_type == "tool_call_filter":
            counts["invalid_tool_call_count"] += _coerce_nonnegative_int(
                payload.get("invalid_call_count")
            )
            counts["blocked_finalizer_count"] += _coerce_nonnegative_int(
                payload.get("blocked_finalizer_count")
            )
            continue
        if event_type == "tool_arg_compat":
            counts["malformed_tool_call_recoveries"] += _coerce_nonnegative_int(
                payload.get("recoverable_call_count")
            )
            continue
        if event_type == "final_answer" and payload.get("action") == "autofilled":
            counts["malformed_tool_call_recoveries"] += 1
            continue
        if (
            event_type == "edit_result"
            and payload.get("status") == "APPLIED"
            and not first_edit_seen
        ):
            first_edit_seen = True
            continue
        if event_type == "run_tests":
            if payload.get("repeated_failed_run_suppressed") is True:
                counts["repeated_failed_run_test_count"] += 1
            first_verification_seen = True
            continue
        if event_type == "read_search_target" and first_edit_seen and not first_verification_seen:
            counts["post_edit_exploration_count"] += 1
    return counts


def _coerce_nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _turn_gap(start: object, end: object) -> int | None:
    try:
        start_turn = int(start) if start is not None else None
        end_turn = int(end) if end is not None else None
    except (TypeError, ValueError):
        return None
    if start_turn is None or end_turn is None:
        return None
    return max(0, end_turn - start_turn)


def _config_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _row_value(row: sqlite3.Row, key: str, default=None):
    keys = row.keys() if hasattr(row, "keys") else ()
    if key in keys:
        return row[key]
    return default


def merge_shard_dbs(*, out_path: Path, shard_paths: list[Path], force: bool = False) -> dict:
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

    out_db = ResultsDB(out_path)
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
