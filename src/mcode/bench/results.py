from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcode.bench import results_export, results_ingest, results_merge, results_metrics
from mcode.bench.artifacts import TaskArtifactManifest
from mcode.bench.results_schema import init_results_schema, table_columns
from mcode.bench.results_sqlite import row_value as _row_value
from mcode.bench.results_sqlite import sqlite_table_exists as _sqlite_table_exists


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


def merge_shard_dbs(*, out_path: Path, shard_paths: list[Path], force: bool = False) -> dict:
    return results_merge.merge_shard_dbs(
        out_path=out_path,
        shard_paths=shard_paths,
        force=force,
        results_db_factory=ResultsDB,
    )


def export_csv(
    *,
    inputs: list[Path],
    out_dir: Path,
    prefix: str = "mcode",
    include_logs: bool = False,
) -> dict:
    return results_export.export_csv(
        inputs=inputs,
        out_dir=out_dir,
        prefix=prefix,
        include_logs=include_logs,
    )
