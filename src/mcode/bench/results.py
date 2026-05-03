from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcode.bench.artifacts import TaskArtifactManifest

_TERMINAL_REASON_BUCKETS = (
    "budget_exhausted",
    "unverified_diff_discarded",
    "wrong_patch_after_verification",
    "infra_failure",
    "submitted",
)


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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY,
              timestamp TEXT NOT NULL,
              benchmark TEXT NOT NULL,
              backend_name TEXT NOT NULL,
              model_id TEXT NOT NULL,
              loop_budget INTEGER NOT NULL,
              timeout_s INTEGER NOT NULL,
              config_json TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              passed INTEGER NOT NULL,
              attempts_used INTEGER NOT NULL,
              time_ms INTEGER NOT NULL,
              exit_code INTEGER,
              timed_out INTEGER NOT NULL,
              stdout TEXT,
              stderr TEXT,
              error TEXT,
              code_sha256 TEXT,
              terminal_reason TEXT,
              turns_to_first_edit INTEGER,
              turns_to_first_verification INTEGER,
              turns_after_first_edit_before_first_verification INTEGER,
              zero_edit INTEGER NOT NULL DEFAULT 1,
              zero_verification INTEGER NOT NULL DEFAULT 1,
              verification_succeeded INTEGER NOT NULL DEFAULT 0,
              malformed_tool_call_recoveries INTEGER NOT NULL DEFAULT 0,
              invalid_tool_call_count INTEGER NOT NULL DEFAULT 0,
              blocked_finalizer_count INTEGER NOT NULL DEFAULT 0,
              repeated_failed_run_test_count INTEGER NOT NULL DEFAULT 0,
              post_edit_exploration_count INTEGER NOT NULL DEFAULT 0,
              prompt_snapshot TEXT,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              provider TEXT,
              response_model TEXT,
              submission_json TEXT,
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostic_events (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              event_index INTEGER NOT NULL,
              turn INTEGER,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_tasks (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              benchmark TEXT NOT NULL,
              phase TEXT NOT NULL,
              artifact_root TEXT NOT NULL,
              manifest_path TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              repo_id TEXT,
              task_digest TEXT,
              candidate_count INTEGER NOT NULL DEFAULT 0,
              evaluation_count INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_candidates (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              candidate_index INTEGER NOT NULL,
              selected INTEGER NOT NULL DEFAULT 0,
              patch_path TEXT NOT NULL,
              patch_sha256 TEXT,
              patch_byte_count INTEGER,
              touched_file_count INTEGER NOT NULL DEFAULT 0,
              added_lines INTEGER NOT NULL DEFAULT 0,
              deleted_lines INTEGER NOT NULL DEFAULT 0,
              terminal_reason TEXT,
              submission_json TEXT,
              generation_time_ms INTEGER,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              provider TEXT,
              response_model TEXT,
              validation_passed_count INTEGER,
              validation_failed_count INTEGER,
              zero_edit INTEGER NOT NULL DEFAULT 1,
              zero_verification INTEGER NOT NULL DEFAULT 1,
              verification_succeeded INTEGER NOT NULL DEFAULT 0,
              trace_path TEXT,
              failure_counters_json TEXT NOT NULL DEFAULT '{}',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_verification_evidence (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              candidate_index INTEGER NOT NULL,
              evidence_index INTEGER NOT NULL,
              verifier_name TEXT NOT NULL,
              command_label TEXT NOT NULL,
              command_digest TEXT NOT NULL,
              status TEXT NOT NULL,
              counted_as_verification INTEGER NOT NULL DEFAULT 0,
              output_digest TEXT NOT NULL,
              output_preview_path TEXT,
              execution_time_ms INTEGER,
              started_at TEXT,
              ended_at TEXT,
              timed_out INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_evaluations (
              id INTEGER PRIMARY KEY,
              run_id INTEGER NOT NULL,
              task_id TEXT NOT NULL,
              evaluation_index INTEGER NOT NULL,
              source_candidate_index INTEGER NOT NULL,
              evaluator_name TEXT NOT NULL,
              passed INTEGER NOT NULL DEFAULT 0,
              timed_out INTEGER NOT NULL DEFAULT 0,
              exit_code INTEGER,
              report_path TEXT,
              stdout_preview_path TEXT,
              stderr_preview_path TEXT,
              error_class TEXT,
              runtime_ms INTEGER,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_bench ON runs(benchmark)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_task_results_run ON task_results(run_id)")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_run_task_unique ON task_results(run_id, task_id)"
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_diagnostic_events_run_task_event
            ON diagnostic_events(run_id, task_id, event_index)
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_tasks_run_task
            ON artifact_tasks(run_id, task_id)
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_candidates_run_task_candidate
            ON artifact_candidates(run_id, task_id, candidate_index)
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_verification_run_task_candidate_event
            ON artifact_verification_evidence(
                run_id, task_id, candidate_index, evidence_index
            )
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_evaluations_run_task_eval
            ON artifact_evaluations(run_id, task_id, evaluation_index)
            """
        )
        self._ensure_column("runs", "backend_name", "TEXT NOT NULL DEFAULT 'ollama'")
        self._ensure_column("task_results", "terminal_reason", "TEXT")
        self._ensure_column("task_results", "turns_to_first_edit", "INTEGER")
        self._ensure_column(
            "task_results",
            "turns_after_first_edit_before_first_verification",
            "INTEGER",
        )
        self._ensure_column("runs", "suite_name", "TEXT")
        self._ensure_column("runs", "suite_entry_name", "TEXT")
        self._ensure_column("task_results", "turns_to_first_verification", "INTEGER")
        self._ensure_column("task_results", "zero_edit", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("task_results", "zero_verification", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("task_results", "verification_succeeded", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(
            "task_results",
            "malformed_tool_call_recoveries",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "task_results",
            "invalid_tool_call_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "task_results",
            "blocked_finalizer_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "task_results",
            "repeated_failed_run_test_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "task_results",
            "post_edit_exploration_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column("task_results", "prompt_snapshot", "TEXT")
        self._ensure_column("task_results", "prompt_tokens", "INTEGER")
        self._ensure_column("task_results", "completion_tokens", "INTEGER")
        self._ensure_column("task_results", "total_tokens", "INTEGER")
        self._ensure_column("task_results", "provider", "TEXT")
        self._ensure_column("task_results", "response_model", "TEXT")
        self._ensure_column("task_results", "submission_json", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cols = self._table_columns(table)
        if column in cols:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _table_columns(self, table: str) -> set[str]:
        return {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}

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
        group_map = {
            "loop_budget": "r.loop_budget",
            "backend_name": "r.backend_name",
            "timeout_s": "r.timeout_s",
            "suite_name": "r.suite_name",
            "suite_entry_name": "r.suite_entry_name",
        }
        if any(g not in group_map for g in group_by):
            raise ValueError(f"Unsupported group_by: {group_by}")

        where = ["1=1"]
        params: list[object] = []
        if benchmark:
            where.append("r.benchmark = ?")
            params.append(benchmark)
        if model_id:
            where.append("r.model_id = ?")
            params.append(model_id)
        if backend_name:
            where.append("r.backend_name = ?")
            params.append(backend_name)
        if timeout_s is not None:
            where.append("r.timeout_s = ?")
            params.append(int(timeout_s))
        if suite_name:
            where.append("r.suite_name = ?")
            params.append(suite_name)
        if suite_entry_name:
            where.append("r.suite_entry_name = ?")
            params.append(suite_entry_name)
        if loop_budget is not None:
            where.append("r.loop_budget = ?")
            params.append(int(loop_budget))

        if not group_by:
            sql = f"""
              SELECT
                r.id AS run_id,
                r.timestamp AS timestamp,
                r.benchmark AS benchmark,
                r.backend_name AS backend_name,
                r.model_id AS model_id,
                r.config_json AS config_json,
                r.suite_name AS suite_name,
                r.suite_entry_name AS suite_entry_name,
                r.loop_budget AS loop_budget,
                r.timeout_s AS timeout_s,
                COUNT(tr.id) AS total,
                COALESCE(SUM(tr.passed), 0) AS passed,
                COALESCE(a.generated_tasks, 0) AS artifact_generated_tasks,
                COALESCE(a.evaluated_tasks, 0) AS artifact_evaluated_tasks
              FROM runs r
              LEFT JOIN task_results tr ON tr.run_id = r.id
              LEFT JOIN (
                SELECT
                  at.run_id AS run_id,
                  COUNT(DISTINCT at.task_id) AS generated_tasks,
                  COUNT(DISTINCT CASE WHEN at.evaluation_count > 0 THEN at.task_id END)
                    AS evaluated_tasks
                FROM artifact_tasks at
                GROUP BY at.run_id
              ) a ON a.run_id = r.id
              WHERE {" AND ".join(where)}
                AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
              GROUP BY r.id
              ORDER BY r.timestamp DESC
            """
            rows = self.conn.execute(sql, params).fetchall()
            out: list[dict] = []
            for row in rows:
                total = int(row["total"] or 0)
                passed = int(row["passed"] or 0)
                out.append(
                    {
                        "run_id": int(row["run_id"]),
                        "timestamp": str(row["timestamp"]),
                        "benchmark": str(row["benchmark"]),
                        "backend_name": str(row["backend_name"]),
                        "model_id": str(row["model_id"]),
                        "suite_name": _row_value(row, "suite_name"),
                        "suite_entry_name": _row_value(row, "suite_entry_name"),
                        "loop_budget": int(row["loop_budget"]),
                        "timeout_s": int(row["timeout_s"]),
                        "config_json": str(row["config_json"]),
                        "total": total,
                        "passed": passed,
                        "artifact_generated_tasks": int(
                            row["artifact_generated_tasks"] or 0
                        ),
                        "artifact_evaluated_tasks": int(
                            row["artifact_evaluated_tasks"] or 0
                        ),
                        "pass_rate": passed / total if total else 0.0,
                    }
                )
            return out

        group_exprs = [group_map[g] for g in group_by]
        base_group_cols = [
            "r.benchmark",
            "r.backend_name",
            "r.model_id",
            "r.suite_name",
            "r.suite_entry_name",
            "r.timeout_s",
            "r.loop_budget",
        ]
        group_cols = list(dict.fromkeys([*base_group_cols, *group_exprs]))
        sql = f"""
          SELECT
            r.benchmark AS benchmark,
            r.backend_name AS backend_name,
            r.model_id AS model_id,
            r.suite_name AS suite_name,
            r.suite_entry_name AS suite_entry_name,
            r.loop_budget AS loop_budget,
            r.timeout_s AS timeout_s,
            COUNT(tr.id) AS total,
            COALESCE(SUM(tr.passed), 0) AS passed,
            COALESCE(SUM(a.generated_tasks), 0) AS artifact_generated_tasks,
            COALESCE(SUM(a.evaluated_tasks), 0) AS artifact_evaluated_tasks
          FROM runs r
          LEFT JOIN task_results tr ON tr.run_id = r.id
          LEFT JOIN (
            SELECT
              at.run_id AS run_id,
              COUNT(DISTINCT at.task_id) AS generated_tasks,
              COUNT(DISTINCT CASE WHEN at.evaluation_count > 0 THEN at.task_id END)
                AS evaluated_tasks
            FROM artifact_tasks at
            GROUP BY at.run_id
          ) a ON a.run_id = r.id
          WHERE {" AND ".join(where)}
            AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
          GROUP BY {", ".join(group_cols)}
          ORDER BY
            r.benchmark,
            r.model_id,
            r.backend_name,
            r.suite_name,
            r.suite_entry_name,
            r.timeout_s,
            r.loop_budget
        """
        rows = self.conn.execute(sql, params).fetchall()
        out: list[dict] = []
        for row in rows:
            total = int(row["total"] or 0)
            passed = int(row["passed"] or 0)
            out.append(
                {
                    "benchmark": str(row["benchmark"]),
                    "backend_name": str(row["backend_name"]),
                    "model_id": str(row["model_id"]),
                    "suite_name": _row_value(row, "suite_name"),
                    "suite_entry_name": _row_value(row, "suite_entry_name"),
                    "loop_budget": int(row["loop_budget"]),
                    "timeout_s": int(row["timeout_s"]),
                    "total": total,
                    "passed": passed,
                    "artifact_generated_tasks": int(
                        row["artifact_generated_tasks"] or 0
                    ),
                    "artifact_evaluated_tasks": int(
                        row["artifact_evaluated_tasks"] or 0
                    ),
                    "pass_rate": passed / total if total else 0.0,
                }
            )
        return out

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
        group_map = {
            "loop_budget": "r.loop_budget",
            "backend_name": "r.backend_name",
            "timeout_s": "r.timeout_s",
            "suite_name": "r.suite_name",
            "suite_entry_name": "r.suite_entry_name",
        }
        if any(g not in group_map for g in group_by):
            raise ValueError(f"Unsupported group_by: {group_by}")

        where = ["1=1"]
        params: list[object] = []
        if benchmark:
            where.append("r.benchmark = ?")
            params.append(benchmark)
        if model_id:
            where.append("r.model_id = ?")
            params.append(model_id)
        if backend_name:
            where.append("r.backend_name = ?")
            params.append(backend_name)
        if timeout_s is not None:
            where.append("r.timeout_s = ?")
            params.append(int(timeout_s))
        if suite_name:
            where.append("r.suite_name = ?")
            params.append(suite_name)
        if suite_entry_name:
            where.append("r.suite_entry_name = ?")
            params.append(suite_entry_name)
        if loop_budget is not None:
            where.append("r.loop_budget = ?")
            params.append(int(loop_budget))

        reason_selects = ",\n".join(
            (
                "                SUM(CASE WHEN tr.terminal_reason = "
                f"'{reason}' THEN 1 ELSE 0 END) AS {reason}"
            )
            for reason in _TERMINAL_REASON_BUCKETS
        )
        grouped_reason_selects = ",\n".join(
            (
                "            COALESCE(SUM(run_metrics."
                f"{reason}), 0) AS {reason}"
            )
            for reason in _TERMINAL_REASON_BUCKETS
        )

        def _scaffold_fields(row: sqlite3.Row, total: int) -> dict[str, object]:
            zero_edit = int(row["zero_edit"] or 0)
            zero_verification = int(row["zero_verification"] or 0)
            verification_succeeded = int(row["verification_succeeded"] or 0)
            malformed_tool_call_recoveries = int(row["malformed_tool_call_recoveries"] or 0)
            invalid_tool_call_count = int(row["invalid_tool_call_count"] or 0)
            blocked_finalizer_count = int(row["blocked_finalizer_count"] or 0)
            repeated_failed_run_test_count = int(row["repeated_failed_run_test_count"] or 0)
            post_edit_exploration_count = int(row["post_edit_exploration_count"] or 0)
            usage_recorded = int(row["usage_recorded"] or 0)
            prompts_recorded = int(row["prompts_recorded"] or 0)
            return {
                "zero_edit": zero_edit,
                "zero_edit_rate": zero_edit / total if total else 0.0,
                "zero_verification": zero_verification,
                "zero_verification_rate": zero_verification / total if total else 0.0,
                "verification_succeeded": verification_succeeded,
                "verification_success_rate": verification_succeeded / total if total else 0.0,
                "malformed_tool_call_recoveries": malformed_tool_call_recoveries,
                "invalid_tool_call_count": invalid_tool_call_count,
                "blocked_finalizer_count": blocked_finalizer_count,
                "repeated_failed_run_test_count": repeated_failed_run_test_count,
                "post_edit_exploration_count": post_edit_exploration_count,
                "prompts_recorded": prompts_recorded,
                "prompt_record_rate": prompts_recorded / total if total else 0.0,
                "usage_recorded": usage_recorded,
                "usage_record_rate": usage_recorded / total if total else 0.0,
                "prompt_tokens_total": int(row["prompt_tokens_total"] or 0),
                "completion_tokens_total": int(row["completion_tokens_total"] or 0),
                "total_tokens_total": int(row["total_tokens_total"] or 0),
                "prompt_tokens_avg": row["prompt_tokens_avg"],
                "completion_tokens_avg": row["completion_tokens_avg"],
                "total_tokens_avg": row["total_tokens_avg"],
                "turns_to_first_edit_avg": row["turns_to_first_edit_avg"],
                "turns_to_first_verification_avg": row["turns_to_first_verification_avg"],
                "turns_after_first_edit_before_first_verification_avg": row[
                    "turns_after_first_edit_before_first_verification_avg"
                ],
                **{reason: int(row[reason] or 0) for reason in _TERMINAL_REASON_BUCKETS},
            }

        def _artifact_fields(row: sqlite3.Row) -> dict[str, object]:
            return {
                "artifact_generated_tasks": int(row["artifact_generated_tasks"] or 0),
                "artifact_evaluated_tasks": int(row["artifact_evaluated_tasks"] or 0),
                "artifact_candidate_count": int(row["artifact_candidate_count"] or 0),
                "artifact_selected_candidate_count": int(
                    row["artifact_selected_candidate_count"] or 0
                ),
                "artifact_selected_patch_byte_count_total": int(
                    row["artifact_selected_patch_byte_count_total"] or 0
                ),
                "artifact_selected_touched_file_count_total": int(
                    row["artifact_selected_touched_file_count_total"] or 0
                ),
                "artifact_selected_added_lines_total": int(
                    row["artifact_selected_added_lines_total"] or 0
                ),
                "artifact_selected_deleted_lines_total": int(
                    row["artifact_selected_deleted_lines_total"] or 0
                ),
            }

        if not group_by:
            sql = f"""
              SELECT
                r.id AS run_id,
                r.timestamp AS timestamp,
                r.benchmark AS benchmark,
                r.backend_name AS backend_name,
                r.model_id AS model_id,
                r.suite_name AS suite_name,
                r.suite_entry_name AS suite_entry_name,
                r.loop_budget AS loop_budget,
                r.timeout_s AS timeout_s,
                COUNT(tr.id) AS total,
                COALESCE(SUM(tr.passed), 0) AS passed,
                COALESCE(SUM(tr.timed_out), 0) AS timed_out,
                COALESCE(SUM(tr.time_ms), 0) AS time_ms_total,
                COALESCE(SUM(tr.zero_edit), 0) AS zero_edit,
                COALESCE(SUM(tr.zero_verification), 0) AS zero_verification,
                COALESCE(SUM(tr.verification_succeeded), 0) AS verification_succeeded,
                COALESCE(SUM(tr.malformed_tool_call_recoveries), 0)
                  AS malformed_tool_call_recoveries,
                COALESCE(SUM(tr.invalid_tool_call_count), 0) AS invalid_tool_call_count,
                COALESCE(SUM(tr.blocked_finalizer_count), 0) AS blocked_finalizer_count,
                COALESCE(SUM(tr.repeated_failed_run_test_count), 0)
                  AS repeated_failed_run_test_count,
                COALESCE(SUM(tr.post_edit_exploration_count), 0)
                  AS post_edit_exploration_count,
                COUNT(tr.prompt_snapshot) AS prompts_recorded,
                COUNT(tr.total_tokens) AS usage_recorded,
                COALESCE(SUM(tr.prompt_tokens), 0) AS prompt_tokens_total,
                COALESCE(SUM(tr.completion_tokens), 0) AS completion_tokens_total,
                COALESCE(SUM(tr.total_tokens), 0) AS total_tokens_total,
                AVG(tr.prompt_tokens) AS prompt_tokens_avg,
                AVG(tr.completion_tokens) AS completion_tokens_avg,
                AVG(tr.total_tokens) AS total_tokens_avg,
                AVG(tr.turns_to_first_edit) AS turns_to_first_edit_avg,
                AVG(tr.turns_to_first_verification) AS turns_to_first_verification_avg,
                AVG(tr.turns_after_first_edit_before_first_verification)
                AS turns_after_first_edit_before_first_verification_avg,
                COALESCE(a.generated_tasks, 0) AS artifact_generated_tasks,
                COALESCE(a.evaluated_tasks, 0) AS artifact_evaluated_tasks,
                COALESCE(a.candidate_count, 0) AS artifact_candidate_count,
                COALESCE(a.selected_candidate_count, 0) AS artifact_selected_candidate_count,
                COALESCE(a.selected_patch_byte_count_total, 0)
                  AS artifact_selected_patch_byte_count_total,
                COALESCE(a.selected_touched_file_count_total, 0)
                  AS artifact_selected_touched_file_count_total,
                COALESCE(a.selected_added_lines_total, 0)
                  AS artifact_selected_added_lines_total,
                COALESCE(a.selected_deleted_lines_total, 0)
                  AS artifact_selected_deleted_lines_total,
{reason_selects}
              FROM runs r
              LEFT JOIN task_results tr ON tr.run_id = r.id
              LEFT JOIN (
                SELECT
                  at.run_id AS run_id,
                  COUNT(DISTINCT at.task_id) AS generated_tasks,
                  COUNT(DISTINCT CASE WHEN at.evaluation_count > 0 THEN at.task_id END)
                    AS evaluated_tasks,
                  COALESCE(SUM(at.candidate_count), 0) AS candidate_count,
                  COUNT(ac.id) AS selected_candidate_count,
                  COALESCE(SUM(ac.patch_byte_count), 0) AS selected_patch_byte_count_total,
                  COALESCE(SUM(ac.touched_file_count), 0) AS selected_touched_file_count_total,
                  COALESCE(SUM(ac.added_lines), 0) AS selected_added_lines_total,
                  COALESCE(SUM(ac.deleted_lines), 0) AS selected_deleted_lines_total
                FROM artifact_tasks at
                LEFT JOIN artifact_candidates ac
                  ON ac.run_id = at.run_id AND ac.task_id = at.task_id AND ac.selected = 1
                GROUP BY at.run_id
              ) a ON a.run_id = r.id
              WHERE {" AND ".join(where)}
                AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
              GROUP BY r.id
              ORDER BY r.timestamp DESC
            """
            rows = self.conn.execute(sql, params).fetchall()
            run_ids = [int(r["run_id"]) for r in rows]
            time_stats: dict[int, dict[str, float | None]] = {}
            if include_percentiles and run_ids:
                q = ",".join("?" for _ in run_ids)
                time_rows = self.conn.execute(
                    f"SELECT run_id, time_ms FROM task_results WHERE run_id IN ({q})",
                    run_ids,
                ).fetchall()
                times_by_run: dict[int, list[int]] = {}
                for tr in time_rows:
                    rid = int(tr["run_id"])
                    times_by_run.setdefault(rid, []).append(int(tr["time_ms"]))
                for rid, times in times_by_run.items():
                    time_stats[rid] = _time_percentiles_ms(times)

            out: list[dict] = []
            for row in rows:
                total = int(row["total"])
                passed = int(row["passed"] or 0)
                timed_out = int(row["timed_out"] or 0)
                time_ms_total = int(row["time_ms_total"] or 0)
                avg_ms = (time_ms_total / total) if total else 0.0
                total_s = time_ms_total / 1000.0
                sec_per_solve = (total_s / passed) if passed else None
                solves_per_hour = (passed * 3600.0 / total_s) if total_s > 0 else 0.0

                rid = int(row["run_id"])
                p = time_stats.get(rid) if include_percentiles else None
                p50_ms = p.get("p50_ms") if p else None
                p95_ms = p.get("p95_ms") if p else None

                out.append(
                    {
                        "run_id": rid,
                        "timestamp": str(row["timestamp"]),
                        "benchmark": str(row["benchmark"]),
                        "backend_name": str(row["backend_name"]),
                        "model_id": str(row["model_id"]),
                        "suite_name": _row_value(row, "suite_name"),
                        "suite_entry_name": _row_value(row, "suite_entry_name"),
                        "loop_budget": int(row["loop_budget"]),
                        "timeout_s": int(row["timeout_s"]),
                        "total": total,
                        "passed": passed,
                        "timed_out": timed_out,
                        "pass_rate": passed / total if total else 0.0,
                        "timeout_rate": timed_out / total if total else 0.0,
                        "time_ms_total": time_ms_total,
                        "time_ms_avg": avg_ms,
                        "time_ms_p50": p50_ms,
                        "time_ms_p95": p95_ms,
                        "time_s_total": total_s,
                        "time_s_avg": avg_ms / 1000.0,
                        "time_s_p50": (p50_ms / 1000.0) if p50_ms is not None else None,
                        "time_s_p95": (p95_ms / 1000.0) if p95_ms is not None else None,
                        "sec_per_solve": sec_per_solve,
                        "solves_per_hour": solves_per_hour,
                        **_scaffold_fields(row, total),
                        **_artifact_fields(row),
                    }
                )
            return out

        group_exprs = [group_map[g] for g in group_by]
        base_group_cols = [
            "r.benchmark",
            "r.backend_name",
            "r.model_id",
            "r.suite_name",
            "r.suite_entry_name",
            "r.timeout_s",
            "r.loop_budget",
        ]
        group_cols = list(dict.fromkeys([*base_group_cols, *group_exprs]))
        group_cols_sql = ", ".join(group_cols).replace("r.", "")
        sql = f"""
          WITH run_metrics AS (
            SELECT
              r.id AS run_id,
              r.benchmark AS benchmark,
              r.backend_name AS backend_name,
              r.model_id AS model_id,
              r.suite_name AS suite_name,
              r.suite_entry_name AS suite_entry_name,
              r.loop_budget AS loop_budget,
              r.timeout_s AS timeout_s,
              COUNT(tr.id) AS total,
              COALESCE(SUM(tr.passed), 0) AS passed,
              COALESCE(SUM(tr.timed_out), 0) AS timed_out,
              COALESCE(SUM(tr.time_ms), 0) AS time_ms_total,
              COALESCE(SUM(tr.zero_edit), 0) AS zero_edit,
              COALESCE(SUM(tr.zero_verification), 0) AS zero_verification,
              COALESCE(SUM(tr.verification_succeeded), 0) AS verification_succeeded,
              COALESCE(SUM(tr.malformed_tool_call_recoveries), 0)
                AS malformed_tool_call_recoveries,
              COALESCE(SUM(tr.invalid_tool_call_count), 0) AS invalid_tool_call_count,
              COALESCE(SUM(tr.blocked_finalizer_count), 0) AS blocked_finalizer_count,
              COALESCE(SUM(tr.repeated_failed_run_test_count), 0)
                AS repeated_failed_run_test_count,
              COALESCE(SUM(tr.post_edit_exploration_count), 0)
                AS post_edit_exploration_count,
              COUNT(tr.prompt_snapshot) AS prompts_recorded,
              COUNT(tr.total_tokens) AS usage_recorded,
              COUNT(tr.prompt_tokens) AS prompt_tokens_recorded,
              COUNT(tr.completion_tokens) AS completion_tokens_recorded,
              COALESCE(SUM(tr.prompt_tokens), 0) AS prompt_tokens_total,
              COALESCE(SUM(tr.completion_tokens), 0) AS completion_tokens_total,
              COALESCE(SUM(tr.total_tokens), 0) AS total_tokens_total,
              COUNT(tr.turns_to_first_edit) AS turns_to_first_edit_recorded,
              COUNT(tr.turns_to_first_verification) AS turns_to_first_verification_recorded,
              COUNT(tr.turns_after_first_edit_before_first_verification)
                AS turns_after_first_edit_before_first_verification_recorded,
              COALESCE(SUM(tr.turns_to_first_edit), 0) AS turns_to_first_edit_total,
              COALESCE(SUM(tr.turns_to_first_verification), 0)
                AS turns_to_first_verification_total,
              COALESCE(SUM(tr.turns_after_first_edit_before_first_verification), 0)
                AS turns_after_first_edit_before_first_verification_total,
              COALESCE(a.generated_tasks, 0) AS artifact_generated_tasks,
              COALESCE(a.evaluated_tasks, 0) AS artifact_evaluated_tasks,
              COALESCE(a.candidate_count, 0) AS artifact_candidate_count,
              COALESCE(a.selected_candidate_count, 0) AS artifact_selected_candidate_count,
              COALESCE(a.selected_patch_byte_count_total, 0)
                AS artifact_selected_patch_byte_count_total,
              COALESCE(a.selected_touched_file_count_total, 0)
                AS artifact_selected_touched_file_count_total,
              COALESCE(a.selected_added_lines_total, 0)
                AS artifact_selected_added_lines_total,
              COALESCE(a.selected_deleted_lines_total, 0)
                AS artifact_selected_deleted_lines_total,
{reason_selects}
            FROM runs r
            LEFT JOIN task_results tr ON tr.run_id = r.id
            LEFT JOIN (
              SELECT
                at.run_id AS run_id,
                COUNT(DISTINCT at.task_id) AS generated_tasks,
                COUNT(DISTINCT CASE WHEN at.evaluation_count > 0 THEN at.task_id END)
                  AS evaluated_tasks,
                COALESCE(SUM(at.candidate_count), 0) AS candidate_count,
                COUNT(ac.id) AS selected_candidate_count,
                COALESCE(SUM(ac.patch_byte_count), 0) AS selected_patch_byte_count_total,
                COALESCE(SUM(ac.touched_file_count), 0) AS selected_touched_file_count_total,
                COALESCE(SUM(ac.added_lines), 0) AS selected_added_lines_total,
                COALESCE(SUM(ac.deleted_lines), 0) AS selected_deleted_lines_total
              FROM artifact_tasks at
              LEFT JOIN artifact_candidates ac
                ON ac.run_id = at.run_id AND ac.task_id = at.task_id AND ac.selected = 1
              GROUP BY at.run_id
            ) a ON a.run_id = r.id
            WHERE {" AND ".join(where)}
              AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
            GROUP BY r.id
          )
          SELECT
            benchmark,
            backend_name,
            model_id,
            suite_name,
            suite_entry_name,
            loop_budget,
            timeout_s,
            COUNT(*) AS runs,
            COALESCE(SUM(total), 0) AS total,
            COALESCE(SUM(passed), 0) AS passed,
            COALESCE(SUM(timed_out), 0) AS timed_out,
            COALESCE(SUM(time_ms_total), 0) AS time_ms_total,
            COALESCE(SUM(zero_edit), 0) AS zero_edit,
            COALESCE(SUM(zero_verification), 0) AS zero_verification,
            COALESCE(SUM(verification_succeeded), 0) AS verification_succeeded,
            COALESCE(SUM(malformed_tool_call_recoveries), 0)
              AS malformed_tool_call_recoveries,
            COALESCE(SUM(invalid_tool_call_count), 0) AS invalid_tool_call_count,
            COALESCE(SUM(blocked_finalizer_count), 0) AS blocked_finalizer_count,
            COALESCE(SUM(repeated_failed_run_test_count), 0)
              AS repeated_failed_run_test_count,
            COALESCE(SUM(post_edit_exploration_count), 0) AS post_edit_exploration_count,
            COALESCE(SUM(prompts_recorded), 0) AS prompts_recorded,
            COALESCE(SUM(usage_recorded), 0) AS usage_recorded,
            COALESCE(SUM(prompt_tokens_total), 0) AS prompt_tokens_total,
            COALESCE(SUM(completion_tokens_total), 0) AS completion_tokens_total,
            COALESCE(SUM(total_tokens_total), 0) AS total_tokens_total,
            CASE WHEN SUM(prompt_tokens_recorded) > 0
              THEN SUM(prompt_tokens_total) * 1.0 / SUM(prompt_tokens_recorded)
              ELSE NULL END AS prompt_tokens_avg,
            CASE WHEN SUM(completion_tokens_recorded) > 0
              THEN SUM(completion_tokens_total) * 1.0 / SUM(completion_tokens_recorded)
              ELSE NULL END AS completion_tokens_avg,
            CASE WHEN SUM(usage_recorded) > 0
              THEN SUM(total_tokens_total) * 1.0 / SUM(usage_recorded)
              ELSE NULL END AS total_tokens_avg,
            CASE WHEN SUM(turns_to_first_edit_recorded) > 0
              THEN SUM(turns_to_first_edit_total) * 1.0 / SUM(turns_to_first_edit_recorded)
              ELSE NULL END AS turns_to_first_edit_avg,
            CASE WHEN SUM(turns_to_first_verification_recorded) > 0
              THEN SUM(turns_to_first_verification_total) * 1.0
                / SUM(turns_to_first_verification_recorded)
              ELSE NULL END AS turns_to_first_verification_avg,
            CASE WHEN SUM(turns_after_first_edit_before_first_verification_recorded) > 0
              THEN SUM(turns_after_first_edit_before_first_verification_total) * 1.0
                / SUM(turns_after_first_edit_before_first_verification_recorded)
              ELSE NULL END AS turns_after_first_edit_before_first_verification_avg,
            COALESCE(SUM(artifact_generated_tasks), 0) AS artifact_generated_tasks,
            COALESCE(SUM(artifact_evaluated_tasks), 0) AS artifact_evaluated_tasks,
            COALESCE(SUM(artifact_candidate_count), 0) AS artifact_candidate_count,
            COALESCE(SUM(artifact_selected_candidate_count), 0)
              AS artifact_selected_candidate_count,
            COALESCE(SUM(artifact_selected_patch_byte_count_total), 0)
              AS artifact_selected_patch_byte_count_total,
            COALESCE(SUM(artifact_selected_touched_file_count_total), 0)
              AS artifact_selected_touched_file_count_total,
            COALESCE(SUM(artifact_selected_added_lines_total), 0)
              AS artifact_selected_added_lines_total,
            COALESCE(SUM(artifact_selected_deleted_lines_total), 0)
              AS artifact_selected_deleted_lines_total,
{grouped_reason_selects}
          FROM run_metrics
          GROUP BY {group_cols_sql}
          ORDER BY
            benchmark,
            model_id,
            backend_name,
            timeout_s,
            loop_budget
        """
        rows = self.conn.execute(sql, params).fetchall()

        time_stats: dict[tuple, dict[str, float | None]] = {}
        if include_percentiles and rows:
            detail_sql = f"""
              SELECT
                r.benchmark AS benchmark,
                r.backend_name AS backend_name,
                r.model_id AS model_id,
                r.suite_name AS suite_name,
                r.suite_entry_name AS suite_entry_name,
                r.timeout_s AS timeout_s,
                r.loop_budget AS loop_budget,
                tr.time_ms AS time_ms
              FROM runs r
              JOIN task_results tr ON tr.run_id = r.id
              WHERE {" AND ".join(where)}
            """
            detail_rows = self.conn.execute(detail_sql, params).fetchall()
            times_by_key: dict[tuple, list[int]] = {}
            for dr in detail_rows:
                key = (
                    str(dr["benchmark"]),
                    str(dr["backend_name"]),
                    str(dr["model_id"]),
                    _row_value(dr, "suite_name"),
                    _row_value(dr, "suite_entry_name"),
                    int(dr["timeout_s"]),
                    int(dr["loop_budget"]),
                )
                times_by_key.setdefault(key, []).append(int(dr["time_ms"]))
            for key, times in times_by_key.items():
                time_stats[key] = _time_percentiles_ms(times)

        out: list[dict] = []
        for row in rows:
            total = int(row["total"])
            passed = int(row["passed"] or 0)
            timed_out = int(row["timed_out"] or 0)
            time_ms_total = int(row["time_ms_total"] or 0)
            avg_ms = (time_ms_total / total) if total else 0.0
            total_s = time_ms_total / 1000.0
            sec_per_solve = (total_s / passed) if passed else None
            solves_per_hour = (passed * 3600.0 / total_s) if total_s > 0 else 0.0

            key = (
                str(row["benchmark"]),
                str(row["backend_name"]),
                str(row["model_id"]),
                _row_value(row, "suite_name"),
                _row_value(row, "suite_entry_name"),
                int(row["timeout_s"]),
                int(row["loop_budget"]),
            )
            p = time_stats.get(key) if include_percentiles else None
            p50_ms = p.get("p50_ms") if p else None
            p95_ms = p.get("p95_ms") if p else None

            out.append(
                {
                    "benchmark": str(row["benchmark"]),
                    "backend_name": str(row["backend_name"]),
                    "model_id": str(row["model_id"]),
                    "suite_name": _row_value(row, "suite_name"),
                    "suite_entry_name": _row_value(row, "suite_entry_name"),
                    "loop_budget": int(row["loop_budget"]),
                    "timeout_s": int(row["timeout_s"]),
                    "runs": int(row["runs"] or 0),
                    "total": total,
                    "passed": passed,
                    "timed_out": timed_out,
                    "pass_rate": passed / total if total else 0.0,
                    "timeout_rate": timed_out / total if total else 0.0,
                    "time_ms_total": time_ms_total,
                    "time_ms_avg": avg_ms,
                    "time_ms_p50": p50_ms,
                    "time_ms_p95": p95_ms,
                    "time_s_total": total_s,
                    "time_s_avg": avg_ms / 1000.0,
                    "time_s_p50": (p50_ms / 1000.0) if p50_ms is not None else None,
                    "time_s_p95": (p95_ms / 1000.0) if p95_ms is not None else None,
                    "sec_per_solve": sec_per_solve,
                    "solves_per_hour": solves_per_hour,
                    **_scaffold_fields(row, total),
                    **_artifact_fields(row),
                }
            )
        return out

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
        if self.path.resolve() == input_db.resolve():
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
                cur = self._insert_run(
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

                self.conn.executemany(
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
                    self.conn.executemany(
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
                    self.conn.executemany(
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
                    self.conn.executemany(
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
                        self.conn.executemany(
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
                        self.conn.executemany(
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


def _percentile(sorted_values: list[int], p: float) -> float | None:
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    h = (n - 1) * p
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return float(sorted_values[lower])
    frac = h - lower
    return float(sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower]))


def _time_percentiles_ms(time_ms: list[int]) -> dict[str, float | None]:
    values = sorted(int(v) for v in time_ms if v is not None)
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


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


def _copy_artifact_task_from_conn(
    *,
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    src_run_id: int,
    dst_run_id: int,
    task_id: str,
) -> None:
    if not _sqlite_table_exists(src_conn, "artifact_tasks"):
        return
    dst_conn.execute(
        "DELETE FROM artifact_tasks WHERE run_id = ? AND task_id = ?",
        (dst_run_id, task_id),
    )
    dst_conn.execute(
        "DELETE FROM artifact_candidates WHERE run_id = ? AND task_id = ?",
        (dst_run_id, task_id),
    )
    dst_conn.execute(
        "DELETE FROM artifact_verification_evidence WHERE run_id = ? AND task_id = ?",
        (dst_run_id, task_id),
    )
    dst_conn.execute(
        "DELETE FROM artifact_evaluations WHERE run_id = ? AND task_id = ?",
        (dst_run_id, task_id),
    )
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
    if _sqlite_table_exists(src_conn, "artifact_verification_evidence"):
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
    if _sqlite_table_exists(src_conn, "artifact_evaluations"):
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
                    _copy_artifact_task_from_conn(
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
                            "zero_verification": int(
                                _row_value(tr, "zero_verification", 1) or 0
                            ),
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
