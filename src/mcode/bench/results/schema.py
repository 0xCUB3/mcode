from __future__ import annotations

import sqlite3


def init_results_schema(conn: sqlite3.Connection) -> None:
    """Create and lightly migrate the results database schema.

    The project keeps old benchmark DBs around as research artifacts, so schema
    setup is intentionally append-only: create missing tables/indexes, then add
    columns that appeared after early runs. There is no destructive migration
    here.
    """

    conn.execute(
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
    conn.execute(
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
    conn.execute(
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
    conn.execute(
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
    conn.execute(
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
    conn.execute(
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
    conn.execute(
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

    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_bench ON runs(benchmark)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_results_run ON task_results(run_id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_run_task_unique ON task_results(run_id, task_id)"
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_diagnostic_events_run_task_event
        ON diagnostic_events(run_id, task_id, event_index)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_tasks_run_task
        ON artifact_tasks(run_id, task_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_candidates_run_task_candidate
        ON artifact_candidates(run_id, task_id, candidate_index)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_verification_run_task_candidate_event
        ON artifact_verification_evidence(
            run_id, task_id, candidate_index, evidence_index
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_evaluations_run_task_eval
        ON artifact_evaluations(run_id, task_id, evaluation_index)
        """
    )

    _ensure_column(conn, "runs", "backend_name", "TEXT NOT NULL DEFAULT 'ollama'")
    _ensure_column(conn, "task_results", "terminal_reason", "TEXT")
    _ensure_column(conn, "task_results", "turns_to_first_edit", "INTEGER")
    _ensure_column(
        conn,
        "task_results",
        "turns_after_first_edit_before_first_verification",
        "INTEGER",
    )
    _ensure_column(conn, "runs", "suite_name", "TEXT")
    _ensure_column(conn, "runs", "suite_entry_name", "TEXT")
    _ensure_column(conn, "task_results", "turns_to_first_verification", "INTEGER")
    _ensure_column(conn, "task_results", "zero_edit", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "task_results", "zero_verification", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "task_results", "verification_succeeded", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(
        conn,
        "task_results",
        "malformed_tool_call_recoveries",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "task_results",
        "invalid_tool_call_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "task_results",
        "blocked_finalizer_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "task_results",
        "repeated_failed_run_test_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "task_results",
        "post_edit_exploration_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(conn, "task_results", "prompt_snapshot", "TEXT")
    _ensure_column(conn, "task_results", "prompt_tokens", "INTEGER")
    _ensure_column(conn, "task_results", "completion_tokens", "INTEGER")
    _ensure_column(conn, "task_results", "total_tokens", "INTEGER")
    _ensure_column(conn, "task_results", "provider", "TEXT")
    _ensure_column(conn, "task_results", "response_model", "TEXT")
    _ensure_column(conn, "task_results", "submission_json", "TEXT")
    conn.commit()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column in table_columns(conn, table):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
