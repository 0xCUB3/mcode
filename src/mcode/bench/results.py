from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


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

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY,
              timestamp TEXT NOT NULL,
              benchmark TEXT NOT NULL,
              backend_name TEXT NOT NULL,
              model_id TEXT NOT NULL,
              samples INTEGER NOT NULL,
              max_debug_iterations INTEGER NOT NULL,
              timeout_s INTEGER NOT NULL,
              retrieval INTEGER NOT NULL,
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
              samples_generated INTEGER NOT NULL,
              debug_iterations_used INTEGER NOT NULL,
              time_ms INTEGER NOT NULL,
              exit_code INTEGER,
              timed_out INTEGER NOT NULL,
              stdout TEXT,
              stderr TEXT,
              error TEXT,
              code_sha256 TEXT,
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
        self._ensure_column("runs", "backend_name", "TEXT NOT NULL DEFAULT 'ollama'")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cols = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column in cols:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def start_run(self, benchmark: str, config: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """
            INSERT INTO runs
            (
              timestamp,
              benchmark,
              backend_name,
              model_id,
              samples,
              max_debug_iterations,
              timeout_s,
              retrieval,
              config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                benchmark,
                config.get("backend_name", "ollama"),
                config["model_id"],
                config["samples"],
                config["max_debug_iterations"],
                config["timeout_s"],
                1 if config["retrieval"] else 0,
                json.dumps(config, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def save_task_result(self, run_id: int, result: dict) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO task_results
            (run_id, task_id, passed, samples_generated, debug_iterations_used, time_ms, exit_code,
             timed_out, stdout, stderr, error, code_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result["task_id"],
                1 if result["passed"] else 0,
                result["samples_generated"],
                result["debug_iterations_used"],
                result["time_ms"],
                result.get("exit_code"),
                1 if result.get("timed_out", False) else 0,
                result.get("stdout"),
                result.get("stderr"),
                result.get("error"),
                result.get("code_sha256"),
            ),
        )
        self.conn.commit()

    def pass_rates_grouped(
        self,
        *,
        benchmark: Optional[str],
        model_id: Optional[str],
        backend_name: Optional[str] = None,
        max_debug_iterations: Optional[int] = None,
        timeout_s: Optional[int] = None,
        group_by: Sequence[str],
        retrieval: Optional[bool] = None,
        samples: Optional[int] = None,
    ) -> list[dict]:
        group_map = {
            "samples": "r.samples",
            "backend_name": "r.backend_name",
            "max_debug_iterations": "r.max_debug_iterations",
            "timeout_s": "r.timeout_s",
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
        if max_debug_iterations is not None:
            where.append("r.max_debug_iterations = ?")
            params.append(int(max_debug_iterations))
        if timeout_s is not None:
            where.append("r.timeout_s = ?")
            params.append(int(timeout_s))
        if retrieval is not None:
            where.append("r.retrieval = ?")
            params.append(1 if retrieval else 0)
        if samples is not None:
            where.append("r.samples = ?")
            params.append(int(samples))

        if not group_by:
            sql = f"""
              SELECT
                r.id AS run_id,
                r.timestamp AS timestamp,
                r.benchmark AS benchmark,
                r.backend_name AS backend_name,
                r.model_id AS model_id,
                r.retrieval AS retrieval,
                r.config_json AS config_json,
                r.samples AS samples,
                r.max_debug_iterations AS max_debug_iterations,
                r.timeout_s AS timeout_s,
                COUNT(*) AS total,
                SUM(tr.passed) AS passed
              FROM runs r
              JOIN task_results tr ON tr.run_id = r.id
              WHERE {' AND '.join(where)}
              GROUP BY r.id
              ORDER BY r.timestamp DESC
            """
            rows = self.conn.execute(sql, params).fetchall()
            out: list[dict] = []
            for row in rows:
                total = int(row["total"])
                passed = int(row["passed"] or 0)
                out.append(
                    {
                        "run_id": int(row["run_id"]),
                        "timestamp": str(row["timestamp"]),
                        "benchmark": str(row["benchmark"]),
                        "backend_name": str(row["backend_name"]),
                        "model_id": str(row["model_id"]),
                        "retrieval": bool(int(row["retrieval"])),
                        "samples": int(row["samples"]),
                        "max_debug_iterations": int(row["max_debug_iterations"]),
                        "timeout_s": int(row["timeout_s"]),
                        "config_json": str(row["config_json"]),
                        "total": total,
                        "passed": passed,
                        "pass_rate": passed / total if total else 0.0,
                    }
                )
            return out

        group_exprs = [group_map[g] for g in group_by]
        base_group_cols = [
            "r.benchmark",
            "r.backend_name",
            "r.model_id",
            "r.max_debug_iterations",
            "r.timeout_s",
            "r.retrieval",
            "r.samples",
        ]
        group_cols = list(dict.fromkeys([*base_group_cols, *group_exprs]))
        sql = f"""
          SELECT
            r.benchmark AS benchmark,
            r.backend_name AS backend_name,
            r.model_id AS model_id,
            r.retrieval AS retrieval,
            r.samples AS samples,
            r.max_debug_iterations AS max_debug_iterations,
            r.timeout_s AS timeout_s,
            COUNT(*) AS total,
            SUM(tr.passed) AS passed
          FROM runs r
          JOIN task_results tr ON tr.run_id = r.id
          WHERE {' AND '.join(where)}
          GROUP BY {', '.join(group_cols)}
          ORDER BY r.benchmark, r.model_id, r.backend_name, r.max_debug_iterations, r.timeout_s, r.samples
        """
        rows = self.conn.execute(sql, params).fetchall()
        out: list[dict] = []
        for row in rows:
            total = int(row["total"])
            passed = int(row["passed"] or 0)
            out.append(
                {
                    "benchmark": str(row["benchmark"]),
                    "backend_name": str(row["backend_name"]),
                    "model_id": str(row["model_id"]),
                    "retrieval": bool(int(row["retrieval"])),
                    "samples": int(row["samples"]),
                    "max_debug_iterations": int(row["max_debug_iterations"]),
                    "timeout_s": int(row["timeout_s"]),
                    "total": total,
                    "passed": passed,
                    "pass_rate": passed / total if total else 0.0,
                }
            )
        return out
