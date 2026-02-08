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
        now = datetime.now(UTC).isoformat()
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
        benchmark: str | None,
        model_id: str | None,
        backend_name: str | None = None,
        max_debug_iterations: int | None = None,
        timeout_s: int | None = None,
        group_by: Sequence[str],
        retrieval: bool | None = None,
        samples: int | None = None,
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
          ORDER BY
            r.benchmark,
            r.model_id,
            r.backend_name,
            r.max_debug_iterations,
            r.timeout_s,
            r.samples
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

    def run_metrics_grouped(
        self,
        *,
        benchmark: str | None,
        model_id: str | None,
        backend_name: str | None = None,
        max_debug_iterations: int | None = None,
        timeout_s: int | None = None,
        group_by: Sequence[str],
        retrieval: bool | None = None,
        samples: int | None = None,
        include_percentiles: bool = True,
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
                r.samples AS samples,
                r.max_debug_iterations AS max_debug_iterations,
                r.timeout_s AS timeout_s,
                COUNT(*) AS total,
                SUM(tr.passed) AS passed,
                SUM(tr.timed_out) AS timed_out,
                SUM(tr.time_ms) AS time_ms_total
              FROM runs r
              JOIN task_results tr ON tr.run_id = r.id
              WHERE {' AND '.join(where)}
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
                        "retrieval": bool(int(row["retrieval"])),
                        "samples": int(row["samples"]),
                        "max_debug_iterations": int(row["max_debug_iterations"]),
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
            COUNT(DISTINCT r.id) AS runs,
            COUNT(*) AS total,
            SUM(tr.passed) AS passed,
            SUM(tr.timed_out) AS timed_out,
            SUM(tr.time_ms) AS time_ms_total
          FROM runs r
          JOIN task_results tr ON tr.run_id = r.id
          WHERE {' AND '.join(where)}
          GROUP BY {', '.join(group_cols)}
          ORDER BY
            r.benchmark,
            r.model_id,
            r.backend_name,
            r.max_debug_iterations,
            r.timeout_s,
            r.samples
        """
        rows = self.conn.execute(sql, params).fetchall()

        time_stats: dict[tuple, dict[str, float | None]] = {}
        if include_percentiles and rows:
            detail_sql = f"""
              SELECT
                r.benchmark AS benchmark,
                r.backend_name AS backend_name,
                r.model_id AS model_id,
                r.max_debug_iterations AS max_debug_iterations,
                r.timeout_s AS timeout_s,
                r.retrieval AS retrieval,
                r.samples AS samples,
                tr.time_ms AS time_ms
              FROM runs r
              JOIN task_results tr ON tr.run_id = r.id
              WHERE {' AND '.join(where)}
            """
            detail_rows = self.conn.execute(detail_sql, params).fetchall()
            times_by_key: dict[tuple, list[int]] = {}
            for dr in detail_rows:
                key = (
                    str(dr["benchmark"]),
                    str(dr["backend_name"]),
                    str(dr["model_id"]),
                    int(dr["max_debug_iterations"]),
                    int(dr["timeout_s"]),
                    bool(int(dr["retrieval"])),
                    int(dr["samples"]),
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
                int(row["max_debug_iterations"]),
                int(row["timeout_s"]),
                bool(int(row["retrieval"])),
                int(row["samples"]),
            )
            p = time_stats.get(key) if include_percentiles else None
            p50_ms = p.get("p50_ms") if p else None
            p95_ms = p.get("p95_ms") if p else None

            out.append(
                {
                    "benchmark": str(row["benchmark"]),
                    "backend_name": str(row["backend_name"]),
                    "model_id": str(row["model_id"]),
                    "retrieval": bool(int(row["retrieval"])),
                    "samples": int(row["samples"]),
                    "max_debug_iterations": int(row["max_debug_iterations"]),
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
                  samples,
                  max_debug_iterations,
                  timeout_s,
                  retrieval,
                  config_json
                FROM runs
                ORDER BY id
                """
            ).fetchall()
            for run in runs:
                cur = self.conn.execute(
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
                        str(run["timestamp"]),
                        str(run["benchmark"]),
                        str(run["backend_name"]),
                        str(run["model_id"]),
                        int(run["samples"]),
                        int(run["max_debug_iterations"]),
                        int(run["timeout_s"]),
                        int(run["retrieval"]),
                        str(run["config_json"]),
                    ),
                )
                new_run_id = int(cur.lastrowid)
                old_run_id = int(run["id"])

                task_rows = src.execute(
                    """
                    SELECT
                      task_id,
                      passed,
                      samples_generated,
                      debug_iterations_used,
                      time_ms,
                      exit_code,
                      timed_out,
                      stdout,
                      stderr,
                      error,
                      code_sha256
                    FROM task_results
                    WHERE run_id = ?
                    ORDER BY id
                    """,
                    (old_run_id,),
                ).fetchall()

                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO task_results
                    (run_id, task_id, passed, samples_generated, debug_iterations_used, time_ms,
                     exit_code, timed_out, stdout, stderr, error, code_sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            new_run_id,
                            str(tr["task_id"]),
                            int(tr["passed"]),
                            int(tr["samples_generated"]),
                            int(tr["debug_iterations_used"]),
                            int(tr["time_ms"]),
                            tr["exit_code"],
                            int(tr["timed_out"]),
                            tr["stdout"],
                            tr["stderr"],
                            tr["error"],
                            tr["code_sha256"],
                        )
                        for tr in task_rows
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
                    count = int(conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0])
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
            rows = conn.execute(
                """
                SELECT
                  task_id, passed, samples_generated, debug_iterations_used, time_ms,
                  exit_code, timed_out, stdout, stderr, error, code_sha256
                FROM task_results
                """
            ).fetchall()
            for r in rows:
                task_id = str(r["task_id"])
                if task_id in seen:
                    continue
                seen.add(task_id)
                out_db.save_task_result(
                    run_id,
                    {
                        "task_id": task_id,
                        "passed": bool(r["passed"]),
                        "samples_generated": int(r["samples_generated"]),
                        "debug_iterations_used": int(r["debug_iterations_used"]),
                        "time_ms": int(r["time_ms"]),
                        "exit_code": r["exit_code"],
                        "timed_out": bool(r["timed_out"]),
                        "stdout": r["stdout"],
                        "stderr": r["stderr"],
                        "error": r["error"],
                        "code_sha256": r["code_sha256"],
                    },
                )
                written += 1
        finally:
            conn.close()

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

    runs_fields = [
        "source_db",
        "run_id",
        "timestamp",
        "benchmark",
        "backend_name",
        "model_id",
        "samples",
        "max_debug_iterations",
        "timeout_s",
        "retrieval",
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
        "samples",
        "max_debug_iterations",
        "timeout_s",
        "retrieval",
        "task_id",
        "passed",
        "samples_generated",
        "debug_iterations_used",
        "time_ms",
        "exit_code",
        "timed_out",
        "code_sha256",
        "config_json",
    ]
    if include_logs:
        task_fields.extend(["stdout", "stderr", "error"])

    run_rows = 0
    task_rows = 0

    with runs_csv.open("w", newline="", encoding="utf-8") as rf, tasks_csv.open(
        "w", newline="", encoding="utf-8"
    ) as tf:
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
                            "samples": int(r["samples"]),
                            "max_debug_iterations": int(r["max_debug_iterations"]),
                            "timeout_s": int(r["timeout_s"]),
                            "retrieval": int(r["retrieval"]),
                            "total": total,
                            "passed": passed,
                            "pass_rate": f"{pass_rate:.6f}",
                            "config_json": config_json,
                        }
                    )
                    run_rows += 1

                    tasks = conn.execute(
                        """
                        SELECT
                          tr.task_id,
                          tr.passed,
                          tr.samples_generated,
                          tr.debug_iterations_used,
                          tr.time_ms,
                          tr.exit_code,
                          tr.timed_out,
                          tr.stdout,
                          tr.stderr,
                          tr.error,
                          tr.code_sha256
                        FROM task_results tr
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
                            "samples": int(r["samples"]),
                            "max_debug_iterations": int(r["max_debug_iterations"]),
                            "timeout_s": int(r["timeout_s"]),
                            "retrieval": int(r["retrieval"]),
                            "task_id": str(tr["task_id"]),
                            "passed": int(tr["passed"]),
                            "samples_generated": int(tr["samples_generated"]),
                            "debug_iterations_used": int(tr["debug_iterations_used"]),
                            "time_ms": int(tr["time_ms"]),
                            "exit_code": tr["exit_code"],
                            "timed_out": int(tr["timed_out"]),
                            "code_sha256": tr["code_sha256"],
                            "config_json": config_json,
                        }
                        if include_logs:
                            row.update(
                                {
                                    "stdout": tr["stdout"],
                                    "stderr": tr["stderr"],
                                    "error": tr["error"],
                                }
                            )
                        tasks_writer.writerow(row)
                        task_rows += 1
            finally:
                conn.close()

    return {
        "dbs": len(db_paths),
        "runs": run_rows,
        "task_results": task_rows,
        "runs_csv": runs_csv,
        "task_results_csv": tasks_csv,
    }
