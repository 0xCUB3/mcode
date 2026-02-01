from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ExportReport:
    dbs: int
    runs: int
    task_results: int
    runs_csv: Path
    task_results_csv: Path


def _iter_db_paths(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.glob("*.db")))
        else:
            out.append(p)
    # Ignore shard DBs (they're intermediate artifacts).
    out = [p for p in out if p.exists() and p.suffix == ".db" and "shard-" not in p.name]
    # Stable ordering for repeatable exports.
    return sorted(set(out))


def export_results_csv(*, inputs: list[Path], out_dir: Path, prefix: str = "mcode") -> ExportReport:
    db_paths = _iter_db_paths(inputs)
    if not db_paths:
        raise FileNotFoundError("No .db files found (pass --input <db|dir> ...).")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / f"{prefix}.runs.csv"
    task_csv = out_dir / f"{prefix}.task_results.csv"

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
        "stdout",
        "stderr",
        "error",
        "code_sha256",
        "config_json",
    ]

    run_rows = 0
    task_rows = 0

    with runs_csv.open("w", newline="", encoding="utf-8") as rf, task_csv.open(
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
                    try:
                        config_obj = json.loads(config_json) if config_json else {}
                        config_json = json.dumps(config_obj, sort_keys=True, default=str)
                    except Exception:
                        pass

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
                        tasks_writer.writerow(
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
                                "task_id": str(tr["task_id"]),
                                "passed": int(tr["passed"]),
                                "samples_generated": int(tr["samples_generated"]),
                                "debug_iterations_used": int(tr["debug_iterations_used"]),
                                "time_ms": int(tr["time_ms"]),
                                "exit_code": tr["exit_code"],
                                "timed_out": int(tr["timed_out"]),
                                "stdout": tr["stdout"],
                                "stderr": tr["stderr"],
                                "error": tr["error"],
                                "code_sha256": tr["code_sha256"],
                                "config_json": config_json,
                            }
                        )
                        task_rows += 1
            finally:
                conn.close()

    # Touch a small marker for humans when exporting from suite directories.
    (out_dir / f"{prefix}.exported_at.txt").write_text(
        datetime.now().isoformat(), encoding="utf-8"
    )

    return ExportReport(
        dbs=len(db_paths),
        runs=run_rows,
        task_results=task_rows,
        runs_csv=runs_csv,
        task_results_csv=task_csv,
    )

