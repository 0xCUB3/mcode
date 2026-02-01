#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from mcode.bench.results import ResultsDB


def _read_single_run(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No runs found in shard DB")
    config = json.loads(row["config_json"])
    return {
        "benchmark": row["benchmark"],
        "config": config,
    }


def _iter_task_results(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
          task_id, passed, samples_generated, debug_iterations_used, time_ms,
          exit_code, timed_out, stdout, stderr, error, code_sha256
        FROM task_results
        """
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "task_id": r["task_id"],
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
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge mCode shard SQLite DBs into a single run DB.")
    parser.add_argument("--out", required=True, type=Path, help="Output SQLite DB path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output DB if it already exists",
    )
    parser.add_argument("shards", nargs="+", type=Path, help="Shard SQLite DB paths")
    args = parser.parse_args()

    out_path: Path = args.out
    shard_paths: list[Path] = args.shards

    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing shard DB(s): {', '.join(str(p) for p in missing)}")

    if out_path.exists():
        if not args.force:
            raise SystemExit(f"Output DB already exists: {out_path} (use --force to overwrite)")
        out_path.unlink()

    first = sqlite3.connect(shard_paths[0])
    first.row_factory = sqlite3.Row
    try:
        run = _read_single_run(first)
    finally:
        first.close()

    benchmark = str(run["benchmark"])
    config = dict(run["config"])

    out_db = ResultsDB(out_path)
    run_id = out_db.start_run(benchmark, config)

    seen: set[str] = set()
    total_rows = 0
    for shard_path in shard_paths:
        conn = sqlite3.connect(shard_path)
        conn.row_factory = sqlite3.Row
        try:
            for row in _iter_task_results(conn):
                task_id = str(row["task_id"])
                if task_id in seen:
                    continue
                seen.add(task_id)
                out_db.save_task_result(run_id, row)
                total_rows += 1
        finally:
            conn.close()

    print(f"out={out_path} benchmark={benchmark} run_id={run_id} tasks={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

