from __future__ import annotations

import sqlite3
from pathlib import Path


def compare_runs(
    baseline_dir: str,
    candidate_dir: str,
    *,
    task_ids: list[str] | None = None,
    benchmark: str | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
) -> dict[str, object]:
    baseline = _load_results_from_dir(
        baseline_dir,
        task_ids,
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )
    candidate = _load_results_from_dir(
        candidate_dir,
        task_ids,
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )
    baseline_artifacts = _load_artifact_summary_from_dir(
        baseline_dir,
        task_ids=task_ids,
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )
    candidate_artifacts = _load_artifact_summary_from_dir(
        candidate_dir,
        task_ids=task_ids,
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )

    all_tasks = sorted(set(baseline) | set(candidate))
    gained: list[str] = []
    lost: list[str] = []
    unchanged_pass: list[str] = []

    for task in all_tasks:
        baseline_passed = baseline.get(task, False)
        candidate_passed = candidate.get(task, False)
        if not baseline_passed and candidate_passed:
            gained.append(task)
        elif baseline_passed and not candidate_passed:
            lost.append(task)
        elif baseline_passed and candidate_passed:
            unchanged_pass.append(task)

    return {
        "benchmark": benchmark,
        "suite_name": suite_name,
        "suite_entry_name": suite_entry_name,
        "baseline_total": len(baseline),
        "candidate_total": len(candidate),
        "baseline_passed": sum(baseline.values()),
        "candidate_passed": sum(candidate.values()),
        "baseline_artifacts": baseline_artifacts,
        "candidate_artifacts": candidate_artifacts,
        "gained": gained,
        "lost": lost,
        "unchanged_pass": unchanged_pass,
        "net_change": len(gained) - len(lost),
    }


def format_comparison(report: dict[str, object]) -> str:
    gained = list(report["gained"])
    lost = list(report["lost"])
    unchanged_pass = list(report["unchanged_pass"])
    baseline_artifacts = dict(report.get("baseline_artifacts") or {})
    candidate_artifacts = dict(report.get("candidate_artifacts") or {})
    lines: list[str] = []
    if report.get("benchmark") or report.get("suite_name") or report.get("suite_entry_name"):
        context_parts: list[str] = []
        if report.get("benchmark"):
            context_parts.append(f"benchmark={report['benchmark']}")
        if report.get("suite_name"):
            context_parts.append(f"suite={report['suite_name']}")
        if report.get("suite_entry_name"):
            context_parts.append(f"entry={report['suite_entry_name']}")
        lines.append("Context: " + " ".join(context_parts))
    lines.extend([
        f"Baseline: {report['baseline_passed']}/{report['baseline_total']} passed",
        f"Candidate: {report['candidate_passed']}/{report['candidate_total']} passed",
        f"Net change: {report['net_change']:+d}",
    ])
    if baseline_artifacts or candidate_artifacts:
        lines.append(
            "Artifacts: "
            f"baseline generated={baseline_artifacts.get('generated_tasks', 0)} "
            f"evaluated={baseline_artifacts.get('evaluated_tasks', 0)} "
            f"candidates={baseline_artifacts.get('candidate_count', 0)} "
            f"selected_verified={baseline_artifacts.get('selected_verified_count', 0)} "
            f"patch_bytes={baseline_artifacts.get('selected_patch_byte_count_total', 0)}"
        )
        lines.append(
            "Artifacts: "
            f"candidate generated={candidate_artifacts.get('generated_tasks', 0)} "
            f"evaluated={candidate_artifacts.get('evaluated_tasks', 0)} "
            f"candidates={candidate_artifacts.get('candidate_count', 0)} "
            f"selected_verified={candidate_artifacts.get('selected_verified_count', 0)} "
            f"patch_bytes={candidate_artifacts.get('selected_patch_byte_count_total', 0)}"
        )
    if gained:
        lines.append(f"\nGained ({len(gained)}):")
        lines.extend(f"  + {task_id}" for task_id in gained)
    if lost:
        lines.append(f"\nLost ({len(lost)}):")
        lines.extend(f"  - {task_id}" for task_id in lost)
    if unchanged_pass:
        lines.append(f"\nStill passing ({len(unchanged_pass)}):")
        lines.extend(f"  = {task_id}" for task_id in unchanged_pass)
    return "\n".join(lines)


def _iter_db_paths(dir_path: str) -> list[Path]:
    root = Path(dir_path)
    return [root] if root.is_file() else sorted(root.glob("*.db"))


def _load_results_from_dir(
    dir_path: str,
    task_ids: list[str] | None,
    *,
    benchmark: str | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for db_file in _iter_db_paths(dir_path):
        results.update(
            _load_results(
                str(db_file),
                task_ids,
                benchmark=benchmark,
                suite_name=suite_name,
                suite_entry_name=suite_entry_name,
            )
        )
    return results


def _empty_artifact_summary() -> dict[str, int]:
    return {
        "generated_tasks": 0,
        "evaluated_tasks": 0,
        "candidate_count": 0,
        "selected_candidate_count": 0,
        "selected_verified_count": 0,
        "selected_patch_byte_count_total": 0,
    }


def _load_artifact_summary_from_dir(
    dir_path: str,
    *,
    task_ids: list[str] | None,
    benchmark: str | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
) -> dict[str, int]:
    summary = _empty_artifact_summary()
    for db_file in _iter_db_paths(dir_path):
        partial = _load_artifact_summary(
            str(db_file),
            task_ids=task_ids,
            benchmark=benchmark,
            suite_name=suite_name,
            suite_entry_name=suite_entry_name,
        )
        for key, value in partial.items():
            summary[key] += value
    return summary


def _load_results(
    db_path: str,
    task_ids: list[str] | None,
    *,
    benchmark: str | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
) -> dict[str, bool]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(task_results)").fetchall()]
        if not columns:
            return {}
        passed_col = (
            "passed" if "passed" in columns else "resolved" if "resolved" in columns else None
        )
        if passed_col is None:
            return {}

        where = ["1=1"]
        params: list[object] = []
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            where.append(f"tr.task_id IN ({placeholders})")
            params.extend(task_ids)
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        join_runs = bool(benchmark or suite_name or suite_entry_name) and {
            "benchmark",
        } <= run_columns
        if benchmark and join_runs:
            where.append("r.benchmark = ?")
            params.append(benchmark)
        if suite_name and join_runs:
            where.append("r.suite_name = ?")
            params.append(suite_name)
        if suite_entry_name and join_runs:
            where.append("r.suite_entry_name = ?")
            params.append(suite_entry_name)
        if benchmark and not join_runs:
            return {}
        if suite_name and not join_runs:
            return {}
        if suite_entry_name and not join_runs:
            return {}
        if join_runs:
            sql = (
                f"SELECT tr.task_id, tr.{passed_col} "
                "FROM task_results tr JOIN runs r ON r.id = tr.run_id "
                f"WHERE {' AND '.join(where)}"
            )
        else:
            sql = (
                f"SELECT tr.task_id, tr.{passed_col} "
                f"FROM task_results tr WHERE {' AND '.join(where)}"
            )
        rows = conn.execute(sql, params).fetchall()
        return {row["task_id"]: bool(row[passed_col]) for row in rows}
    finally:
        conn.close()


def _load_artifact_summary(
    db_path: str,
    *,
    task_ids: list[str] | None,
    benchmark: str | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        empty = _empty_artifact_summary()
        if "artifact_tasks" not in tables:
            return empty

        where = ["1=1"]
        params: list[object] = []
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            where.append(f"at.task_id IN ({placeholders})")
            params.extend(task_ids)
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        join_runs = bool(benchmark or suite_name or suite_entry_name) and {
            "benchmark",
        } <= run_columns
        if benchmark and join_runs:
            where.append("r.benchmark = ?")
            params.append(benchmark)
        if suite_name and join_runs:
            where.append("r.suite_name = ?")
            params.append(suite_name)
        if suite_entry_name and join_runs:
            where.append("r.suite_entry_name = ?")
            params.append(suite_entry_name)
        if benchmark and not join_runs:
            return empty
        if suite_name and not join_runs:
            return empty
        if suite_entry_name and not join_runs:
            return empty

        from_clause = "artifact_tasks at"
        if join_runs:
            from_clause += " JOIN runs r ON r.id = at.run_id"
        sql = f"""
            SELECT
              COUNT(DISTINCT at.task_id) AS generated_tasks,
              COUNT(DISTINCT CASE WHEN at.evaluation_count > 0 THEN at.task_id END)
                AS evaluated_tasks,
              COALESCE(SUM(at.candidate_count), 0) AS candidate_count,
              COALESCE(SUM(CASE WHEN ac.selected THEN 1 ELSE 0 END), 0)
                AS selected_candidate_count,
              COALESCE(
                SUM(
                    CASE
                        WHEN ac.selected AND ac.verification_succeeded THEN 1
                        ELSE 0
                    END
                ),
                0
              ) AS selected_verified_count,
              COALESCE(SUM(CASE WHEN ac.selected THEN ac.patch_byte_count ELSE 0 END), 0)
                AS selected_patch_byte_count_total
            FROM {from_clause}
            LEFT JOIN artifact_candidates ac
              ON ac.run_id = at.run_id AND ac.task_id = at.task_id
            WHERE {' AND '.join(where)}
        """
        row = conn.execute(sql, params).fetchone()
        return {
            "generated_tasks": int(row["generated_tasks"] or 0),
            "evaluated_tasks": int(row["evaluated_tasks"] or 0),
            "candidate_count": int(row["candidate_count"] or 0),
            "selected_candidate_count": int(row["selected_candidate_count"] or 0),
            "selected_verified_count": int(row["selected_verified_count"] or 0),
            "selected_patch_byte_count_total": int(row["selected_patch_byte_count_total"] or 0),
        }
    finally:
        conn.close()
