from __future__ import annotations

import sqlite3
from pathlib import Path


def compare_run_dirs(
    *,
    baseline_dir: str,
    candidate_dir: str,
    task_ids: list[str] | None = None,
) -> str:
    report = compare_runs(baseline_dir, candidate_dir, task_ids=task_ids)
    return format_comparison(report)


def compare_runs(
    baseline_dir: str, candidate_dir: str, *, task_ids: list[str] | None = None
) -> dict[str, object]:
    baseline = _load_results_from_dir(baseline_dir, task_ids)
    candidate = _load_results_from_dir(candidate_dir, task_ids)

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
        "baseline_total": len(baseline),
        "candidate_total": len(candidate),
        "baseline_passed": sum(baseline.values()),
        "candidate_passed": sum(candidate.values()),
        "gained": gained,
        "lost": lost,
        "unchanged_pass": unchanged_pass,
        "net_change": len(gained) - len(lost),
    }


def format_comparison(report: dict[str, object]) -> str:
    gained = list(report["gained"])
    lost = list(report["lost"])
    unchanged_pass = list(report["unchanged_pass"])
    lines = [
        f"Baseline: {report['baseline_passed']}/{report['baseline_total']} passed",
        f"Candidate: {report['candidate_passed']}/{report['candidate_total']} passed",
        f"Net change: {report['net_change']:+d}",
    ]
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


def _load_results_from_dir(dir_path: str, task_ids: list[str] | None) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for db_file in sorted(Path(dir_path).glob("*.db")):
        results.update(_load_results(str(db_file), task_ids))
    return results


def _load_results(db_path: str, task_ids: list[str] | None) -> dict[str, bool]:
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

        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            rows = conn.execute(
                f"SELECT task_id, {passed_col} FROM task_results WHERE task_id IN ({placeholders})",
                task_ids,
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT task_id, {passed_col} FROM task_results").fetchall()
        return {row["task_id"]: bool(row[passed_col]) for row in rows}
    finally:
        conn.close()
