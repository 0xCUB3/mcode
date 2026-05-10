from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence

_TERMINAL_REASON_BUCKETS = (
    "budget_exhausted",
    "unverified_diff_discarded",
    "wrong_patch_after_verification",
    "infra_failure",
    "submitted",
)


def _run_filter_sql(
    *,
    benchmark: str | None,
    model_id: str | None,
    backend_name: str | None = None,
    timeout_s: int | None = None,
    suite_name: str | None = None,
    suite_entry_name: str | None = None,
    loop_budget: int | None = None,
) -> tuple[str, list[object]]:
    where = ["1=1"]
    params: list[object] = []
    filters = (
        ("r.benchmark = ?", benchmark),
        ("r.model_id = ?", model_id),
        ("r.backend_name = ?", backend_name),
        ("r.timeout_s = ?", int(timeout_s) if timeout_s is not None else None),
        ("r.suite_name = ?", suite_name),
        ("r.suite_entry_name = ?", suite_entry_name),
        ("r.loop_budget = ?", int(loop_budget) if loop_budget is not None else None),
    )
    for clause, value in filters:
        if value is None:
            continue
        where.append(clause)
        params.append(value)
    return " AND ".join(where), params


def pass_rates_grouped(
    conn: sqlite3.Connection,
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

    where_sql, params = _run_filter_sql(
        benchmark=benchmark,
        model_id=model_id,
        backend_name=backend_name,
        timeout_s=timeout_s,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
        loop_budget=loop_budget,
    )

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
          WHERE {where_sql}
            AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
          GROUP BY r.id
          ORDER BY r.timestamp DESC
        """
        rows = conn.execute(sql, params).fetchall()
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
                    "artifact_generated_tasks": int(row["artifact_generated_tasks"] or 0),
                    "artifact_evaluated_tasks": int(row["artifact_evaluated_tasks"] or 0),
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
      WHERE {where_sql}
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
    rows = conn.execute(sql, params).fetchall()
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
                "artifact_generated_tasks": int(row["artifact_generated_tasks"] or 0),
                "artifact_evaluated_tasks": int(row["artifact_evaluated_tasks"] or 0),
                "pass_rate": passed / total if total else 0.0,
            }
        )
    return out


def run_metrics_grouped(
    conn: sqlite3.Connection,
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

    where_sql, params = _run_filter_sql(
        benchmark=benchmark,
        model_id=model_id,
        backend_name=backend_name,
        timeout_s=timeout_s,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
        loop_budget=loop_budget,
    )

    reason_selects = ",\n".join(
        (
            "                SUM(CASE WHEN tr.terminal_reason = "
            f"'{reason}' THEN 1 ELSE 0 END) AS {reason}"
        )
        for reason in _TERMINAL_REASON_BUCKETS
    )
    grouped_reason_selects = ",\n".join(
        (f"            COALESCE(SUM(run_metrics.{reason}), 0) AS {reason}")
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
            "artifact_selected_candidate_count": int(row["artifact_selected_candidate_count"] or 0),
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
          WHERE {where_sql}
            AND (tr.id IS NOT NULL OR a.generated_tasks IS NOT NULL)
          GROUP BY r.id
          ORDER BY r.timestamp DESC
        """
        rows = conn.execute(sql, params).fetchall()
        run_ids = [int(r["run_id"]) for r in rows]
        time_stats: dict[int, dict[str, float | None]] = {}
        if include_percentiles and run_ids:
            q = ",".join("?" for _ in run_ids)
            time_rows = conn.execute(
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
        WHERE {where_sql}
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
    rows = conn.execute(sql, params).fetchall()

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
          WHERE {where_sql}
        """
        detail_rows = conn.execute(detail_sql, params).fetchall()
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


def _row_value(row: sqlite3.Row, key: str, default=None):
    keys = row.keys() if hasattr(row, "keys") else ()
    if key in keys:
        return row[key]
    return default
