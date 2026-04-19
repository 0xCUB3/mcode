from __future__ import annotations


def compare_run_dirs(
    *,
    baseline_dir: str,
    candidate_dir: str,
    task_ids: list[str] | None = None,
) -> str:
    from mellea.eval.compare import compare_runs, format_comparison

    report = compare_runs(
        baseline_dir,
        candidate_dir,
        task_ids=task_ids,
    )
    return format_comparison(report)
