from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from mcode.bench.results import export_csv as export_csv_results
from mcode.bench.results import merge_shard_dbs
from mcode.cli_shared import _expand_db_paths, _open_results_view, _parse_task_ids
from mcode.ui.console import console
from mcode.ui.flags import JsonFlag


def register_results_commands(app: typer.Typer) -> None:
    app.command("results")(results)
    app.command("compare")(compare)
    app.command("merge-shards", hidden=True)(merge_shards)
    app.command("export-csv")(export_csv)


def results(
    db: Annotated[
        list[Path] | None,
        typer.Option("--db", help="SQLite DB path (repeatable)"),
    ] = None,
    db_glob: Annotated[
        list[str] | None,
        typer.Option("--db-glob", help="Glob for SQLite DBs (quote to prevent shell expansion)"),
    ] = None,
    db_dir: Annotated[
        list[Path] | None,
        typer.Option("--db-dir", help="Directory to scan recursively for *.db files"),
    ] = None,
    benchmark: Annotated[str | None, typer.Option("--benchmark")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    suite_name: Annotated[str | None, typer.Option("--suite")] = None,
    suite_entry_name: Annotated[str | None, typer.Option("--suite-entry")] = None,
    loop_budget: Annotated[int | None, typer.Option("--loop-budget", min=1)] = None,
    timeout_s: Annotated[int | None, typer.Option("--timeout", min=1)] = None,
    compare_configs: Annotated[
        bool,
        typer.Option("--compare-configs", help="Group results by config"),
    ] = False,
    time_metrics: Annotated[
        bool,
        typer.Option("--time", help="Include time-to-solve metrics (sec/solve, solves/hour, p95)"),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Query pass rates from the results DB."""
    group_by_config = (
        "suite_name",
        "suite_entry_name",
        "backend_name",
        "timeout_s",
        "loop_budget",
    )

    db_paths = _expand_db_paths(db=db, db_glob=db_glob, db_dir=db_dir)
    with _open_results_view(db_paths) as rdb:
        if compare_configs:
            if time_metrics:
                rows = rdb.run_metrics_grouped(
                    benchmark=benchmark,
                    model_id=model,
                    backend_name=backend,
                    timeout_s=timeout_s,
                    suite_name=suite_name,
                    suite_entry_name=suite_entry_name,
                    group_by=group_by_config,
                    loop_budget=loop_budget,
                )
                rows = sorted(
                    rows,
                    key=lambda r: (r["solves_per_hour"], r["pass_rate"]),
                    reverse=True,
                )
                if json_mode:
                    console.print_json(data=rows)
                    return
                table = Table(title="Pass rates + time (grouped)")
                table.add_column("benchmark")
                table.add_column("suite")
                table.add_column("entry")
                table.add_column("backend")
                table.add_column("model")
                table.add_column("budget", justify="right")
                table.add_column("timeout", justify="right")
                table.add_column("runs", justify="right")
                table.add_column("total", justify="right")
                table.add_column("passed", justify="right")
                table.add_column("generated", justify="right")
                table.add_column("evaluated", justify="right")
                table.add_column("pass_rate", justify="right")
                table.add_column("avg_s", justify="right")
                table.add_column("p95_s", justify="right")
                table.add_column("tok/task", justify="right")
                table.add_column("sec/solve", justify="right")
                table.add_column("solves/hr", justify="right")
                for row in rows:
                    table.add_row(
                        row["benchmark"],
                        str(row.get("suite_name") or "-"),
                        str(row.get("suite_entry_name") or "-"),
                        row["backend_name"],
                        row["model_id"],
                        str(row.get("loop_budget", "")),
                        str(row["timeout_s"]),
                        str(row.get("runs", "")),
                        str(row["total"]),
                        str(row["passed"]),
                        str(row.get("artifact_generated_tasks", 0)),
                        str(row.get("artifact_evaluated_tasks", 0)),
                        f"{row['pass_rate']:.1%}",
                        f"{row['time_s_avg']:.2f}",
                        f"{row['time_s_p95']:.2f}" if row.get("time_s_p95") is not None else "-",
                        f"{row['total_tokens_avg']:.1f}"
                        if row.get("total_tokens_avg") is not None
                        else "-",
                        f"{row['sec_per_solve']:.2f}"
                        if row.get("sec_per_solve") is not None
                        else "-",
                        f"{row['solves_per_hour']:.2f}",
                    )
                console.print(table)
                return

            rows = rdb.pass_rates_grouped(
                benchmark=benchmark,
                model_id=model,
                backend_name=backend,
                timeout_s=timeout_s,
                suite_name=suite_name,
                suite_entry_name=suite_entry_name,
                group_by=group_by_config,
                loop_budget=loop_budget,
            )
            if json_mode:
                console.print_json(data=rows)
                return
            table = Table(title="Pass rates by config")
            table.add_column("benchmark")
            table.add_column("suite")
            table.add_column("entry")
            table.add_column("backend")
            table.add_column("model")
            table.add_column("budget", justify="right")
            table.add_column("timeout", justify="right")
            table.add_column("total", justify="right")
            table.add_column("passed", justify="right")
            table.add_column("generated", justify="right")
            table.add_column("evaluated", justify="right")
            table.add_column("pass_rate", justify="right")
            for row in rows:
                table.add_row(
                    row["benchmark"],
                    str(row.get("suite_name") or "-"),
                    str(row.get("suite_entry_name") or "-"),
                    row["backend_name"],
                    row["model_id"],
                    str(row.get("loop_budget", "")),
                    str(row["timeout_s"]),
                    str(row["total"]),
                    str(row["passed"]),
                    str(row.get("artifact_generated_tasks", 0)),
                    str(row.get("artifact_evaluated_tasks", 0)),
                    f"{row['pass_rate']:.1%}",
                )
            console.print(table)
            return

        if time_metrics:
            rows = rdb.run_metrics_grouped(
                benchmark=benchmark,
                model_id=model,
                backend_name=backend,
                timeout_s=timeout_s,
                suite_name=suite_name,
                suite_entry_name=suite_entry_name,
                group_by=(),
                loop_budget=loop_budget,
            )
            rows = sorted(rows, key=lambda r: (r["solves_per_hour"], r["pass_rate"]), reverse=True)
            if json_mode:
                console.print_json(data=rows)
                return
            table = Table(title="Pass rates + time (per run)")
            table.add_column("run_id", justify="right")
            table.add_column("timestamp")
            table.add_column("benchmark")
            table.add_column("suite")
            table.add_column("entry")
            table.add_column("backend")
            table.add_column("model")
            table.add_column("budget", justify="right")
            table.add_column("timeout", justify="right")
            table.add_column("total", justify="right")
            table.add_column("passed", justify="right")
            table.add_column("generated", justify="right")
            table.add_column("evaluated", justify="right")
            table.add_column("pass_rate", justify="right")
            table.add_column("avg_s", justify="right")
            table.add_column("p95_s", justify="right")
            table.add_column("tok/task", justify="right")
            table.add_column("sec/solve", justify="right")
            table.add_column("solves/hr", justify="right")
            for row in rows:
                table.add_row(
                    str(row["run_id"]),
                    row["timestamp"],
                    row["benchmark"],
                    str(row.get("suite_name") or "-"),
                    str(row.get("suite_entry_name") or "-"),
                    row["backend_name"],
                    row["model_id"],
                    str(row.get("loop_budget", "")),
                    str(row["timeout_s"]),
                    str(row["total"]),
                    str(row["passed"]),
                    str(row.get("artifact_generated_tasks", 0)),
                    str(row.get("artifact_evaluated_tasks", 0)),
                    f"{row['pass_rate']:.1%}",
                    f"{row['time_s_avg']:.2f}",
                    f"{row['time_s_p95']:.2f}" if row.get("time_s_p95") is not None else "-",
                    f"{row['total_tokens_avg']:.1f}"
                    if row.get("total_tokens_avg") is not None
                    else "-",
                    f"{row['sec_per_solve']:.2f}" if row.get("sec_per_solve") is not None else "-",
                    f"{row['solves_per_hour']:.2f}",
                )
            console.print(table)
            return

        rows = rdb.pass_rates_grouped(
            benchmark=benchmark,
            model_id=model,
            backend_name=backend,
            timeout_s=timeout_s,
            suite_name=suite_name,
            suite_entry_name=suite_entry_name,
            group_by=(),
            loop_budget=loop_budget,
        )
        if json_mode:
            console.print_json(data=rows)
            return
        table = Table(title="Pass rates (per run)")
        table.add_column("run_id", justify="right")
        table.add_column("timestamp")
        table.add_column("benchmark")
        table.add_column("suite")
        table.add_column("entry")
        table.add_column("backend")
        table.add_column("model")
        table.add_column("budget", justify="right")
        table.add_column("timeout", justify="right")
        table.add_column("total", justify="right")
        table.add_column("passed", justify="right")
        table.add_column("generated", justify="right")
        table.add_column("evaluated", justify="right")
        table.add_column("pass_rate", justify="right")
        for row in rows:
            table.add_row(
                str(row["run_id"]),
                row["timestamp"],
                row["benchmark"],
                str(row.get("suite_name") or "-"),
                str(row.get("suite_entry_name") or "-"),
                row["backend_name"],
                row["model_id"],
                str(row.get("loop_budget", "")),
                str(row["timeout_s"]),
                str(row["total"]),
                str(row["passed"]),
                str(row.get("artifact_generated_tasks", 0)),
                str(row.get("artifact_evaluated_tasks", 0)),
                f"{row['pass_rate']:.1%}",
            )
        console.print(table)


def compare(
    baseline_dir: Annotated[
        Path,
        typer.Option("--baseline-dir", help="Baseline DB file or directory"),
    ],
    candidate_dir: Annotated[
        Path,
        typer.Option("--candidate-dir", help="Candidate DB file or directory"),
    ],
    task_ids: Annotated[
        str | None,
        typer.Option(
            "--task-ids",
            help="Comma-separated task IDs to compare (or path to JSON/text file)",
        ),
    ] = None,
    benchmark: Annotated[str | None, typer.Option("--benchmark")] = None,
    suite_name: Annotated[str | None, typer.Option("--suite")] = None,
    suite_entry_name: Annotated[str | None, typer.Option("--suite-entry")] = None,
    max_lost: Annotated[
        int | None,
        typer.Option("--max-lost", min=0, help="Fail if more than N tasks regress"),
    ] = None,
    min_net: Annotated[
        int | None,
        typer.Option("--min-net", help="Fail if gained-lost is below N"),
    ] = None,
    min_candidate_pass_rate: Annotated[
        float | None,
        typer.Option(
            "--min-candidate-pass-rate",
            min=0.0,
            max=1.0,
            help="Fail if candidate pass rate is below this 0-1 fraction",
        ),
    ] = None,
    min_candidate_passed: Annotated[
        int | None,
        typer.Option("--min-candidate-passed", min=0, help="Fail if candidate passes fewer tasks"),
    ] = None,
    json_mode: JsonFlag = False,
) -> None:
    """Compare baseline and candidate result DBs, with optional gate checks."""
    from mcode.bench.compare import compare_gate_failures, compare_runs, format_comparison

    report = compare_runs(
        baseline_dir=str(baseline_dir),
        candidate_dir=str(candidate_dir),
        task_ids=_parse_task_ids(task_ids),
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )
    failures = compare_gate_failures(
        report,
        max_lost=max_lost,
        min_net=min_net,
        min_candidate_pass_rate=min_candidate_pass_rate,
        min_candidate_passed=min_candidate_passed,
    )
    if json_mode:
        console.print_json(data={**report, "gate_failures": failures})
    else:
        console.print(format_comparison(report))
        if failures:
            console.print("\nGate failed:", style="red")
            for failure in failures:
                console.print(f"  - {failure}", style="red")
    if failures:
        raise typer.Exit(1)


def merge_shards(
    out: Annotated[Path, typer.Option("--out", help="Output SQLite DB path")],
    shards: Annotated[list[Path], typer.Argument(..., help="Shard SQLite DB paths")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite output DB if it exists"),
    ] = False,
) -> None:
    """Merge shard SQLite DBs into a single run DB."""
    report = merge_shard_dbs(out_path=out, shard_paths=shards, force=force)
    console.print(
        f"out={report['out_path']} benchmark={report['benchmark']} run_id={report['run_id']} "
        f"tasks={report['tasks_written']} shards_used={report['shards_used']} "
        f"shards_ignored={report['shards_ignored']}"
    )


def export_csv(
    inputs: Annotated[
        list[Path],
        typer.Option(
            "--input",
            "-i",
            help=(
                "DB file or directory (directories: exports all top-level *.db; "
                "shard DBs excluded)."
            ),
        ),
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path("."),
    prefix: Annotated[
        str, typer.Option("--prefix", help="Output filename prefix (writes <prefix>.runs.csv, etc)")
    ] = "mcode",
    include_logs: Annotated[
        bool,
        typer.Option(
            "--include-logs",
            help="Include stdout/stderr/error columns (can make CSV rows very large).",
        ),
    ] = False,
) -> None:
    """Export one or more results DBs to CSV (runs + task_results)."""
    if not inputs:
        raise typer.BadParameter("Provide at least one --input (DB file or directory).")
    report = export_csv_results(
        inputs=inputs, out_dir=out_dir, prefix=prefix, include_logs=include_logs
    )
    message = (
        f"exported dbs={report['dbs']} runs={report['runs']} "
        f"task_results={report['task_results']} "
        f"diagnostic_events={report.get('diagnostic_events', 0)} "
        f"artifact_tasks={report.get('artifact_tasks', 0)} "
        f"artifact_candidates={report.get('artifact_candidates', 0)} "
        f"artifact_evaluations={report.get('artifact_evaluations', 0)}\n"
        f"runs_csv={report['runs_csv']}\n"
        f"task_results_csv={report['task_results_csv']}\n"
        f"artifact_tasks_csv={report['artifact_tasks_csv']}\n"
        f"artifact_candidates_csv={report['artifact_candidates_csv']}\n"
        f"artifact_evaluations_csv={report['artifact_evaluations_csv']}\n"
        f"artifact_verification_evidence_csv={report['artifact_verification_evidence_csv']}"
    )
    if report.get("diagnostic_events_csv"):
        message += f"\ndiagnostic_events_csv={report['diagnostic_events_csv']}"
    console.print(message)
