from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.table import Table

from mcode.bench.artifacts import read_task_manifest
from mcode.bench.results import (
    ResultsDB,
    RunSummary,
    merge_shard_dbs,
)
from mcode.bench.results import (
    export_csv as export_csv_results,
)
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.bench.suite import SuiteEntry, load_suite_manifest, task_ids_arg
from mcode.ui.console import configure_logging as _configure_logging
from mcode.ui.console import console
from mcode.ui.flags import JsonFlag
from mcode.util import temporary_directory

app = typer.Typer(add_completion=False, no_args_is_help=True)
bench_app = typer.Typer(add_completion=False, no_args_is_help=True)
deps_app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_DB_PATH = Path("experiments/results/results.db")
DEFAULT_ARTIFACT_DIR_NAME = "artifacts"


def _default_artifact_dir(db: Path) -> Path:
    return db.parent / db.stem / DEFAULT_ARTIFACT_DIR_NAME


def _resolve_artifact_dir(db: Path, artifact_dir: Path | None) -> Path:
    if artifact_dir is not None:
        return artifact_dir
    return _default_artifact_dir(db)


def _configure_mellea_logging(verbose: bool) -> None:
    """Back-compat shim. Logic now lives in mcode.ui.console.configure_logging."""
    _configure_logging(verbose=verbose)


def _optional_str(v: str) -> str | None:
    if v.strip().lower() in {"", "none", "null"}:
        return None
    return v


def _validate_shards(
    *, shard_count: int | None, shard_index: int | None
) -> tuple[int | None, int | None]:
    if shard_index is not None and shard_count is None:
        raise typer.BadParameter("--shard-index requires --shard-count")
    if shard_count is not None and shard_index is not None and shard_index >= shard_count:
        raise typer.BadParameter("--shard-index must be < --shard-count")
    return shard_count, shard_index


def _validate_shard_options(
    *,
    shards: int | None,
    shard_count: int | None,
    shard_index: int | None,
) -> tuple[int | None, int | None, int | None]:
    if shards is not None and (shard_count is not None or shard_index is not None):
        raise typer.BadParameter("--shards cannot be combined with --shard-count/--shard-index")
    shard_count, shard_index = _validate_shards(shard_count=shard_count, shard_index=shard_index)
    return shards, shard_count, shard_index


def _append_option(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _parse_task_ids(raw: str | None) -> list[str] | None:
    """Parse --task-ids: comma-separated string or path to JSON/text file."""
    if not raw:
        return None
    try:
        p = Path(raw)
        exists = p.exists()
    except OSError:
        exists = False
    if exists:
        text = p.read_text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "tasks" in data:
            ids: list[str] = []
            for v in data["tasks"].values():
                if isinstance(v, list):
                    ids.extend(v)
            return ids
        raise typer.BadParameter(f"Cannot parse task IDs from {raw}")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _validate_sampling(
    *,
    sampling: str,
    sampling_budget: int | None,
) -> tuple[str, int | None]:
    if sampling == "none" and sampling_budget is not None:
        raise typer.BadParameter("--sampling-budget requires --sampling != none")
    return sampling, sampling_budget


@contextmanager
def _open_results_view(db_paths: tuple[Path, ...] | list[Path]):
    if not db_paths:
        db_paths = [DEFAULT_DB_PATH]

    resolved: list[Path] = []
    for p in db_paths:
        if not p.exists():
            raise typer.BadParameter(f"SQLite DB not found: {p}")
        resolved.append(p.resolve())

    if len(resolved) == 1:
        rdb = ResultsDB(resolved[0])
        try:
            yield rdb
        finally:
            rdb.close()
        return

    with temporary_directory(prefix="mcode-results-") as td:
        merged_path = Path(td) / "merged.db"
        rdb = ResultsDB(merged_path)
        try:
            rdb.merge_from(resolved)
            yield rdb
        finally:
            rdb.close()


def _expand_db_paths(
    *,
    db: list[Path] | None,
    db_glob: list[str] | None,
    db_dir: list[Path] | None,
) -> list[Path]:
    paths: list[Path] = []

    for p in db or []:
        paths.append(p)

    for d in db_dir or []:
        if not d.exists() or not d.is_dir():
            raise typer.BadParameter(f"--db-dir must be a directory: {d}")
        paths.extend(sorted(d.rglob("*.db")))

    for pattern in db_glob or []:
        matches = glob(pattern, recursive=True)
        if not matches:
            raise typer.BadParameter(f"--db-glob matched no files: {pattern}")
        paths.extend([Path(m) for m in matches])

    if not paths:
        paths = [DEFAULT_DB_PATH]

    # De-dupe while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _version_callback(value: bool) -> None:
    if value:
        try:
            from importlib.metadata import PackageNotFoundError, version

            v = version("mcode")
        except (ImportError, PackageNotFoundError):
            v = "unknown"
        print(f"mcode {v}")
        raise typer.Exit(0)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show Mellea INFO logs")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print mcode version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """mCode benchmarking harness."""
    _configure_mellea_logging(verbose)


@deps_app.command("sync")
def deps_sync(
    extra: Annotated[
        list[str] | None,
        typer.Option("--extra", help="Optional dependency extra to install (repeatable)."),
    ] = None,
    no_dev: Annotated[
        bool,
        typer.Option("--no-dev", help="Do not install the default dev extra."),
    ] = False,
) -> None:
    """Sync uv dependencies, using MCODE_MELLEA_PATH when you want a local mellea checkout."""
    from mcode.uv_setup import sync_uv_environment

    extras = list(extra or [])
    if not no_dev:
        extras.insert(0, "dev")
    sync_args: list[str] = []
    for name in extras:
        sync_args.extend(["--extra", name])

    selection = sync_uv_environment(Path.cwd(), sync_args=sync_args)
    if selection.source == "local":
        console.print(f"Using local mellea override at {selection.local_path}")
    else:
        console.print("Using upstream mellea package")


@app.command("doctor")
def doctor_cmd(
    target: str = typer.Argument(
        None,
        help="optional: bluevela | local-vllm | local-ollama. Omit for system-wide checks.",
    ),
    deep: bool = typer.Option(False, "--deep"),
    init: bool = typer.Option(False, "--init", help="bootstrap launch.toml (bluevela only)"),
    login: str | None = typer.Option(None, "--login", help="user@host for --init"),
) -> None:
    """System + launch diagnostics. Subsumes `mcode launch doctor`."""
    from mcode.doctor import render_check_lines, system_checks
    from mcode.launch import bluevela, local_ollama, local_vllm
    from mcode.launch import config as config_mod
    from mcode.launch.cli import _run as _launch_run
    from mcode.launch.models import Check as _Check
    from mcode.ui.errors import MCodeError, print_error

    if init:
        if target != "bluevela":
            print_error(
                MCodeError(
                    what="--init is only supported for `bluevela`",
                    why=f"target was {target!r}",
                    next="local targets don't need probing — edit launch.toml by hand",
                )
            )
            raise typer.Exit(1)
        if not login:
            login = typer.prompt("Blue Vela login (user@host)")
        written = _launch_run(lambda: bluevela.doctor_init(login=login))
        print(f"wrote {written}")
        print(f"review with `cat {written}` and re-run `mcode doctor bluevela`")
        return

    checks: list[_Check] = []
    if target is None:
        checks.extend(system_checks())
        try:
            cfg = config_mod.load()
            checks.extend(bluevela.doctor(cfg))
            checks.extend(local_vllm.doctor(cfg))
            checks.extend(local_ollama.doctor(cfg))
        except Exception as e:
            checks.append(
                _Check(
                    name="launch config",
                    ok=False,
                    detail=str(e),
                    next="fix or recreate launch.toml; run `mcode doctor bluevela --init`",
                )
            )
    else:
        # Validate target BEFORE loading config so an unknown target produces
        # a clean error instead of surfacing an unrelated TOML parse failure.
        if target not in ("bluevela", "local-vllm", "local-ollama"):
            print_error(
                MCodeError(
                    what=f"unknown target {target!r}",
                    why="valid: bluevela, local-vllm, local-ollama",
                    next="pick one or omit for system-wide checks",
                )
            )
            raise typer.Exit(1)
        cfg = _launch_run(config_mod.load)
        if target == "bluevela":
            checks = bluevela.doctor(cfg)
        elif target == "local-vllm":
            checks = local_vllm.doctor(cfg)
        else:
            checks = local_ollama.doctor(cfg)

    lines, any_failed = render_check_lines(checks)
    for line in lines:
        print(line)
    if any_failed:
        raise typer.Exit(1)


@app.command("watch")
def watch_cmd() -> None:
    """Live dashboard combining `mcode launch status` + `mcode bench list`.

    Refreshes every 2s. Quits cleanly on Ctrl+C. Recovers automatically from
    transient state-file read failures (partial writes, lock contention)."""
    from mcode.watch import watch

    raise typer.Exit(watch())


@app.command("results")
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



@app.command("compare")
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
    json_mode: JsonFlag = False,
 ) -> None:
    from mcode.bench.compare import compare_runs, format_comparison

    report = compare_runs(
        baseline_dir=str(baseline_dir),
        candidate_dir=str(candidate_dir),
        task_ids=_parse_task_ids(task_ids),
        benchmark=benchmark,
        suite_name=suite_name,
        suite_entry_name=suite_entry_name,
    )
    if json_mode:
        console.print_json(data=report)
        return
    console.print(format_comparison(report))


def _config_label(r: dict) -> str:
    parts = [
        str(r.get("benchmark", "")),
        f"{r.get('backend_name', '')}:{r.get('model_id', '')}",
        f"budget={r.get('loop_budget', '')}",
        f"timeout={r.get('timeout_s', '')}",
    ]
    if r.get("strategy") and r["strategy"] != "none":
        parts.append(f"strategy={r['strategy']}")
    if "runs" in r:
        parts.append(f"runs={r.get('runs')}")
    return " | ".join(p for p in parts if p and p != " | ")


def _render_report_html(rows: list[dict], *, title: str) -> str:
    # Keep the report dependency-free: load Plotly from a CDN.
    data_json = json.dumps(rows, sort_keys=True)
    title_json = json.dumps(title)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
      body {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica,
          Arial, sans-serif;
        margin: 24px;
        background: #fff;
        color: #111827;
      }}
      .container {{
        max-width: 1200px;
        margin: 0 auto;
      }}
      #title {{
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 6px;
      }}
      #subtitle {{
        margin: 0 0 14px;
        color: #4b5563;
        font-size: 13px;
        line-height: 1.35;
      }}
      .controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px 12px;
        align-items: center;
        margin: 10px 0 14px;
        font-size: 13px;
        color: #374151;
      }}
        .controls label {{
          display: inline-flex;
          gap: 6px;
          align-items: center;
        }}
        details.dd {{
          position: relative;
          display: inline-block;
        }}
        summary.dd-btn {{
          font-size: 13px;
          padding: 4px 10px;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          background: #fff;
          color: #111827;
          cursor: pointer;
          list-style: none;
        }}
        summary.dd-btn::-webkit-details-marker {{
          display: none;
        }}
        summary.dd-btn::marker {{
          content: "";
        }}
        summary.dd-btn:hover {{
          background: #f9fafb;
        }}
        details.dd > .dd-menu {{
          display: none;
        }}
        details.dd[open] > .dd-menu {{
          display: block;
        }}
        .dd-menu {{
          position: absolute;
          top: calc(100% + 6px);
          left: 0;
          z-index: 1000;
          min-width: 220px;
          max-width: 320px;
          background: #fff;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 8px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }}
        .dd-actions {{
          display: flex;
          gap: 8px;
          margin-bottom: 6px;
        }}
        .dd-action {{
          font-size: 12px;
          padding: 2px 8px;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          background: #f9fafb;
          color: #111827;
          cursor: pointer;
        }}
        .dd-action:hover {{
          background: #f3f4f6;
        }}
        .dd-items {{
          max-height: 240px;
          overflow: auto;
        }}
        label.dd-item {{
          display: flex;
          gap: 8px;
          align-items: center;
          width: 100%;
          box-sizing: border-box;
          padding: 4px 6px;
          border-radius: 8px;
          cursor: pointer;
          user-select: none;
        }}
        label.dd-item:hover {{
          background: #f3f4f6;
        }}
        .dd-item input[type="checkbox"] {{
          margin: 0;
        }}
      select {{
        font-size: 13px;
        padding: 4px 8px;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: #fff;
        color: #111827;
      }}
      input[type="checkbox"] {{
        width: 14px;
        height: 14px;
      }}
      .plot {{
        width: 100%;
        height: 560px;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1 id="title"></h1>
      <p id="subtitle"></p>
      <div class="controls" id="controls"></div>
      <div id="scatter" class="plot"></div>
      <div class="controls" id="summary_controls"></div>
      <div id="summary" class="plot" style="height:420px"></div>
    </div>
    <script>
      const BASE_TITLE = {title_json};
      const ROWS = {data_json};

      function showError(msg) {{
        const p = document.createElement("p");
        p.style.margin = "10px 0 0";
        p.style.color = "#b91c1c";
        p.style.fontSize = "13px";
        p.textContent = "Report error: " + msg;
        const anchor = document.getElementById("scatter");
        if (anchor && anchor.parentElement) {{
          anchor.parentElement.insertBefore(p, anchor);
        }} else {{
          document.body.appendChild(p);
        }}
      }}

      window.addEventListener("error", (e) => {{
        if (e && e.message) showError(e.message);
      }});
      window.addEventListener("unhandledrejection", (e) => {{
        const reason = (e && e.reason) ? String(e.reason) : "unhandled promise rejection";
        showError(reason);
      }});

      if (typeof Plotly === "undefined") {{
        const plotlyLoadMsg =
          "Plotly failed to load. If you're offline or the CDN is blocked, " +
          "the graphs won't render.";
        showError(plotlyLoadMsg);
      }}

      const CONTROLS = document.getElementById("controls");
      const SUMMARY_CONTROLS = document.getElementById("summary_controls");

        function uniqVals(rs, field) {{
          const s = new Set();
          for (const r of rs) {{
            const v = r[field];
            if (v === undefined || v === null) continue;
            s.add(JSON.stringify(v));
          }}
          return Array.from(s).map(x => JSON.parse(x));
        }}

        function constantValue(rs, field) {{
          const u = uniqVals(rs, field);
          return (u.length === 1) ? u[0] : null;
        }}

      function fmtBool(v) {{
        return v ? "on" : "off";
      }}

      function fmtTimeout(v) {{
        return `${{v}}s`;
      }}

      function valueKey(v) {{
        return JSON.stringify(v);
      }}

      const points = ROWS.filter(r => r.sec_per_solve !== null && r.sec_per_solve !== undefined);
      const dropped = ROWS.length - points.length;

      // Promote constants into the title, and omit them from per-point labels.
      let finalTitle = BASE_TITLE;
      const fixed = {{
        benchmark: constantValue(points, "benchmark"),
        backend: constantValue(points, "backend_name"),
        model: constantValue(points, "model_id"),
      }};
      function maybeAppendTitle(k, v) {{
        if (v === null || v === undefined) return;
        const needle = `${{k}}=`;
        if (finalTitle.includes(needle)) return;
        finalTitle += ` | ${{k}}=${{v}}`;
      }}
      maybeAppendTitle("benchmark", fixed.benchmark);
      maybeAppendTitle("backend", fixed.backend);
      maybeAppendTitle("model", fixed.model);

      document.getElementById("title").textContent = finalTitle;
      const fixedTokens = [];
      const fixedBudget = constantValue(points, "loop_budget");
      const fixedTimeout = constantValue(points, "timeout_s");
      if (fixedBudget !== null && fixedBudget !== undefined) {{
        fixedTokens.push(`budget=${{fixedBudget}}`);
      }}
      if (fixedTimeout !== null && fixedTimeout !== undefined) {{
        fixedTokens.push(`t=${{fixedTimeout}}s`);
      }}

      const droppedText = dropped ? ` (hidden: ${{dropped}} with 0 solves)` : "";
      let subtitle = `Plotting ${{points.length}} configs` + droppedText;
      subtitle += ". X = seconds/solve (lower is better). Y = pass rate (higher is better).";
      if (fixedTokens.length) subtitle += ` Fixed: ${{fixedTokens.join(" ")}}.`;
      document.getElementById("subtitle").textContent = subtitle;

      const CONFIG_FIELDS = [
        ["benchmark", "Benchmark"],
        ["suite_name", "Suite"],
        ["suite_entry_name", "Entry"],
        ["backend_name", "Backend"],
        ["model_id", "Model"],
        ["loop_budget", "Budget"],
        ["timeout_s", "Timeout"],
      ];

      const varying = new Map();
      for (const [f] of CONFIG_FIELDS) {{
        varying.set(f, uniqVals(points, f).length > 1);
      }}

      function fmtValue(field, v) {{
        if (field === "timeout_s") return fmtTimeout(v);
        return String(v);
      }}

      function shortToken(field, v) {{
        if (field === "loop_budget") return `budget=${{v}}`;
        if (field === "timeout_s") return `t=${{v}}s`;
        if (field === "benchmark") return String(v);
        if (field === "backend_name") return `backend=${{v}}`;
        if (field === "model_id") return `model=${{v}}`;
        return `${{field}}=${{v}}`;
      }}

      function label(r) {{
        const parts = [];
        if (r.run_id !== undefined && r.run_id !== null) parts.push(`run=${{r.run_id}}`);
        for (const [f] of CONFIG_FIELDS) {{
          if (!varying.get(f)) continue;
          const v = r[f];
          if (v === undefined || v === null) continue;
          parts.push(shortToken(f, v));
        }}
        if (r.runs !== undefined && r.runs !== null && r.runs > 0) parts.push(`runs=${{r.runs}}`);
        return parts.join(" ");
      }}

        function paretoFrontier(rs) {{
          // Maximize pass_rate, minimize sec_per_solve.
          const pts = rs
            .filter(r => r.sec_per_solve !== null && r.sec_per_solve !== undefined)
            .filter(r => r.pass_rate !== null && r.pass_rate !== undefined)
            .map(r => ({{ r, x: Number(r.sec_per_solve), y: Number(r.pass_rate) }}))
            .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y))
            // Sort by x asc, then y desc. Keep strictly improving y as x increases.
            .sort((a, b) => (a.x - b.x) || (b.y - a.y));

          const out = [];
          let bestY = -Infinity;
          for (const p of pts) {{
            if (p.y > bestY) {{
              out.push(p);
              bestY = p.y;
            }}
          }}
          return out;
        }}

      const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2", "#6b7280"];
      const COLOR_PRIORITY = [
        "benchmark",
        "loop_budget",
        "timeout_s",
        "model_id",
        "backend_name",
      ];

        function buildSelect(id, labelText, options, initial) {{
          const wrap = document.createElement("label");
          wrap.htmlFor = id;
          wrap.textContent = labelText;
          const sel = document.createElement("select");
        sel.id = id;
        for (const [value, text] of options) {{
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = text;
          if (value === initial) opt.selected = true;
          sel.appendChild(opt);
        }}
          wrap.appendChild(sel);
          return sel;
        }}

        function buildCheckbox(id, labelText, initial) {{
          const wrap = document.createElement("label");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.id = id;
          cb.checked = !!initial;
          wrap.appendChild(cb);
          const span = document.createElement("span");
          span.textContent = labelText;
          wrap.appendChild(span);
          return cb;
        }}

        function buildMultiSelectDropdown(field, labelText, values) {{
          const wrap = document.createElement("details");
          wrap.className = "dd";

          const summary = document.createElement("summary");
          summary.className = "dd-btn";

          const menu = document.createElement("div");
          menu.className = "dd-menu";

          const actions = document.createElement("div");
          actions.className = "dd-actions";
          const allBtn = document.createElement("button");
          allBtn.type = "button";
          allBtn.className = "dd-action";
          allBtn.textContent = "All";
          const noneBtn = document.createElement("button");
          noneBtn.type = "button";
          noneBtn.className = "dd-action";
          noneBtn.textContent = "None";
          actions.appendChild(allBtn);
          actions.appendChild(noneBtn);

          const items = document.createElement("div");
          items.className = "dd-items";

          const checkboxes = [];
          for (const v of values) {{
            const lab = document.createElement("label");
            lab.className = "dd-item";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = true;
            cb.dataset.field = field;
            cb.dataset.key = valueKey(v);
            lab.appendChild(cb);
            const span = document.createElement("span");
            span.textContent = fmtValue(field, v);
            lab.appendChild(span);
            items.appendChild(lab);
            checkboxes.push(cb);
          }}

          function update() {{
            const selected = checkboxes
              .filter(cb => cb.checked)
              .map(cb => JSON.parse(cb.dataset.key));
            let summaryText = "all";
            if (selected.length === 0) summaryText = "none";
            else if (selected.length !== checkboxes.length) {{
              const texts = selected.map(v => fmtValue(field, v));
              summaryText = (texts.length <= 3)
                ? texts.join(", ")
                : `${{texts.length}}/${{checkboxes.length}}`;
            }}
            summary.textContent = `${{labelText}}: ${{truncate(summaryText, 28)}} ▾`;
          }}

          update();

          allBtn.addEventListener("click", (e) => {{
            e.preventDefault();
            for (const cb of checkboxes) cb.checked = true;
            update();
            render();
          }});
          noneBtn.addEventListener("click", (e) => {{
            e.preventDefault();
            for (const cb of checkboxes) cb.checked = false;
            update();
            render();
          }});

          wrap.appendChild(summary);
          menu.appendChild(actions);
          menu.appendChild(items);
          wrap.appendChild(menu);

          return {{ wrap, checkboxes, update }};
        }}

      function varyingFields() {{
        const out = [];
        for (const [f] of CONFIG_FIELDS) {{
          if (varying.get(f)) out.push(f);
        }}
        return out;
      }}

        const vfields = varyingFields();

        // Filters: allow narrowing by any varying config field, while staying compact.
        const filterCheckboxes = new Map();
        const filterUpdaters = new Map();

        function sortedUnique(field) {{
          const u = uniqVals(points, field);
          return u.slice().sort((a, b) => {{
            const na = Number(a), nb = Number(b);
            if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
            return String(a).localeCompare(String(b));
          }});
        }}

        function currentFilters() {{
          const out = new Map();
          for (const [f, cbs] of filterCheckboxes.entries()) {{
            const selected = cbs.filter(cb => cb.checked).map(cb => cb.dataset.key);
            if (selected.length === cbs.length) continue; // no-op filter
            out.set(f, new Set(selected));
          }}
          return out;
        }}

      function applyFilters(rs, filters) {{
        if (!filters.size) return rs;
        return rs.filter(r => {{
          for (const [f, set] of filters.entries()) {{
            if (!set.has(valueKey(r[f]))) return false;
          }}
          return true;
        }});
      }}

      function colorableFields() {{
        const out = [];
        for (const f of vfields) {{
          // Keep the legend readable.
          if (uniqVals(points, f).length <= 12) out.push(f);
        }}
        return out;
      }}

      const cfields = colorableFields();

      let defaultColorBy = "none";
      for (const f of COLOR_PRIORITY) {{
        if (cfields.includes(f)) {{ defaultColorBy = f; break; }}
      }}

      // Controls: keep it minimal.
      const colorOptions = [["none", "none"]];
        for (const f of cfields) {{
          const label = CONFIG_FIELDS.find(x => x[0] === f)?.[1] ?? f;
          colorOptions.push([f, label]);
        }}
        const colorBySel = buildSelect("color_by", "Color:", colorOptions, defaultColorBy);
        const paretoCb = buildCheckbox("show_pareto", "Best tradeoffs (Pareto)", true);
        const onlyFrontierCb = buildCheckbox("only_frontier", "Only best tradeoffs", false);
        paretoCb.parentElement.title =
          "Best tradeoffs = configs where no other config is both faster and more accurate.";
        onlyFrontierCb.parentElement.title =
          "Hide configs that are dominated on both accuracy and speed.";

        CONTROLS.appendChild(colorBySel.parentElement);
        CONTROLS.appendChild(paretoCb.parentElement);
        CONTROLS.appendChild(onlyFrontierCb.parentElement);

        // Summary controls (simple + focused on accuracy/speed).
        const summarySetSel = buildSelect(
          "summary_set",
          "Summary:",
          [["frontier", "best tradeoffs"], ["shown", "all shown"]],
          "frontier",
        );
        const speedMetricSel = buildSelect(
          "speed_metric",
          "Speed:",
          [
            ["sec_per_solve", "sec/solve"],
            ["time_s_avg", "avg_s"],
            ["time_s_p95", "p95_s"],
          ],
          "sec_per_solve",
        );
        speedMetricSel.parentElement.title =
          "sec/solve = total seconds / passed tasks. avg_s = mean seconds per task. " +
          "p95_s = 95th percentile seconds per task.";
        const summaryViewSel = buildSelect(
          "summary_view",
          "View:",
          [["split", "split"], ["overlay", "overlay"]],
          "split",
        );
        const defaultTop = (points.length <= 20) ? "all" : "20";
        const topNSel = buildSelect(
          "top_n",
          "Top:",
          [["5", "5"], ["10", "10"], ["15", "15"], ["20", "20"], ["30", "30"], ["all", "all"]],
          defaultTop,
        );

        SUMMARY_CONTROLS.appendChild(summarySetSel.parentElement);
        SUMMARY_CONTROLS.appendChild(speedMetricSel.parentElement);
        SUMMARY_CONTROLS.appendChild(summaryViewSel.parentElement);
        SUMMARY_CONTROLS.appendChild(topNSel.parentElement);

        for (const f of vfields) {{
          const label = CONFIG_FIELDS.find(x => x[0] === f)?.[1] ?? f;
          const u = sortedUnique(f);
          const dd = buildMultiSelectDropdown(f, label, u);
          filterCheckboxes.set(f, dd.checkboxes);
          filterUpdaters.set(f, dd.update);
          SUMMARY_CONTROLS.appendChild(dd.wrap);
        }}

      function makeTraces(rs, colorBy) {{
        const baseCustom = r => [
          r.total,
          r.passed,
          (r.timed_out ?? 0),
          (r.timeout_rate ?? ((r.total && r.total > 0) ? ((r.timed_out ?? 0) / r.total) : 0)),
          r.time_s_avg,
          r.time_s_p50,
          r.time_s_p95,
          (r.zero_edit ?? 0),
          (r.zero_verification ?? 0),
          (r.wrong_patch_after_verification ?? 0),
          (r.invalid_tool_call_count ?? 0),
          (r.malformed_tool_call_recoveries ?? 0),
          (r.blocked_finalizer_count ?? 0),
          (r.repeated_failed_run_test_count ?? 0),
          (r.post_edit_exploration_count ?? 0),
          r.turns_after_first_edit_before_first_verification_avg,
        ];
        const hover =
          "%{{text}}" +
          "<br>pass_rate=%{{y:.1%}}" +
          "<br>sec/solve=%{{x:.2f}}" +
          "<br>avg_s=%{{customdata[4]:.2f}}" +
          "<br>p50_s=%{{customdata[5]:.2f}}" +
          "<br>p95_s=%{{customdata[6]:.2f}}" +
          "<br>passed=%{{customdata[1]}}/%{{customdata[0]}}" +
          "<br>timed_out=%{{customdata[2]}}/%{{customdata[0]}} (%{{customdata[3]:.1%}})" +
          "<br>zero_edit=%{{customdata[7]}} zero_verification=%{{customdata[8]}}" +
          "<br>wrong_patch_after_verification=%{{customdata[9]}}" +
          "<br>invalid_tool_calls=%{{customdata[10]}} recovered_tool_args=%{{customdata[11]}}" +
          (
            "<br>blocked_finalizers=%{{customdata[12]}} " +
            "repeated_failed_run_tests=%{{customdata[13]}}"
          ) +
          "<br>post_edit_exploration=%{{customdata[14]}} edit_gap_avg=%{{customdata[15]:.2f}}" +
          "<extra></extra>";

        if (!colorBy || colorBy === "none") {{
          return [{{
            type: "scatter",
            mode: "markers",
            name: "configs",
            x: rs.map(r => r.sec_per_solve),
            y: rs.map(r => r.pass_rate),
            text: rs.map(r => label(r)),
            customdata: rs.map(baseCustom),
            hovertemplate: hover,
            marker: {{
              size: 10,
              opacity: 0.9,
              color: "#2563eb",
              line: {{ width: 1, color: "rgba(0,0,0,0.18)" }},
            }},
          }}];
        }}

        const u = uniqVals(rs, colorBy);
        const sorted = u.slice().sort((a, b) => {{
          const na = Number(a), nb = Number(b);
          if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
          return String(a).localeCompare(String(b));
        }});
        const traces = [];
        for (let i = 0; i < sorted.length; i++) {{
          const v = sorted[i];
          const sub = rs.filter(r => r[colorBy] === v);
          const colorLabel = CONFIG_FIELDS.find(x => x[0] === colorBy)?.[1] ?? colorBy;
          traces.push({{
            type: "scatter",
            mode: "markers",
            name: `${{colorLabel}}=${{fmtValue(colorBy, v)}}`,
            x: sub.map(r => r.sec_per_solve),
            y: sub.map(r => r.pass_rate),
            text: sub.map(r => label(r)),
            customdata: sub.map(baseCustom),
            hovertemplate: hover,
            marker: {{
              size: 10,
              opacity: 0.9,
              color: PALETTE[i % PALETTE.length],
              line: {{ width: 1, color: "rgba(0,0,0,0.18)" }},
            }},
          }});
        }}
        return traces;
      }}

      function truncate(s, maxLen) {{
        const text = String(s ?? "");
        if (text.length <= maxLen) return text;
        return text.slice(0, Math.max(0, maxLen - 1)) + "…";
      }}

      function axisLabel(r) {{
        const parts = [];
        if (r.run_id !== undefined && r.run_id !== null) parts.push(`run=${{r.run_id}}`);
        for (const [f] of CONFIG_FIELDS) {{
          if (!varying.get(f)) continue;
          const v = r[f];
          if (v === undefined || v === null) continue;
          parts.push(shortToken(f, v));
        }}
        return parts.join(" ");
      }}

      function summaryTitle(setName) {{
        const context = [];
        if (fixed.benchmark !== null && fixed.benchmark !== undefined) {{
          context.push(String(fixed.benchmark));
        }}
        if (fixed.model !== null && fixed.model !== undefined) {{
          context.push(String(fixed.model));
        }}
        const contextText = context.length ? ` (${{context.join(" | ")}})` : "";
        return `Summary${{contextText}} — ${{setName}}`;
      }}

        function renderSummary(rs, setName, speedField, viewMode) {{
          const rows = rs
            .filter(r => r.pass_rate !== null && r.pass_rate !== undefined)
            .filter(r => r[speedField] !== null && r[speedField] !== undefined)
            .slice();

        const speedTitle =
          (speedMetricSel.options && speedMetricSel.selectedIndex >= 0)
            ? speedMetricSel.options[speedMetricSel.selectedIndex].textContent
            : speedField;

        if (rows.length === 0) {{
          Plotly.react("summary", [], {{
            title: summaryTitle(setName),
            template: "plotly_white",
          }}, {{ displaylogo: false, responsive: true }});
          return;
        }}

        // Lower is better for speed metrics.
        rows.sort((a, b) => Number(a[speedField]) - Number(b[speedField]));

        const ids = rows.map((_, i) => String(i + 1));
        const ticks = rows.map(r => truncate(axisLabel(r) || "config", 64));
        const pass = rows.map(r => Number(r.pass_rate));
        const speed = rows.map(r => Number(r[speedField]));
        const details = rows.map(r => label(r));
        const totals = rows.map(r => Number(r.total ?? 0));
        const timedOut = rows.map(r => Number(r.timed_out ?? 0));
        const timeoutRates = rows.map((r, i) => {{
          const fromRow = r.timeout_rate;
          if (fromRow !== undefined && fromRow !== null) return Number(fromRow);
          return totals[i] > 0 ? timedOut[i] / totals[i] : 0;
        }});
        const summaryCustom = details.map((d, i) => [d, totals[i], timedOut[i], timeoutRates[i]]);

          const height = Math.max(260, Math.min(1400, 140 + rows.length * 28));
          const summaryDiv = document.getElementById("summary");
          if (summaryDiv) summaryDiv.style.height = height + "px";

          const overlay = (viewMode === "overlay");
          const passTrace = {{
            type: "bar",
            orientation: "h",
            name: "pass_rate",
            x: pass,
            y: ids,
            xaxis: "x",
            marker: {{ color: overlay ? "rgba(37,99,235,0.70)" : "rgba(37,99,235,0.75)" }},
            width: overlay ? 0.70 : undefined,
            customdata: summaryCustom,
            hovertemplate:
              "%{{customdata[0]}}" +
              "<br>pass_rate=%{{x:.1%}}" +
              "<br>timed_out=%{{customdata[2]}}/%{{customdata[1]}} (%{{customdata[3]:.1%}})" +
              "<extra></extra>",
          }};
          const speedTrace = {{
            type: "bar",
            orientation: "h",
            name: speedTitle,
            x: speed,
            y: ids,
            xaxis: "x2",
            marker: {{ color: overlay ? "rgba(17,24,39,0.28)" : "rgba(17,24,39,0.20)" }},
            width: overlay ? 0.34 : undefined,
            customdata: summaryCustom,
            hovertemplate:
              "%{{customdata[0]}}" +
              "<br>" + speedTitle + "=%{{x:.2f}}" +
              "<br>timed_out=%{{customdata[2]}}/%{{customdata[1]}} (%{{customdata[3]:.1%}})" +
              "<extra></extra>",
          }};

          Plotly.react("summary", [
            passTrace,
            speedTrace,
          ], {{
            title: summaryTitle(setName),
            showlegend: false,
            barmode: "overlay",
            template: "plotly_white",
            margin: {{ t: overlay ? 80 : 60, r: 55, b: 30, l: 10 }},
            yaxis: {{
              tickvals: ids,
              ticktext: ticks,
              tickfont: {{ size: 11 }},
              automargin: true,
              autorange: "reversed",
            }},
            xaxis: {{
              domain: overlay ? [0.0, 1.0] : [0.0, 0.47],
              title: "pass rate",
              tickformat: ".0%",
              range: [0, 1],
              gridcolor: "rgba(0,0,0,0.06)",
              zerolinecolor: "rgba(0,0,0,0.12)",
            }},
            xaxis2: {{
              domain: overlay ? [0.0, 1.0] : [0.53, 1.0],
              overlaying: overlay ? "x" : undefined,
              side: overlay ? "top" : undefined,
              title: speedTitle,
              rangemode: "tozero",
              showgrid: overlay ? false : true,
              gridcolor: "rgba(0,0,0,0.06)",
              zerolinecolor: "rgba(0,0,0,0.12)",
            }},
            shapes: overlay
              ? []
              : [{{
                type: "line",
                xref: "paper",
                yref: "paper",
                x0: 0.5,
                x1: 0.5,
                y0: 0,
                y1: 1,
                line: {{ color: "rgba(0,0,0,0.12)", width: 1 }},
              }}],
          }}, {{
            displaylogo: false,
            responsive: true,
          }});
      }}

        function perBenchmarkFrontiers(rs) {{
          const byBench = new Map();
          for (const r of rs) {{
            const b = r.benchmark ?? "";
            if (!byBench.has(b)) byBench.set(b, []);
            byBench.get(b).push(r);
          }}
          const result = new Map();
          for (const [b, rows] of byBench.entries()) {{
            result.set(b, paretoFrontier(rows));
          }}
          return result;
        }}

        function render() {{
          for (const u of filterUpdaters.values()) u();
          const colorBy = colorBySel.value;
          const filters = currentFilters();
          const base = applyFilters(points, filters);

          const multiBench = uniqVals(base, "benchmark").length > 1;
          let frontierPts, frontierRows, benchFrontiers;
          if (multiBench) {{
            benchFrontiers = perBenchmarkFrontiers(base);
            frontierPts = [];
            frontierRows = [];
            for (const [, pts] of benchFrontiers.entries()) {{
              frontierPts.push(...pts);
              frontierRows.push(...pts.map(p => p.r));
            }}
          }} else {{
            frontierPts = paretoFrontier(base);
            frontierRows = frontierPts.map(p => p.r);
            benchFrontiers = null;
          }}

        let rs = base;
        if (onlyFrontierCb.checked) {{
          const fset = new Set(frontierRows);
          rs = base.filter(r => fset.has(r));
        }}

        const shownTotal = rs.reduce((acc, r) => acc + Number(r.total ?? 0), 0);
        const shownTimedOut = rs.reduce((acc, r) => acc + Number(r.timed_out ?? 0), 0);
        const shownTimeoutRate = shownTotal > 0 ? shownTimedOut / shownTotal : 0;
        const shownText =
          (base.length === points.length)
            ? `Showing ${{base.length}} configs. `
            : `Showing ${{base.length}} / ${{points.length}} configs (filtered). `;
        const shownTimeoutPct = (shownTimeoutRate * 100).toFixed(1);
        const timeoutText =
          `Timed out: ${{shownTimedOut}}/${{shownTotal}} (${{shownTimeoutPct}}%). `;
        const fixedText = fixedTokens.length ? `Fixed: ${{fixedTokens.join(" ")}}.` : "";
        const axisHelp = "X = seconds/solve (lower is better). Y = pass rate (higher is better). ";
        document.getElementById("subtitle").textContent =
          shownText + timeoutText + axisHelp + fixedText;

          const traces = makeTraces(rs, colorBy);
          if (paretoCb.checked) {{
            if (benchFrontiers && benchFrontiers.size > 1) {{
              const benchNames = Array.from(benchFrontiers.keys()).sort();
              for (let bi = 0; bi < benchNames.length; bi++) {{
                const bPts = benchFrontiers.get(benchNames[bi]);
                if (!bPts || bPts.length < 2) continue;
                traces.push({{
                  type: "scatter",
                  mode: "lines",
                  name: `Pareto: ${{benchNames[bi]}}`,
                  x: bPts.map(p => p.x),
                  y: bPts.map(p => p.y),
                  hoverinfo: "skip",
                  line: {{ color: PALETTE[bi % PALETTE.length], width: 2, dash: "dot" }},
                  showlegend: false,
                }});
              }}
            }} else if (frontierPts.length >= 2) {{
              traces.push({{
                type: "scatter",
                mode: "lines",
                name: "Best tradeoffs",
                x: frontierPts.map(p => p.x),
                y: frontierPts.map(p => p.y),
                hoverinfo: "skip",
                line: {{ color: "rgba(17,24,39,0.55)", width: 2, dash: "dot" }},
              }});
            }}
          }}

          Plotly.react("scatter", traces, {{
            title: "Pass rate vs seconds/solve",
            xaxis: {{
              title: {{ text: "seconds per solve (lower is better)", standoff: 18 }},
              rangemode: "tozero",
            }},
            yaxis: {{
              title: {{ text: "pass rate (higher is better)", standoff: 10 }},
              tickformat: ".0%",
              rangemode: "tozero",
            }},
            legend: {{
              orientation: "h",
              y: -0.50,
              yanchor: "top",
              x: 0,
              xanchor: "left",
            }},
            margin: {{ t: 60, r: 20, b: 150, l: 65 }},
            template: "plotly_white",
          }}, {{
          displaylogo: false,
          responsive: true,
        }});

          const speedField = speedMetricSel.value;
          const summarySet = summarySetSel.value;
          const viewMode = summaryViewSel.value;
          let summaryRows = (summarySet === "shown") ? rs.slice() : frontierRows.slice();
          summaryRows = summaryRows
            .filter(r => r[speedField] !== null && r[speedField] !== undefined)
            .sort((a, b) => Number(a[speedField]) - Number(b[speedField]));

        const topVal = topNSel.value;
        if (topVal !== "all") {{
          const n = parseInt(topVal, 10);
          if (!Number.isNaN(n) && n > 0) summaryRows = summaryRows.slice(0, n);
        }}
          const summarySetName =
            (summarySetSel.options && summarySetSel.selectedIndex >= 0)
              ? summarySetSel.options[summarySetSel.selectedIndex].textContent
              : summarySet;
          renderSummary(summaryRows, summarySetName, speedField, viewMode);
        }}

      colorBySel.addEventListener("change", render);
      paretoCb.addEventListener("change", render);
      onlyFrontierCb.addEventListener("change", render);
        for (const cbs of filterCheckboxes.values()) {{
          for (const cb of cbs) cb.addEventListener("change", render);
        }}
        summarySetSel.addEventListener("change", render);
        speedMetricSel.addEventListener("change", render);
        summaryViewSel.addEventListener("change", render);
        topNSel.addEventListener("change", render);

      render();
    </script>
  </body>
</html>
"""


@app.command("report")
def report(
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
    out: Annotated[Path, typer.Option("--out", help="Output HTML report path")] = Path(
        "mcode-report.html"
    ),
    benchmark: Annotated[str | None, typer.Option("--benchmark")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    suite_name: Annotated[str | None, typer.Option("--suite")] = None,
    suite_entry_name: Annotated[str | None, typer.Option("--suite-entry")] = None,
    loop_budget: Annotated[int | None, typer.Option("--loop-budget", min=1)] = None,
    timeout_s: Annotated[int | None, typer.Option("--timeout", min=1)] = None,
    per_run: Annotated[
        bool, typer.Option("--per-run", help="Plot each run separately (vs grouped)")
    ] = False,
) -> None:
    """Generate a lightweight HTML report (Plotly) for pass rate vs time-to-solve."""
    group_by_config = (
        "suite_name",
        "suite_entry_name",
        "backend_name",
        "timeout_s",
        "loop_budget",
    )
    group_by = () if per_run else group_by_config

    db_paths = _expand_db_paths(db=db, db_glob=db_glob, db_dir=db_dir)
    with _open_results_view(db_paths) as rdb:
        rows = rdb.run_metrics_grouped(
            benchmark=benchmark,
            model_id=model,
            backend_name=backend,
            timeout_s=timeout_s,
            suite_name=suite_name,
            suite_entry_name=suite_entry_name,
            group_by=group_by,
            loop_budget=loop_budget,
            include_percentiles=True,
        )

    title = "mCode benchmark report"
    if benchmark:
        title += f" | benchmark={benchmark}"
    if backend:
        title += f" | backend={backend}"
    if model:
        title += f" | model={model}"

    out.parent.mkdir(parents=True, exist_ok=True)
    html = _render_report_html(rows, title=title)
    out.write_text(html, encoding="utf-8")
    typer.echo(f"Wrote report: {out}")


@app.command("merge-shards")
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


@app.command("export-csv")
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
        f"diagnostic_events={report.get('diagnostic_events', 0)}\n"
        f"runs_csv={report['runs_csv']}\n"
        f"task_results_csv={report['task_results_csv']}"
    )
    if report.get("diagnostic_events_csv"):
        message += f"\ndiagnostic_events_csv={report['diagnostic_events_csv']}"
    console.print(message)


app.add_typer(bench_app, name="bench")
app.add_typer(deps_app, name="deps")

# Launcher subcommands: `mcode launch bluevela|local-vllm|local-ollama|status|...`.
# Imported lazily here to keep startup cheap and avoid pulling in rich/typer code
# paths when only `mcode bench` is used.
from mcode.launch.cli import app as launch_app  # noqa: E402

app.add_typer(launch_app, name="launch")


def _print_run_summary(
    *,
    summary: RunSummary,
    benchmark: str,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
) -> None:
    table = Table(title="Run summary")
    table.add_column("run_id", justify="right")
    table.add_column("benchmark")
    table.add_column("backend")
    table.add_column("model")
    table.add_column("budget", justify="right")
    table.add_column("timeout", justify="right")
    table.add_column("total", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_row(
        str(summary.run_id),
        benchmark,
        backend,
        model,
        str(loop_budget),
        str(timeout_s),
        str(summary.total),
        str(summary.passed),
        f"{summary.pass_rate:.1%}",
    )
    console.print(table)


def _latest_run_summary(db: Path) -> RunSummary:
    with ResultsDB(db) as rdb:
        row = rdb.conn.execute(
            """
            SELECT
              r.id AS run_id,
              COUNT(tr.id) AS total,
              COALESCE(SUM(tr.passed), 0) AS passed
            FROM runs r
            LEFT JOIN task_results tr ON tr.run_id = r.id
            WHERE r.id = (SELECT MAX(id) FROM runs)
            GROUP BY r.id
            """
        ).fetchone()
    if row is None:
        raise RuntimeError(f"No runs found in {db}")
    return RunSummary(
        run_id=int(row["run_id"]),
        total=int(row["total"] or 0),
        passed=int(row["passed"] or 0),
    )


def _full_db_summary(db: Path) -> RunSummary:
    with ResultsDB(db) as rdb:
        row = rdb.conn.execute(
            """
            SELECT
              COALESCE(MAX(r.id), 0) AS run_id,
              COUNT(tr.id) AS total,
              COALESCE(SUM(tr.passed), 0) AS passed
            FROM runs r
            LEFT JOIN task_results tr ON tr.run_id = r.id
            """
        ).fetchone()
    if row is None:
        raise RuntimeError(f"No runs found in {db}")
    return RunSummary(
        run_id=int(row["run_id"] or 0),
        total=int(row["total"] or 0),
        passed=int(row["passed"] or 0),
    )


def _merge_into_results_db(
    *,
    db: Path,
    shard_paths: list[Path],
    merge_mode: Literal["single_run", "full_db"] = "single_run",
 ) -> RunSummary:
    with temporary_directory(prefix="mcode-merge-") as td:
        merged = Path(td) / "merged.db"
        if merge_mode == "single_run":
            merge_shard_dbs(out_path=merged, shard_paths=shard_paths, force=True)
        else:
            with ResultsDB(merged) as merged_db:
                merged_db.merge_from(shard_paths)
        with ResultsDB(db) as out_db:
            out_db.merge_from([merged])
    if merge_mode == "single_run":
        return _latest_run_summary(db)
    return _full_db_summary(db)


SHARDED_INFRA_EXIT_CODE = 86
_SHARD_INFRA_POLL_SECONDS = 20.0
_INFRA_ERROR_PATTERNS = (
    "writing blob",
    "adding layer",
    "unpacking failed",
    "chown error detected",
    "insufficient uids or gids",
    "podman system migrate",
    "disk i/o error",
    "database is locked",
    "podman socket did not come up",
    "no such container",
)


def _is_retryable_infra_exception(exc: object) -> bool:
    try:
        from mcode.execution.swebench import _is_retryable_podman_image_error

        return _is_retryable_podman_image_error(exc)
    except Exception:
        text = str(exc).lower()
        return any(pattern in text for pattern in _INFRA_ERROR_PATTERNS)


def _shard_run_fingerprint(
    *,
    command: str,
    base_argv: list[str],
    shards: int,
    db: Path,
    benchmark: str,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
) -> str:
    payload = {
        "command": command,
        "base_argv": base_argv,
        "shards": shards,
        "db": str(db.resolve()),
        "benchmark": benchmark,
        "backend": backend,
        "model": model,
        "loop_budget": loop_budget,
        "timeout_s": timeout_s,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _shard_db_has_rows(shard_db: Path) -> bool:
    if not shard_db.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{shard_db}?mode=ro", uri=True, timeout=1)
        try:
            task_rows = conn.execute("SELECT COUNT(*) FROM task_results").fetchone()
            artifact_rows = None
            artifact_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'artifact_tasks'"
            ).fetchone()
            if artifact_exists is not None:
                artifact_rows = conn.execute("SELECT COUNT(*) FROM artifact_tasks").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    task_count = int(task_rows[0]) if task_rows else 0
    artifact_count = int(artifact_rows[0]) if artifact_rows else 0
    return task_count > 0 or artifact_count > 0


def _stop_running_shards(
    procs: list[tuple[int, subprocess.Popen[str], Path, Path, threading.Thread]],
) -> None:
    for _, proc, _, _, _ in procs:
        if proc.poll() is None:
            proc.terminate()
    for _, proc, _, _, thread in procs:
        if proc.poll() is None:
            proc.wait()
        thread.join()


def _stream_shard_output(
    *,
    proc: subprocess.Popen[str],
    shard_index: int,
    log_path: Path,
    dashboard,
) -> threading.Thread:
    def _worker() -> None:
        with log_path.open("w", encoding="utf-8") as handle:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                text = line.rstrip()
                if not text:
                    continue
                dashboard.post("shard_stdout", shard=shard_index, line=text)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _run_sharded_benchmark(
    *,
    command: str,
    base_argv: list[str],
    shards: int,
    db: Path,
    benchmark: str,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
    json_mode: bool = False,
    merge_mode: Literal["single_run", "full_db"] = "single_run",
 ) -> None:
    from mcode.bench import runstate
    from mcode.launch.models import RunStatus, Target
    from mcode.ui.dashboard import open_dashboard

    fingerprint = _shard_run_fingerprint(
        command=command,
        base_argv=base_argv,
        shards=shards,
        db=db,
        benchmark=benchmark,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )
    run_dir = db.parent / f"{db.stem}-shards" / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    procs: list[tuple[int, subprocess.Popen[str], Path, Path, threading.Thread]] = []
    failed: list[tuple[int, int, Path]] = []
    partial: list[tuple[int, int, Path]] = []
    shard_paths: list[Path] = []
    restart_counts: dict[int, int] = {}

    run_id = runstate.make_run_id(benchmark)
    # Target is the closest fit, not literal: bench runs use a launch target
    # only as a "where am I executing" hint. Backend lives in metadata.
    runstate.open_run(run_id=run_id, benchmark=benchmark, target=Target.LOCAL_VLLM, db_path=db)
    runstate.patch_run(run_id=run_id, progress={"current": 0, "total": 0})
    final_status: RunStatus = RunStatus.FAILED
    cancel_reason: str | None = None
    try:
        with open_dashboard(
            json_mode=json_mode,
            total_shards=shards,
            benchmark=benchmark,
            model=model,
        ) as dashboard:
            dashboard.post(
                "info",
                text=(
                    f"▶ sharded run command={command} shards={shards} out={db} artifacts={run_dir}"
                ),
            )

            def launch_shard(shard_index: int) -> None:
                shard_db = run_dir / f"{db.stem}-shard-{shard_index}.db"
                shard_log = run_dir / f"{db.stem}-shard-{shard_index}.log"
                argv = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mcode",
                    "bench",
                    command,
                    *base_argv,
                    "--db",
                    str(shard_db),
                    "--shard-count",
                    str(shards),
                    "--shard-index",
                    str(shard_index),
                ]
                dashboard.post(
                    "shard_start",
                    shard=shard_index,
                    db=str(shard_db),
                    log=str(shard_log),
                )
                proc = subprocess.Popen(
                    argv,
                    cwd=str(Path.cwd()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                thread = _stream_shard_output(
                    proc=proc,
                    shard_index=shard_index,
                    log_path=shard_log,
                    dashboard=dashboard,
                )
                entry = (shard_index, proc, shard_db, shard_log, thread)
                for index, existing in enumerate(procs):
                    if existing[0] == shard_index:
                        procs[index] = entry
                        break
                else:
                    procs.append(entry)

            try:
                for shard_index in range(shards):
                    launch_shard(shard_index)

                runstate.patch_run(run_id=run_id, shard_pids=[p.pid for _, p, *_ in procs])

                remaining = {shard_index for shard_index, *_ in procs}
                while remaining:
                    made_progress = False
                    for shard_index, proc, shard_db, shard_log, thread in list(procs):
                        if shard_index not in remaining:
                            continue
                        rc = proc.poll()
                        if rc is None:
                            continue
                        thread.join()
                        remaining.remove(shard_index)
                        made_progress = True
                        has_rows = _shard_db_has_rows(shard_db)
                        if rc == 0:
                            if has_rows:
                                shard_paths.append(shard_db)
                            dashboard.post("shard_done", shard=shard_index)
                            continue
                        if (
                            rc == SHARDED_INFRA_EXIT_CODE
                            and not has_rows
                            and restart_counts.get(shard_index, 0) < 1
                        ):
                            restart_counts[shard_index] = restart_counts.get(shard_index, 0) + 1
                            dashboard.post(
                                "shard_infra",
                                shard=shard_index,
                                rc=rc,
                                log=str(shard_log),
                            )
                            launch_shard(shard_index)
                            runstate.patch_run(
                                run_id=run_id,
                                shard_pids=[p.pid for _, p, *_ in procs],
                            )
                            remaining.add(shard_index)
                            continue
                        if has_rows:
                            shard_paths.append(shard_db)
                            partial.append((shard_index, rc, shard_log))
                            dashboard.post(
                                "shard_failed",
                                shard=shard_index,
                                rc=rc,
                                log=str(shard_log),
                            )
                            continue
                        failed.append((shard_index, rc, shard_log))
                        dashboard.post(
                            "shard_failed",
                            shard=shard_index,
                            rc=rc,
                            log=str(shard_log),
                        )
                    if remaining and not made_progress:
                        time.sleep(_SHARD_INFRA_POLL_SECONDS)
            except KeyboardInterrupt:
                _stop_running_shards(procs)
                final_status = RunStatus.STOPPED
                cancel_reason = "interrupt"
                raise

            for shard_index, rc, shard_log in partial:
                dashboard.post(
                    "info",
                    text=f"partial shard={shard_index} exit={rc} log={shard_log}",
                )
            for shard_index, rc, shard_log in failed:
                dashboard.post(
                    "info",
                    text=f"failed shard={shard_index} exit={rc} log={shard_log}",
                )
            if not shard_paths:
                raise typer.Exit(1)

            summary = _merge_into_results_db(
                db=db,
                shard_paths=shard_paths,
                merge_mode=merge_mode,
            )
            dashboard.post("merged", db=str(db))
            final_status = RunStatus.DONE
        # _print_run_summary lives outside the dashboard so its Rich Table
        # renders cleanly to stdout/console after the Live region releases.
        _print_run_summary(
            summary=summary,
            benchmark=benchmark,
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
        )
    finally:
        # Best-effort close so partial state doesn't permanently mark the run
        # RUNNING. Wrapped so a second Ctrl+C during teardown cannot prevent
        # the close.
        try:
            runstate.close_run(run_id=run_id, status=final_status, cancel_reason=cancel_reason)
        except Exception:
            pass


def _suite_cli_args(
    *,
    model: str,
    backend: str,
    loop_budget: int,
    retry_loop_budget: int,
    timeout_s: int,
    mem_limit: str,
    pids_limit: int,
    cpu_limit: float | None,
    suite_file: Path | None,
    phase: str,
    artifact_dir: Path,
    diagnostic_traces: bool,
    check_image_digests: bool,
 ) -> list[str]:
    argv = [
        "--model",
        model,
        "--backend",
        backend,
        "--loop-budget",
        str(loop_budget),
        "--retry-loop-budget",
        str(retry_loop_budget),
        "--timeout",
        str(timeout_s),
        "--mem-limit",
        mem_limit,
        "--pids-limit",
        str(pids_limit),
        "--phase",
        phase,
        "--artifact-dir",
        str(artifact_dir),
    ]
    _append_option(argv, "--suite-file", suite_file)
    _append_option(argv, "--cpu-limit", cpu_limit)
    if diagnostic_traces:
        argv.append("--diagnostic-traces")
    if not check_image_digests:
        argv.append("--no-check-image-digests")
    return argv


def _suite_entry_timeout(entry: SuiteEntry, swebench_timeout_s: int) -> int:
    return 300 if entry.benchmark == "aider-polyglot" else swebench_timeout_s


def _suite_entry_loop_budget(
    entry: SuiteEntry, *, loop_budget: int, retry_loop_budget: int
 ) -> int:
    if entry.benchmark != "aider-polyglot":
        return loop_budget
    return loop_budget + (0 if entry.no_retry else retry_loop_budget)


def _resolve_suite_polyglot_root(entry: SuiteEntry) -> Path | None:
    if entry.benchmark != "aider-polyglot":
        return None
    if entry.benchmark_root:
        candidate = Path(entry.benchmark_root)
        if candidate.is_dir():
            return candidate
    bundled_root = Path("benchmarks/polyglot-benchmark")
    if bundled_root.is_dir():
        return bundled_root
    return None


def _run_suite_entry(
    *,
    suite_name: str,
    entry: SuiteEntry,
    model: str,
    backend: str,
    loop_budget: int,
    retry_loop_budget: int,
    timeout_s: int,
    mem_limit: str,
    pids_limit: int,
    cpu_limit: float | None,
    check_image_digests: bool,
    phase: str,
    artifact_dir: Path,
    db: Path,
    shard_count: int | None,
    shard_index: int | None,
    diagnostic_traces: bool,
 ) -> None:
    task_ids = task_ids_arg(entry)
    entry_timeout = _suite_entry_timeout(entry, timeout_s)
    entry_loop_budget = _suite_entry_loop_budget(
        entry,
        loop_budget=loop_budget,
        retry_loop_budget=retry_loop_budget,
    )
    if entry.benchmark == "swebench-lite":
        config = BenchConfig(
            backend_name=backend,
            model_id=model,
            loop_budget=loop_budget,
            timeout_s=entry_timeout,
            phase=phase,
            artifact_dir=artifact_dir,
            swebench_split=entry.split or "test",
            swebench_namespace="swebench",
            swebench_mem_limit=mem_limit,
            swebench_pids_limit=pids_limit,
            swebench_cpu_limit=cpu_limit,
            swebench_check_image_digests=check_image_digests,
            task_shard_count=shard_count,
            task_shard_index=shard_index,
            swebench_dataset=entry.dataset or "SWE-bench/SWE-bench_Lite",
            diagnostic_traces=diagnostic_traces,
            suite_name=suite_name,
            suite_entry_name=entry.name,
        )
    elif entry.benchmark == "swebench-live":
        config = BenchConfig(
            backend_name=backend,
            model_id=model,
            loop_budget=loop_budget,
            timeout_s=entry_timeout,
            phase=phase,
            artifact_dir=artifact_dir,
            swebench_split=entry.split or "verified",
            swebench_mem_limit=mem_limit,
            swebench_pids_limit=pids_limit,
            swebench_cpu_limit=cpu_limit,
            swebench_check_image_digests=check_image_digests,
            task_shard_count=shard_count,
            task_shard_index=shard_index,
            diagnostic_traces=diagnostic_traces,
            suite_name=suite_name,
            suite_entry_name=entry.name,
        )
    elif entry.benchmark == "aider-polyglot":
        config = BenchConfig(
            backend_name=backend,
            model_id=model,
            loop_budget=loop_budget,
            timeout_s=entry_timeout,
            phase=phase,
            artifact_dir=artifact_dir,
            aider_polyglot_root=_resolve_suite_polyglot_root(entry),
            aider_polyglot_language=entry.language or "all",
            aider_polyglot_retry=not entry.no_retry,
            aider_polyglot_retry_loop_budget=retry_loop_budget,
            task_shard_count=shard_count,
            task_shard_index=shard_index,
            diagnostic_traces=diagnostic_traces,
            suite_name=suite_name,
            suite_entry_name=entry.name,
        )
    else:
        raise typer.BadParameter(f"unsupported suite benchmark {entry.benchmark!r}")
    _run_single_benchmark(
        benchmark=entry.benchmark,
        config=config,
        db=db,
        limit=entry.limit,
        task_ids=task_ids,
        backend=backend,
        model=model,
        loop_budget=entry_loop_budget,
        timeout_s=entry_timeout,
    )


def _run_single_benchmark(
    *,
    benchmark: str,
    config: BenchConfig,
    db: Path,
    limit: int | None,
    task_ids: str | None,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
) -> None:
    from mcode.bench import runstate
    from mcode.launch.models import RunStatus, Target

    parsed_task_ids = _parse_task_ids(task_ids)
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db))
    run_id = runstate.make_run_id(benchmark)
    runstate.open_run(run_id=run_id, benchmark=benchmark, target=Target.LOCAL_VLLM, db_path=db)
    final_status: RunStatus = RunStatus.FAILED
    cancel_reason: str | None = None
    try:
        try:
            summary = runner.run_benchmark(benchmark, limit=limit, task_ids=parsed_task_ids)
        except KeyboardInterrupt:
            final_status = RunStatus.STOPPED
            cancel_reason = "interrupt"
            raise
        except Exception as e:
            if benchmark.startswith("swebench") and _is_retryable_infra_exception(e):
                typer.echo(f"✗ retryable infra failure before task loop: {e}", err=True)
                raise typer.Exit(SHARDED_INFRA_EXIT_CODE) from e
            raise
        _print_run_summary(
            summary=summary,
            benchmark=benchmark,
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
        )
        final_status = RunStatus.DONE
    finally:
        try:
            runstate.close_run(run_id=run_id, status=final_status, cancel_reason=cancel_reason)
        except Exception:
            pass


def _run_bluevela_benchmark(
    *,
    command: str,
    argv: list[str],
    model: str,
    db: Path,
    fetch_db: bool,
    fetch_artifacts: bool = False,
 ) -> None:
    from mcode.bench.remote import RemoteBenchError, run_bench_on_bluevela

    try:
        rc = run_bench_on_bluevela(
            bench_argv=[command, *argv],
            model=model,
            local_db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    except RemoteBenchError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from e
    raise typer.Exit(rc)


def _swebench_live_cli_args(
    *,
    model: str,
    backend: str,
    loop_budget: int,
    temperature: float | None,
    seed: int | None,
    timeout_s: int,
    split: str,
    mem_limit: str,
    pids_limit: int,
    cpu_limit: float | None,
    limit: int | None,
    n_samples: int,
    sampling: str,
    sampling_budget: int | None,
    selection_attempts: int,
    task_ids: str | None,
    diagnostic_traces: bool,
    check_image_digests: bool,
    phase: str,
    artifact_dir: Path,
 ) -> list[str]:
    argv = [
        "--model",
        model,
        "--backend",
        backend,
        "--loop-budget",
        str(loop_budget),
        "--timeout",
        str(timeout_s),
        "--split",
        split,
        "--mem-limit",
        mem_limit,
        "--pids-limit",
        str(pids_limit),
        "--n-samples",
        str(n_samples),
        "--sampling",
        sampling,
        "--phase",
        phase,
        "--artifact-dir",
        str(artifact_dir),
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--sampling-budget", sampling_budget)
    if selection_attempts != 1:
        _append_option(argv, "--selection-attempts", selection_attempts)
    _append_option(argv, "--task-ids", task_ids)
    _append_option(argv, "--cpu-limit", cpu_limit)
    if not check_image_digests:
        argv.append("--no-check-image-digests")
    if diagnostic_traces:
        argv.append("--diagnostic-traces")
    return argv


def _swebench_lite_cli_args(
    *,
    model: str,
    backend: str,
    loop_budget: int,
    temperature: float | None,
    seed: int | None,
    timeout_s: int,
    split: str,
    arch: str,
    namespace: str,
    max_workers: int,
    force_rebuild: bool,
    mem_limit: str,
    pids_limit: int,
    cpu_limit: float | None,
    limit: int | None,
    n_samples: int,
    sampling: str,
    sampling_budget: int | None,
    selection_attempts: int,
    task_ids: str | None,
    dataset: str,
    diagnostic_traces: bool,
    check_image_digests: bool,
    phase: str,
    artifact_dir: Path,
) -> list[str]:
    argv = [
        "--model",
        model,
        "--backend",
        backend,
        "--loop-budget",
        str(loop_budget),
        "--timeout",
        str(timeout_s),
        "--split",
        split,
        "--arch",
        arch,
        "--namespace",
        namespace,
        "--max-workers",
        str(max_workers),
        "--mem-limit",
        mem_limit,
        "--pids-limit",
        str(pids_limit),
        "--n-samples",
        str(n_samples),
        "--sampling",
        sampling,
        "--dataset",
        dataset,
        "--phase",
        phase,
        "--artifact-dir",
        str(artifact_dir),
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--sampling-budget", sampling_budget)
    if selection_attempts != 1:
        _append_option(argv, "--selection-attempts", selection_attempts)
    _append_option(argv, "--task-ids", task_ids)
    _append_option(argv, "--cpu-limit", cpu_limit)
    if force_rebuild:
        argv.append("--force-rebuild")
    if not check_image_digests:
        argv.append("--no-check-image-digests")
    if diagnostic_traces:
        argv.append("--diagnostic-traces")
    return argv


def _aider_polyglot_cli_args(
    *,
    model: str,
    backend: str,
    loop_budget: int,
    retry_loop_budget: int,
    temperature: float | None,
    seed: int | None,
    benchmark_root: Path,
    language: str,
    exercise: str | None,
    limit: int | None,
    no_retry: bool,
    task_ids: str | None,
    phase: str,
    artifact_dir: Path,
) -> list[str]:
    argv = [
        "--model",
        model,
        "--backend",
        backend,
        "--loop-budget",
        str(loop_budget),
        "--retry-loop-budget",
        str(retry_loop_budget),
        "--benchmark-root",
        str(benchmark_root),
        "--language",
        language,
        "--phase",
        phase,
        "--artifact-dir",
        str(artifact_dir),
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--exercise", exercise)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--task-ids", task_ids)
    if no_retry:
        argv.append("--no-retry")
    return argv


def _resolve_results_run_id(rdb: ResultsDB, run_id: int | None) -> int:
    if run_id is not None:
        return run_id
    row = rdb.conn.execute("SELECT MAX(id) AS run_id FROM runs").fetchone()
    if row is None or row["run_id"] is None:
        raise typer.BadParameter(f"No runs found in {rdb.path}")
    return int(row["run_id"])


@bench_app.command("list")
def bench_list(
    json_mode: JsonFlag = False,
    benchmark: Annotated[str | None, typer.Option("--benchmark")] = None,
    status: Annotated[str | None, typer.Option("--status")] = None,
    artifacts_only: Annotated[
        bool,
        typer.Option("--artifacts", help="Only show runs with remote artifact metadata"),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
 ) -> None:
    """List historical bench runs from the launch state file."""
    from mcode.bench.cancel import list_runs

    rc = list_runs(
        json_mode=json_mode,
        benchmark=benchmark,
        status=status,
        artifacts_only=artifacts_only,
        limit=limit,
    )
    if rc != 0:
        raise typer.Exit(rc)


@bench_app.command("cancel")
def bench_cancel(
    run_id: str = typer.Argument(..., help="run id (from `mcode bench list`)"),
) -> None:
    """Cancel a running bench. Terminates shard pids (local) or SSH-kills the
    remote process group (Blue Vela). In-process single runs are not
    cancellable from another shell, use Ctrl+C in the running terminal."""
    from mcode.bench.cancel import cancel_run
    from mcode.ui.errors import handle_errors

    @handle_errors
    def _do() -> None:
        rc = cancel_run(run_id)
        if rc != 0:
            raise typer.Exit(rc)

    _do()
def _artifact_replay_config(
    *,
    source_db: Path,
    run_id: int,
    task_id: str,
    candidate_index: int | None,
    benchmark_root: Path | None = None,
    artifact_dir_override: Path | None = None,
 ) -> tuple[str, BenchConfig, Path]:
    with ResultsDB(source_db) as rdb:
        row = rdb.conn.execute(
            """
            SELECT
              r.benchmark AS benchmark,
              r.config_json AS config_json,
              at.manifest_path AS manifest_path,
              at.artifact_root AS artifact_root
            FROM artifact_tasks at
            JOIN runs r ON r.id = at.run_id
            WHERE at.run_id = ? AND at.task_id = ?
            LIMIT 1
            """,
            (run_id, task_id),
        ).fetchone()
    if row is None:
        raise typer.BadParameter(
            f"No artifact manifest for task {task_id!r} in run {run_id}"
        )
    manifest_path = Path(str(row["manifest_path"]))
    if artifact_dir_override is not None:
        manifest_path = artifact_dir_override / str(row["artifact_root"]) / "manifest.json"
    manifest = read_task_manifest(manifest_path)
    artifact_dir = manifest_path.parent
    for _ in Path(manifest.task.artifact_root).parts:
        artifact_dir = artifact_dir.parent
    raw_config = json.loads(str(row["config_json"]))
    allowed = set(BenchConfig.__dataclass_fields__)
    config_kwargs = {key: value for key, value in raw_config.items() if key in allowed}
    for path_key in ("cache_dir", "artifact_dir", "aider_polyglot_root"):
        value = config_kwargs.get(path_key)
        if isinstance(value, str) and value:
            config_kwargs[path_key] = Path(value)
    config_kwargs["phase"] = "evaluate"
    config_kwargs["artifact_dir"] = artifact_dir
    config_kwargs["artifact_candidate_index"] = candidate_index
    if benchmark_root is not None:
        config_kwargs["aider_polyglot_root"] = benchmark_root
    return str(row["benchmark"]), BenchConfig(**config_kwargs), artifact_dir


@bench_app.command("artifacts-list")
def bench_artifacts_list(
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = DEFAULT_DB_PATH,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Run id (defaults to latest run)"),
    ] = None,
    task_id: Annotated[str | None, typer.Option("--task-id")] = None,
    phase: Annotated[str | None, typer.Option("--phase")] = None,
    json_mode: JsonFlag = False,
 ) -> None:
    """List artifact-backed tasks for one run."""
    with ResultsDB(db) as rdb:
        resolved_run_id = _resolve_results_run_id(rdb, run_id)
        rows = rdb.task_artifact_rows(resolved_run_id)
    filtered = [
        {"task_id": current_task_id, **rows[current_task_id]}
        for current_task_id in sorted(rows)
        if (task_id is None or current_task_id == task_id)
        and (phase is None or str(rows[current_task_id].get("phase")) == phase)
    ]
    if json_mode:
        console.print_json(data=filtered)
        return
    table = Table(title=f"Artifacts for run {resolved_run_id}")
    table.add_column("task_id", no_wrap=True)
    table.add_column("phase")
    table.add_column("candidates", justify="right")
    table.add_column("evaluations", justify="right")
    table.add_column("manifest", overflow="fold")
    for row in filtered:
        table.add_row(
            row["task_id"],
            str(row.get("phase") or "-"),
            str(row.get("candidate_count", 0)),
            str(row.get("evaluation_count", 0)),
            str(row.get("manifest_path") or "-"),
        )
    console.print(table)


def _resolve_artifact_fetch_run(*, run_id: str | None, db: Path | None):
    from mcode.launch import state as launch_state

    state = launch_state.load()
    if run_id:
        run = state.run(run_id)
        if run is None:
            raise typer.BadParameter(f"No run with id {run_id!r}")
        return run
    if db is None:
        raise typer.BadParameter("Provide either a run id or --db")
    target_db = db.resolve()
    matches = [
        run
        for run in state.runs
        if run.db_path
        and Path(str(run.db_path)).resolve() == target_db
        and (run.remote or {}).get("remote_artifact_dir")
    ]
    if not matches:
        raise typer.BadParameter(f"No artifact-fetchable run recorded for {db}")
    matches.sort(key=lambda run: float(run.started_at or 0.0))
    return matches[-1]


@bench_app.command("artifacts-fetch")
def bench_artifacts_fetch(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Remote run id from `mcode bench list`"),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Resolve the latest fetchable run for this local DB"),
    ] = None,
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Override the local artifact directory destination"),
    ] = None,
    json_mode: JsonFlag = False,
 ) -> None:
    """Fetch a remote artifact directory for a finished Blue Vela run."""
    from mcode.launch.ssh import SshClient
    from mcode.ui.errors import MCodeError, handle_errors

    @handle_errors
    def _do() -> None:
        run = _resolve_artifact_fetch_run(run_id=run_id, db=db)
        resolved_run_id = run.id
        login = str(run.remote.get("login") or "")
        remote_artifact_dir = str(run.remote.get("remote_artifact_dir") or "")
        saved_local_artifact_dir = str(run.remote.get("local_artifact_dir") or "")
        local_artifact_dir = Path(dest) if dest is not None else Path(saved_local_artifact_dir)
        if not login or not remote_artifact_dir or not str(local_artifact_dir):
            raise MCodeError(
                what=f"run {resolved_run_id!r} has no deferred artifact fetch metadata",
                why="the run did not record a remote artifact directory",
                next="rerun the remote bench with --fetch-artifacts or specify a fresh run id",
            )
        ssh = SshClient(login)
        probe = ssh.run(
            f"test -d {shlex.quote(remote_artifact_dir)} && echo ok || echo missing",
            timeout=30,
        )
        if not probe.ok or not probe.stdout.strip().endswith("ok"):
            raise MCodeError(
                what=f"remote artifact directory is missing for {resolved_run_id}",
                why=remote_artifact_dir,
                next="rerun the remote bench with --fetch-artifacts, or inspect the remote run dir",
            )
        try:
            ssh.download_tree(remote_artifact_dir, local_artifact_dir, timeout=300)
        except Exception as exc:
            raise MCodeError(
                what=f"failed to fetch artifacts for {resolved_run_id}",
                why=str(exc),
                next="check SSH reachability, remote paths, and local disk space, then retry",
            ) from exc
        payload = {
            "run_id": resolved_run_id,
            "remote_artifact_dir": remote_artifact_dir,
            "local_artifact_dir": str(local_artifact_dir),
        }
        if json_mode:
            console.print_json(data=payload)
            return
        console.print(f"fetched artifacts to {local_artifact_dir}")

    _do()
@bench_app.command("artifacts-show")
def bench_artifacts_show(
    task_id: Annotated[str, typer.Argument(..., help="Task id to inspect")],
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = DEFAULT_DB_PATH,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Run id (defaults to latest run)"),
    ] = None,
    candidate_index: Annotated[
        int | None,
        typer.Option("--candidate-index", help="Show only one candidate entry"),
    ] = None,
 ) -> None:
    """Show one task artifact manifest."""
    with ResultsDB(db) as rdb:
        resolved_run_id = _resolve_results_run_id(rdb, run_id)
        rows = rdb.task_artifact_rows(resolved_run_id)
    row = rows.get(task_id)
    if row is None:
        raise typer.BadParameter(
            f"No artifact manifest for task {task_id!r} in run {resolved_run_id}"
        )
    manifest = read_task_manifest(Path(str(row["manifest_path"])))
    if candidate_index is not None:
        candidate = next(
            (item for item in manifest.candidates if item.candidate_index == candidate_index),
            None,
        )
        if candidate is None:
            raise typer.BadParameter(
                f"No candidate index {candidate_index} for task {task_id!r} "
                f"in run {resolved_run_id}"
            )
        console.print_json(data=asdict(candidate))
        return
    console.print(json.dumps(asdict(manifest), indent=2, sort_keys=True, default=str))
@bench_app.command("artifacts-replay")
def bench_artifacts_replay(
    task_id: Annotated[str, typer.Argument(..., help="Task id to evaluate")],
    db: Annotated[
        Path,
        typer.Option("--db", help="Source SQLite results DB path"),
    ] = DEFAULT_DB_PATH,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Run id (defaults to latest run)"),
    ] = None,
    out_db: Annotated[
        Path | None,
        typer.Option("--out-db", help="Destination SQLite DB path"),
    ] = None,
    candidate_index: Annotated[
        int | None,
        typer.Option("--candidate-index", help="Candidate index to replay"),
    ] = None,
    benchmark_root: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-root",
            help="Override the saved benchmark root when replaying cross-machine artifacts",
        ),
    ] = None,
    artifact_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifact-dir",
            help="Override the saved artifact directory when replaying artifacts copied elsewhere",
        ),
    ] = None,
 ) -> None:
    """Re-evaluate one saved artifact candidate through the benchmark adapter."""
    with ResultsDB(db) as rdb:
        resolved_run_id = _resolve_results_run_id(rdb, run_id)
    benchmark, config, _artifact_dir = _artifact_replay_config(
        source_db=db,
        run_id=resolved_run_id,
        task_id=task_id,
        candidate_index=candidate_index,
        benchmark_root=benchmark_root,
        artifact_dir_override=artifact_dir,
    )
    target_db = out_db if out_db is not None else db.with_name(f"{db.stem}-replay.db")
    _run_single_benchmark(
        benchmark=benchmark,
        config=config,
        db=target_db,
        limit=None,
        task_ids=task_id,
        backend=config.backend_name,
        model=config.model_id,
        loop_budget=config.loop_budget + (
            config.aider_polyglot_retry_loop_budget
            if benchmark == "aider-polyglot" and config.aider_polyglot_retry
            else 0
        ),
        timeout_s=config.timeout_s,
    )



@bench_app.command("artifacts-patch")
def bench_artifacts_patch(
    task_id: Annotated[str, typer.Argument(..., help="Task id to inspect")],
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = DEFAULT_DB_PATH,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Run id (defaults to latest run)"),
    ] = None,
    candidate_index: Annotated[
        int | None,
        typer.Option("--candidate-index", help="Candidate index (defaults to selected candidate)"),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the patch to a file instead of stdout"),
    ] = None,
 ) -> None:
    """Print one saved candidate patch."""
    with ResultsDB(db) as rdb:
        resolved_run_id = _resolve_results_run_id(rdb, run_id)
        rows = rdb.task_artifact_rows(resolved_run_id)
    row = rows.get(task_id)
    if row is None:
        raise typer.BadParameter(
            f"No artifact manifest for task {task_id!r} in run {resolved_run_id}"
        )
    manifest = read_task_manifest(Path(str(row["manifest_path"])))
    candidate = None
    if candidate_index is None:
        candidate = next((item for item in manifest.candidates if item.selected), None)
    else:
        candidate = next(
            (item for item in manifest.candidates if item.candidate_index == candidate_index),
            None,
        )
    if candidate is None:
        raise typer.BadParameter(
            f"No candidate patch for task {task_id!r} in run {resolved_run_id}"
        )
    manifest_path = Path(str(row["manifest_path"]))
    patch_path = manifest_path.parent / candidate.patch_path
    patch_text = patch_path.read_text(encoding="utf-8")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(patch_text, encoding="utf-8")
        typer.echo(str(out))
        return
    typer.echo(patch_text)





@bench_app.command("swebench-live")
def bench_swebench_live(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    loop_budget: Annotated[
        int,
        typer.Option("--loop-budget", min=1, help="Max attempts per task (with error feedback)"),
    ] = 15,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Sampling temperature"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per SWE-bench eval attempt"),
    ] = 1800,
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split (test/lite/verified/full)"),
    ] = "verified",
    mem_limit: Annotated[
        str,
        typer.Option("--mem-limit", help="Eval container memory limit"),
    ] = "4g",
    pids_limit: Annotated[
        int,
        typer.Option("--pids-limit", min=64, help="Eval container process limit"),
    ] = 512,
    cpu_limit: Annotated[
        float | None,
        typer.Option(
            "--cpu-limit",
            help=(
                "Cap each eval container at N cores (cgroup cpu_quota). "
                "Default: unlimited. Use --on bluevela where login-node admins "
                "kill processes >100 cores."
            ),
            envvar="MCODE_SWEBENCH_CPU_LIMIT",
        ),
    ] = None,
    check_image_digests: Annotated[
        bool,
        typer.Option(
            "--check-image-digests/--no-check-image-digests",
            help="Check registry digests before reusing cached task images",
            envvar="MCODE_SWEBENCH_CHECK_IMAGE_DIGESTS",
        ),
    ] = False,
    shards: Annotated[
        int | None,
        typer.Option("--shards", min=1, help="Run N shard workers and merge the DB automatically"),
    ] = None,
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Manual shard mode: total shard count"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Manual shard mode: shard index"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = DEFAULT_DB_PATH,
    phase: Annotated[
        Literal["run", "generate", "evaluate"],
        typer.Option("--phase", help="Benchmark phase: run, generate, or evaluate"),
    ] = "run",
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for generated task artifacts"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    n_samples: Annotated[
        int,
        typer.Option("--n-samples", min=1, help="Outer attempts or sampling budget"),
    ] = 1,
    sampling: Annotated[
        Literal["none", "multiturn"],
        typer.Option("--sampling", help="Mellea sampling strategy"),
    ] = "none",
    sampling_budget: Annotated[
        int | None,
        typer.Option("--sampling-budget", min=1, help="Sampling loop budget override"),
    ] = None,
    selection_attempts: Annotated[
        int,
        typer.Option(
            "--selection-attempts",
            min=1,
            help="Independent full-budget trajectories; select one before official evaluation",
        ),
    ] = 1,
    task_ids: Annotated[
        str | None,
        typer.Option(
            "--task-ids",
            help="Comma-separated task IDs to run (or path to JSON/text file)",
        ),
    ] = None,
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
    fetch_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-artifacts/--no-fetch-artifacts",
            help="Rsync the artifact directory back when --on bluevela",
        ),
    ] = False,
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Run Microsoft SWE-bench-Live benchmark."""

    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    sampling, sampling_budget = _validate_sampling(
        sampling=sampling,
        sampling_budget=sampling_budget,
    )
    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)
    if on == "bluevela":
        argv = _swebench_live_cli_args(
            model=model,
            backend=backend,
            loop_budget=loop_budget,
            temperature=temperature,
            seed=seed,
            timeout_s=timeout_s,
            split=split,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            cpu_limit=cpu_limit,
            limit=limit,
            n_samples=n_samples,
            sampling=sampling,
            sampling_budget=sampling_budget,
            selection_attempts=selection_attempts,
            task_ids=task_ids,
            diagnostic_traces=diagnostic_traces,
            check_image_digests=check_image_digests,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        if json_mode:
            argv.append("--json")
        _run_bluevela_benchmark(
            command="swebench-live",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)
    if shards and shards > 1:
        _run_sharded_benchmark(
            command="swebench-live",
            base_argv=_swebench_live_cli_args(
                model=model,
                backend=backend,
                loop_budget=loop_budget,
                temperature=temperature,
                seed=seed,
                timeout_s=timeout_s,
                split=split,
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                cpu_limit=cpu_limit,
                limit=limit,
                n_samples=n_samples,
                sampling=sampling,
                sampling_budget=sampling_budget,
                selection_attempts=selection_attempts,
                task_ids=task_ids,
                diagnostic_traces=diagnostic_traces,
                check_image_digests=check_image_digests,
                phase=phase,
                artifact_dir=resolved_artifact_dir,
            ),
            shards=shards,
            db=db,
            benchmark="swebench-live",
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
            json_mode=json_mode,
        )
        return
    if shard_count and shard_count > 1 and db == DEFAULT_DB_PATH:
        typer.echo(
            "Note: when running shards in parallel, use a unique --db per shard to avoid SQLite "
            "locks.",
            err=True,
        )

    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        loop_budget=loop_budget,
        temperature=temperature,
        seed=seed,
        timeout_s=timeout_s,
        phase=phase,
        artifact_dir=resolved_artifact_dir,
        swebench_split=split,
        swebench_mem_limit=mem_limit,
        swebench_pids_limit=pids_limit,
        swebench_cpu_limit=cpu_limit,
        swebench_check_image_digests=check_image_digests,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
        n_samples=n_samples,
        sampling_strategy=sampling,
        sampling_budget=sampling_budget,
        selection_attempts=selection_attempts,
        diagnostic_traces=diagnostic_traces,
    )
    _run_single_benchmark(
        benchmark="swebench-live",
        config=config,
        db=db,
        limit=limit,
        task_ids=task_ids,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )


@bench_app.command("swebench-lite")
def bench_swebench_lite(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    loop_budget: Annotated[
        int,
        typer.Option("--loop-budget", min=1, help="Max attempts per task (with error feedback)"),
    ] = 15,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Sampling temperature"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per SWE-bench eval attempt"),
    ] = 1800,
    split: Annotated[str, typer.Option("--split", help="Dataset split (dev/test)")] = "test",
    arch: Annotated[
        str,
        typer.Option(
            "--arch",
            help=("Image arch: auto/x86_64/arm64 (auto prefers x86_64 for prebuilt images)."),
        ),
    ] = "auto",
    namespace: Annotated[
        str,
        typer.Option(
            "--namespace",
            help=('Prebuilt image namespace (default: swebench); set to "" to build locally.'),
        ),
    ] = "swebench",
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", min=1, help="Parallelism for image building"),
    ] = 4,
    force_rebuild: Annotated[
        bool,
        typer.Option("--force-rebuild", help="Rebuild images even if they exist"),
    ] = False,
    mem_limit: Annotated[
        str,
        typer.Option("--mem-limit", help="Eval container memory limit"),
    ] = "4g",
    pids_limit: Annotated[
        int,
        typer.Option("--pids-limit", min=64, help="Eval container process limit"),
    ] = 512,
    cpu_limit: Annotated[
        float | None,
        typer.Option(
            "--cpu-limit",
            help=(
                "Cap each eval container at N cores (cgroup cpu_quota). "
                "Default: unlimited. Use --on bluevela where login-node admins "
                "kill processes >100 cores."
            ),
            envvar="MCODE_SWEBENCH_CPU_LIMIT",
        ),
    ] = None,
    check_image_digests: Annotated[
        bool,
        typer.Option(
            "--check-image-digests/--no-check-image-digests",
            help="Check registry digests before reusing cached task images",
            envvar="MCODE_SWEBENCH_CHECK_IMAGE_DIGESTS",
        ),
    ] = False,
    shards: Annotated[
        int | None,
        typer.Option("--shards", min=1, help="Run N shard workers and merge the DB automatically"),
    ] = None,
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Manual shard mode: total shard count"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Manual shard mode: shard index"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = DEFAULT_DB_PATH,
    phase: Annotated[
        Literal["run", "generate", "evaluate"],
        typer.Option("--phase", help="Benchmark phase: run, generate, or evaluate"),
    ] = "run",
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for generated task artifacts"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    n_samples: Annotated[
        int,
        typer.Option("--n-samples", min=1, help="Outer attempts or sampling budget"),
    ] = 1,
    sampling: Annotated[
        Literal["none", "multiturn"],
        typer.Option("--sampling", help="Mellea sampling strategy"),
    ] = "none",
    sampling_budget: Annotated[
        int | None,
        typer.Option("--sampling-budget", min=1, help="Sampling loop budget override"),
    ] = None,
    selection_attempts: Annotated[
        int,
        typer.Option(
            "--selection-attempts",
            min=1,
            help="Independent full-budget trajectories; select one before official evaluation",
        ),
    ] = 1,
    task_ids: Annotated[
        str | None,
        typer.Option(
            "--task-ids",
            help="Comma-separated task IDs to run (or path to JSON/text file)",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="HuggingFace dataset name"),
    ] = "SWE-bench/SWE-bench_Lite",
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
    fetch_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-artifacts/--no-fetch-artifacts",
            help="Rsync the artifact directory back when --on bluevela",
        ),
    ] = False,
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    sampling, sampling_budget = _validate_sampling(
        sampling=sampling,
        sampling_budget=sampling_budget,
    )
    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)
    if on == "bluevela":
        argv = _swebench_lite_cli_args(
            model=model,
            backend=backend,
            loop_budget=loop_budget,
            temperature=temperature,
            seed=seed,
            timeout_s=timeout_s,
            split=split,
            arch=arch,
            namespace=namespace,
            max_workers=max_workers,
            force_rebuild=force_rebuild,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            cpu_limit=cpu_limit,
            limit=limit,
            n_samples=n_samples,
            sampling=sampling,
            sampling_budget=sampling_budget,
            selection_attempts=selection_attempts,
            task_ids=task_ids,
            dataset=dataset,
            diagnostic_traces=diagnostic_traces,
            check_image_digests=check_image_digests,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        if json_mode:
            argv.append("--json")
        _run_bluevela_benchmark(
            command="swebench-lite",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)
    if shards and shards > 1:
        _run_sharded_benchmark(
            command="swebench-lite",
            base_argv=_swebench_lite_cli_args(
                model=model,
                backend=backend,
                loop_budget=loop_budget,
                temperature=temperature,
                seed=seed,
                timeout_s=timeout_s,
                split=split,
                arch=arch,
                namespace=namespace,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                cpu_limit=cpu_limit,
                limit=limit,
                n_samples=n_samples,
                sampling=sampling,
                sampling_budget=sampling_budget,
                selection_attempts=selection_attempts,
                task_ids=task_ids,
                dataset=dataset,
                diagnostic_traces=diagnostic_traces,
                check_image_digests=check_image_digests,
                phase=phase,
                artifact_dir=resolved_artifact_dir,
            ),
            shards=shards,
            db=db,
            benchmark="swebench-lite",
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
            json_mode=json_mode,
        )
        return
    if shard_count and shard_count > 1 and db == DEFAULT_DB_PATH:
        typer.echo(
            "Note: when running shards in parallel, use a unique --db per shard to avoid SQLite "
            "locks.",
            err=True,
        )

    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        loop_budget=loop_budget,
        temperature=temperature,
        seed=seed,
        timeout_s=timeout_s,
        phase=phase,
        artifact_dir=resolved_artifact_dir,
        swebench_split=split,
        swebench_namespace=_optional_str(namespace),
        swebench_arch=None if arch == "auto" else arch,
        swebench_max_workers=max_workers,
        swebench_force_rebuild=force_rebuild,
        swebench_mem_limit=mem_limit,
        swebench_pids_limit=pids_limit,
        swebench_cpu_limit=cpu_limit,
        swebench_check_image_digests=check_image_digests,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
        n_samples=n_samples,
        sampling_strategy=sampling,
        sampling_budget=sampling_budget,
        selection_attempts=selection_attempts,
        swebench_dataset=dataset,
        diagnostic_traces=diagnostic_traces,
    )
    _run_single_benchmark(
        benchmark="swebench-lite",
        config=config,
        db=db,
        limit=limit,
        task_ids=task_ids,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )


@bench_app.command("aider-polyglot")
def bench_aider_polyglot(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "openai",
    loop_budget: Annotated[
        int,
        typer.Option("--loop-budget", min=1, help="First-attempt turn budget"),
    ] = 12,
    retry_loop_budget: Annotated[
        int,
        typer.Option("--retry-loop-budget", min=1, help="Second-attempt turn budget"),
    ] = 8,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Sampling temperature"),
    ] = 0.3,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    benchmark_root: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-root",
            help="Path to the cloned Aider Polyglot benchmark repo",
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", help="Language to run (python/go/rust/javascript/cpp/java/all)"),
    ] = "all",
    exercise: Annotated[
        str | None,
        typer.Option("--exercise", help="Single exercise name (requires --language)"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    no_retry: Annotated[
        bool,
        typer.Option("--no-retry", help="Disable the second attempt with test output feedback"),
    ] = False,
    task_ids: Annotated[
        str | None,
        typer.Option(
            "--task-ids",
            help="Comma-separated task IDs like python/hello-world (or path to JSON/text file)",
        ),
    ] = None,
    shards: Annotated[
        int | None,
        typer.Option("--shards", min=1, help="Run N shard workers and merge the DB automatically"),
    ] = None,
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Manual shard mode: total shard count"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Manual shard mode: shard index"),
    ] = None,
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite results DB path"),
    ] = Path("experiments/results/aider-polyglot.db"),
    phase: Annotated[
        Literal["run", "generate", "evaluate"],
        typer.Option("--phase", help="Benchmark phase: run, generate, or evaluate"),
    ] = "run",
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for generated task artifacts"),
    ] = None,
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
    fetch_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-artifacts/--no-fetch-artifacts",
            help="Rsync the artifact directory back when --on bluevela",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Run the Aider Polyglot benchmark through mcode's harness."""

    from mcode.bench.aider_polyglot import default_benchmark_root, supported_languages

    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)

    if exercise is not None and language == "all":
        raise typer.BadParameter("--exercise requires a concrete --language")
    if task_ids is not None and exercise is not None:
        raise typer.BadParameter("--task-ids cannot be combined with --exercise")
    if language != "all" and language not in supported_languages():
        known = ", ".join(supported_languages())
        raise typer.BadParameter(f"unknown --language {language!r}; expected one of {known}, all")

    selected_root = benchmark_root if benchmark_root is not None else default_benchmark_root()
    selected_task_ids = task_ids
    if exercise is not None:
        selected_task_ids = f"{language}/{exercise}"

    if on == "bluevela":
        argv = _aider_polyglot_cli_args(
            model=model,
            backend=backend,
            loop_budget=loop_budget,
            retry_loop_budget=retry_loop_budget,
            temperature=temperature,
            seed=seed,
            benchmark_root=selected_root,
            language=language,
            exercise=exercise,
            limit=limit,
            no_retry=no_retry,
            task_ids=task_ids,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        if json_mode:
            argv.append("--json")
        _run_bluevela_benchmark(
            command="aider-polyglot",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)
    if shards and shards > 1:
        _run_sharded_benchmark(
            command="aider-polyglot",
            base_argv=_aider_polyglot_cli_args(
                model=model,
                backend=backend,
                loop_budget=loop_budget,
                retry_loop_budget=retry_loop_budget,
                temperature=temperature,
                seed=seed,
                benchmark_root=selected_root,
                language=language,
                exercise=exercise,
                limit=limit,
                no_retry=no_retry,
                task_ids=task_ids,
                phase=phase,
                artifact_dir=resolved_artifact_dir,
            ),
            shards=shards,
            db=db,
            benchmark="aider-polyglot",
            backend=backend,
            model=model,
            loop_budget=loop_budget + (0 if no_retry else retry_loop_budget),
            timeout_s=300,
            json_mode=json_mode,
        )
        return
    if shard_count and shard_count > 1 and db == Path("experiments/results/aider-polyglot.db"):
        typer.echo(
            "Note: when running shards in parallel, use a unique --db per shard to avoid SQLite "
            "locks.",
            err=True,
        )

    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        loop_budget=loop_budget,
        temperature=temperature,
        seed=seed,
        timeout_s=300,
        phase=phase,
        artifact_dir=resolved_artifact_dir,
        aider_polyglot_root=selected_root,
        aider_polyglot_language=language,
        aider_polyglot_retry=not no_retry,
        aider_polyglot_retry_loop_budget=retry_loop_budget,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
    )
    _run_single_benchmark(
        benchmark="aider-polyglot",
        config=config,
        db=db,
        limit=limit,
        task_ids=selected_task_ids,
        backend=backend,
        model=model,
        loop_budget=loop_budget + (0 if no_retry else retry_loop_budget),
        timeout_s=300,
    )


@bench_app.command("suite")
def bench_suite(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "openai",
    loop_budget: Annotated[
        int,
        typer.Option("--loop-budget", min=1, help="Shared generation loop budget"),
    ] = 15,
    retry_loop_budget: Annotated[
        int,
        typer.Option(
            "--retry-loop-budget",
            min=1,
            help="Aider Polyglot retry loop budget inside the suite",
        ),
    ] = 8,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per SWE-bench eval attempt"),
    ] = 300,
    mem_limit: Annotated[
        str, typer.Option("--mem-limit", help="Eval container memory limit")
    ] = "8g",
    pids_limit: Annotated[
        int,
        typer.Option("--pids-limit", min=64, help="Eval container process limit"),
    ] = 512,
    cpu_limit: Annotated[
        float | None,
        typer.Option("--cpu-limit", help="Cap each eval container at N cores"),
    ] = None,
    check_image_digests: Annotated[
        bool,
        typer.Option(
            "--check-image-digests/--no-check-image-digests",
            help="Check registry digests before reusing cached task images",
        ),
    ] = False,
    suite_file: Annotated[
        Path | None,
        typer.Option("--suite-file", help="JSON suite manifest (default: bundled suite)"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/suite.db"
    ),
    phase: Annotated[
        Literal["run", "generate", "evaluate"],
        typer.Option("--phase", help="Benchmark phase: run, generate, or evaluate"),
    ] = "run",
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for generated task artifacts"),
    ] = None,
    shards: Annotated[
        int | None,
        typer.Option("--shards", min=1, help="Run N shard workers and merge the DB automatically"),
    ] = None,
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Manual shard mode: total shard count"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Manual shard mode: shard index"),
    ] = None,
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
    fetch_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-artifacts/--no-fetch-artifacts",
            help="Rsync the artifact directory back when --on bluevela",
        ),
    ] = False,
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    json_mode: JsonFlag = False,
 ) -> None:
    """Run the bundled mixed benchmark suite through the shared phase runner."""
    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)
    if on == "bluevela":
        argv = _suite_cli_args(
            model=model,
            backend=backend,
            loop_budget=loop_budget,
            retry_loop_budget=retry_loop_budget,
            timeout_s=timeout_s,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            cpu_limit=cpu_limit,
            suite_file=suite_file,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
            diagnostic_traces=diagnostic_traces,
            check_image_digests=check_image_digests,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        if json_mode:
            argv.append("--json")
        _run_bluevela_benchmark(
            command="suite",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)
    if shards and shards > 1:
        _run_sharded_benchmark(
            command="suite",
            base_argv=_suite_cli_args(
                model=model,
                backend=backend,
                loop_budget=loop_budget,
                retry_loop_budget=retry_loop_budget,
                timeout_s=timeout_s,
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                cpu_limit=cpu_limit,
                suite_file=suite_file,
                phase=phase,
                artifact_dir=resolved_artifact_dir,
                diagnostic_traces=diagnostic_traces,
                check_image_digests=check_image_digests,
            ),
            shards=shards,
            db=db,
            benchmark="suite",
            backend=backend,
            model=model,
            loop_budget=loop_budget + retry_loop_budget,
            timeout_s=timeout_s,
            json_mode=json_mode,
            merge_mode="full_db",
        )
        return
    suite_name = suite_file.stem if suite_file is not None else "default-suite"
    manifest = load_suite_manifest(suite_file)
    for entry in manifest.entries:
        _run_suite_entry(
            suite_name=suite_name,
            entry=entry,
            model=model,
            backend=backend,
            loop_budget=loop_budget,
            retry_loop_budget=retry_loop_budget,
            timeout_s=timeout_s,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            cpu_limit=cpu_limit,
            check_image_digests=check_image_digests,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
            db=db,
            shard_count=shard_count,
            shard_index=shard_index,
            diagnostic_traces=diagnostic_traces,
        )


@bench_app.command("smoke")
def bench_smoke(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "openai",
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/smoke-16.db"
    ),
    phase: Annotated[
        Literal["run", "generate", "evaluate"],
        typer.Option("--phase", help="Benchmark phase: run, generate, or evaluate"),
    ] = "run",
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for generated task artifacts"),
    ] = None,
    mem_limit: Annotated[
        str, typer.Option("--mem-limit", help="Eval container memory limit")
    ] = "8g",
    shards: Annotated[
        int | None,
        typer.Option("--shards", min=1, help="Run N shard workers and merge the DB automatically"),
    ] = None,
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Manual shard mode: total shard count"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Manual shard mode: shard index"),
    ] = None,
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
    fetch_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-artifacts/--no-fetch-artifacts",
            help="Rsync the artifact directory back when --on bluevela",
        ),
    ] = False,
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    cpu_limit: Annotated[
        float | None,
        typer.Option(
            "--cpu-limit",
            help=(
                "Cap each eval container at N cores (cgroup cpu_quota). "
                "Default: unlimited. Use --on bluevela where login-node admins "
                "kill processes >100 cores."
            ),
            envvar="MCODE_SWEBENCH_CPU_LIMIT",
        ),
    ] = None,
    check_image_digests: Annotated[
        bool,
        typer.Option(
            "--check-image-digests/--no-check-image-digests",
            help="Check registry digests before reusing cached task images",
            envvar="MCODE_SWEBENCH_CHECK_IMAGE_DIGESTS",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """16-task SWE-bench Verified diagnostic slice (astropy smoke + 6 projects).

    Fixed slice used for cross-model comparison. Calls `swebench-lite` under
    the hood with a bundled task-id list and sensible defaults.
    """
    import importlib.resources as ir

    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)

    if on == "bluevela":
        argv = [
            "--model",
            model,
            "--backend",
            backend,
            "--mem-limit",
            mem_limit,
            "--phase",
            phase,
            "--artifact-dir",
            str(resolved_artifact_dir),
        ]
        if diagnostic_traces:
            argv.append("--diagnostic-traces")
        if not check_image_digests:
            argv.append("--no-check-image-digests")
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _append_option(argv, "--cpu-limit", cpu_limit)
        if json_mode:
            argv.append("--json")
        _run_bluevela_benchmark(
            command="smoke",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)

    task_ids_resource = ir.files("mcode.bench.fixtures").joinpath("smoke-16.txt")
    with ir.as_file(task_ids_resource) as task_ids_file:
        bench_swebench_lite(
            model=model,
            backend=backend,
            loop_budget=15,
            temperature=None,
            seed=None,
            timeout_s=300,
            split="test",
            arch="auto",
            namespace="swebench",
            max_workers=4,
            force_rebuild=False,
            mem_limit=mem_limit,
            pids_limit=512,
            cpu_limit=cpu_limit,
            check_image_digests=check_image_digests,
            shards=shards,
            shard_count=shard_count,
            shard_index=shard_index,
            db=db,
            phase=phase,
            artifact_dir=resolved_artifact_dir,
            limit=None,
            n_samples=1,
            task_ids=str(task_ids_file),
            dataset="princeton-nlp/SWE-bench_Verified",
            diagnostic_traces=diagnostic_traces,
            json_mode=json_mode,
        )
