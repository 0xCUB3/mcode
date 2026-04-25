from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from mcode.bench.results import (
    ResultsDB,
    RunSummary,
    merge_shard_dbs,
)
from mcode.bench.results import (
    export_csv as export_csv_results,
)
from mcode.bench.runner import BenchConfig, BenchmarkRunner

app = typer.Typer(add_completion=False, no_args_is_help=True)
bench_app = typer.Typer(add_completion=False, no_args_is_help=True)
deps_app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
DEFAULT_DB_PATH = Path("experiments/results/results.db")


def _configure_mellea_logging(verbose: bool) -> None:
    try:
        import logging

        from mellea.helpers.fancy_logger import FancyLogger

        logger = FancyLogger.get_logger()
        level = logging.INFO if verbose else logging.WARNING
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)
    except Exception:
        return


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

    with tempfile.TemporaryDirectory(prefix="mcode-results-") as td:
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


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show Mellea INFO logs")] = False,
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
) -> None:
    """Query pass rates from the results DB."""
    group_by_config = ("backend_name", "timeout_s", "loop_budget")

    db_paths = _expand_db_paths(db=db, db_glob=db_glob, db_dir=db_dir)
    with _open_results_view(db_paths) as rdb:
        if compare_configs:
            if time_metrics:
                rows = rdb.run_metrics_grouped(
                    benchmark=benchmark,
                    model_id=model,
                    backend_name=backend,
                    timeout_s=timeout_s,
                    group_by=group_by_config,
                    loop_budget=loop_budget,
                )
                rows = sorted(
                    rows,
                    key=lambda r: (r["solves_per_hour"], r["pass_rate"]),
                    reverse=True,
                )
                table = Table(title="Pass rates + time (grouped)")
                table.add_column("benchmark")
                table.add_column("backend")
                table.add_column("model")
                table.add_column("budget", justify="right")
                table.add_column("timeout", justify="right")
                table.add_column("runs", justify="right")
                table.add_column("total", justify="right")
                table.add_column("passed", justify="right")
                table.add_column("timed_out", justify="right")
                table.add_column("timeout_rate", justify="right")
                table.add_column("pass_rate", justify="right")
                table.add_column("avg_s", justify="right")
                table.add_column("p95_s", justify="right")
                table.add_column("tok/task", justify="right")
                table.add_column("sec/solve", justify="right")
                table.add_column("solves/hr", justify="right")
                for row in rows:
                    table.add_row(
                        row["benchmark"],
                        row["backend_name"],
                        row["model_id"],
                        str(row.get("loop_budget", "")),
                        str(row["timeout_s"]),
                        str(row.get("runs", "")),
                        str(row["total"]),
                        str(row["passed"]),
                        str(row.get("timed_out", 0)),
                        f"{row.get('timeout_rate', 0.0):.1%}",
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
                group_by=group_by_config,
                loop_budget=loop_budget,
            )
            table = Table(title="Pass rates by config")
            table.add_column("benchmark")
            table.add_column("backend")
            table.add_column("model")
            table.add_column("budget", justify="right")
            table.add_column("timeout", justify="right")
            table.add_column("total", justify="right")
            table.add_column("passed", justify="right")
            table.add_column("pass_rate", justify="right")
            for row in rows:
                table.add_row(
                    row["benchmark"],
                    row["backend_name"],
                    row["model_id"],
                    str(row.get("loop_budget", "")),
                    str(row["timeout_s"]),
                    str(row["total"]),
                    str(row["passed"]),
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
                group_by=(),
                loop_budget=loop_budget,
            )
            rows = sorted(rows, key=lambda r: (r["solves_per_hour"], r["pass_rate"]), reverse=True)
            table = Table(title="Pass rates + time (per run)")
            table.add_column("run_id", justify="right")
            table.add_column("timestamp")
            table.add_column("benchmark")
            table.add_column("backend")
            table.add_column("model")
            table.add_column("budget", justify="right")
            table.add_column("timeout", justify="right")
            table.add_column("total", justify="right")
            table.add_column("passed", justify="right")
            table.add_column("timed_out", justify="right")
            table.add_column("timeout_rate", justify="right")
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
                    row["backend_name"],
                    row["model_id"],
                    str(row.get("loop_budget", "")),
                    str(row["timeout_s"]),
                    str(row["total"]),
                    str(row["passed"]),
                    str(row.get("timed_out", 0)),
                    f"{row.get('timeout_rate', 0.0):.1%}",
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
            group_by=(),
            loop_budget=loop_budget,
        )
        table = Table(title="Pass rates (per run)")
        table.add_column("run_id", justify="right")
        table.add_column("timestamp")
        table.add_column("benchmark")
        table.add_column("backend")
        table.add_column("model")
        table.add_column("budget", justify="right")
        table.add_column("timeout", justify="right")
        table.add_column("total", justify="right")
        table.add_column("passed", justify="right")
        table.add_column("pass_rate", justify="right")
        for row in rows:
            table.add_row(
                str(row["run_id"]),
                row["timestamp"],
                row["benchmark"],
                row["backend_name"],
                row["model_id"],
                str(row.get("loop_budget", "")),
                str(row["timeout_s"]),
                str(row["total"]),
                str(row["passed"]),
                f"{row['pass_rate']:.1%}",
            )
            console.print(table)


@app.command("compare")
def compare(
    baseline_dir: Annotated[Path, typer.Option("--baseline-dir", help="Baseline DB directory")],
    candidate_dir: Annotated[Path, typer.Option("--candidate-dir", help="Candidate DB directory")],
    task_ids: Annotated[
        str | None,
        typer.Option(
            "--task-ids",
            help="Comma-separated task IDs to compare (or path to JSON/text file)",
        ),
    ] = None,
) -> None:
    from mcode.bench.compare import compare_run_dirs

    console.print(
        compare_run_dirs(
            baseline_dir=str(baseline_dir),
            candidate_dir=str(candidate_dir),
            task_ids=_parse_task_ids(task_ids),
        )
    )


def _config_label(r: dict) -> str:
    parts = [
        str(r.get("benchmark", "")),
        f"{r.get('backend_name', '')}:{r.get('model_id', '')}",
        f"budget={r.get('loop_budget', '')}",
        f"timeout={r.get('timeout_s', '')}",
    ]
    if r.get("strategy") and r["strategy"] != "repair":
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
    loop_budget: Annotated[int | None, typer.Option("--loop-budget", min=1)] = None,
    timeout_s: Annotated[int | None, typer.Option("--timeout", min=1)] = None,
    per_run: Annotated[
        bool, typer.Option("--per-run", help="Plot each run separately (vs grouped)")
    ] = False,
) -> None:
    """Generate a lightweight HTML report (Plotly) for pass rate vs time-to-solve."""
    group_by_config = ("backend_name", "timeout_s", "loop_budget")
    group_by = () if per_run else group_by_config

    db_paths = _expand_db_paths(db=db, db_glob=db_glob, db_dir=db_dir)
    with _open_results_view(db_paths) as rdb:
        rows = rdb.run_metrics_grouped(
            benchmark=benchmark,
            model_id=model,
            backend_name=backend,
            timeout_s=timeout_s,
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
    console.print(
        f"exported dbs={report['dbs']} runs={report['runs']} "
        f"task_results={report['task_results']}\n"
        f"runs_csv={report['runs_csv']}\n"
        f"task_results_csv={report['task_results_csv']}"
    )


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


def _merge_into_results_db(*, db: Path, shard_paths: list[Path]) -> RunSummary:
    with tempfile.TemporaryDirectory(prefix="mcode-merge-") as td:
        merged = Path(td) / "merged.db"
        merge_shard_dbs(out_path=merged, shard_paths=shard_paths, force=True)
        with ResultsDB(db) as out_db:
            out_db.merge_from([merged])
    return _latest_run_summary(db)


def _stream_shard_output(
    *,
    proc: subprocess.Popen[str],
    prefix: str,
    log_path: Path,
    echo_lock: threading.Lock,
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
                with echo_lock:
                    typer.echo(f"[{prefix}] {text}")

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
) -> None:
    run_dir = db.parent / f"{db.stem}-shards" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"▶ sharded run command={command} shards={shards} out={db} artifacts={run_dir}",
        err=True,
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    echo_lock = threading.Lock()
    procs: list[tuple[int, subprocess.Popen[str], Path, Path, threading.Thread]] = []
    failed: list[tuple[int, int, Path]] = []
    shard_paths: list[Path] = []

    try:
        for shard_index in range(shards):
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
            typer.echo(
                f"▶ shard {shard_index + 1}/{shards} db={shard_db} log={shard_log}",
                err=True,
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
                prefix=f"shard {shard_index}",
                log_path=shard_log,
                echo_lock=echo_lock,
            )
            procs.append((shard_index, proc, shard_db, shard_log, thread))

        for shard_index, proc, shard_db, shard_log, thread in procs:
            rc = proc.wait()
            thread.join()
            if rc == 0:
                shard_paths.append(shard_db)
                typer.echo(f"✓ shard {shard_index + 1}/{shards} finished", err=True)
                continue
            failed.append((shard_index, rc, shard_log))
            typer.echo(
                f"✗ shard {shard_index + 1}/{shards} failed exit={rc} log={shard_log}",
                err=True,
            )
    except KeyboardInterrupt:
        for _, proc, _, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        raise

    if failed:
        for shard_index, rc, shard_log in failed:
            typer.echo(
                f"failed shard={shard_index} exit={rc} log={shard_log}",
                err=True,
            )
        raise typer.Exit(1)

    summary = _merge_into_results_db(db=db, shard_paths=shard_paths)
    typer.echo(f"✓ merged shards into {db}", err=True)
    _print_run_summary(
        summary=summary,
        benchmark=benchmark,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
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
    parsed_task_ids = _parse_task_ids(task_ids)
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db))
    summary = runner.run_benchmark(benchmark, limit=limit, task_ids=parsed_task_ids)
    _print_run_summary(
        summary=summary,
        benchmark=benchmark,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )


def _run_bluevela_benchmark(
    *,
    command: str,
    argv: list[str],
    model: str,
    db: Path,
    fetch_db: bool,
) -> None:
    from mcode.bench.remote import RemoteBenchError, run_bench_on_bluevela

    try:
        rc = run_bench_on_bluevela(
            bench_argv=[command, *argv],
            model=model,
            local_db=db,
            fetch_db=fetch_db,
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
    limit: int | None,
    n_samples: int,
    sampling: str,
    sampling_budget: int | None,
    selection_attempts: int,
    task_ids: str | None,
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
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--sampling-budget", sampling_budget)
    if selection_attempts != 1:
        _append_option(argv, "--selection-attempts", selection_attempts)
    _append_option(argv, "--task-ids", task_ids)
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
    limit: int | None,
    n_samples: int,
    sampling: str,
    sampling_budget: int | None,
    selection_attempts: int,
    task_ids: str | None,
    dataset: str,
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
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--sampling-budget", sampling_budget)
    if selection_attempts != 1:
        _append_option(argv, "--selection-attempts", selection_attempts)
    _append_option(argv, "--task-ids", task_ids)
    if force_rebuild:
        argv.append("--force-rebuild")
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
    ]
    _append_option(argv, "--temperature", temperature)
    _append_option(argv, "--seed", seed)
    _append_option(argv, "--exercise", exercise)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--task-ids", task_ids)
    if no_retry:
        argv.append("--no-retry")
    return argv


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
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    n_samples: Annotated[
        int,
        typer.Option("--n-samples", min=1, help="Outer attempts or sampling budget"),
    ] = 1,
    sampling: Annotated[
        Literal["none", "rejection", "repair", "multiturn", "sofai"],
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
            limit=limit,
            n_samples=n_samples,
            sampling=sampling,
            sampling_budget=sampling_budget,
            selection_attempts=selection_attempts,
            task_ids=task_ids,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _run_bluevela_benchmark(
            command="swebench-live",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
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
                limit=limit,
                n_samples=n_samples,
                sampling=sampling,
                sampling_budget=sampling_budget,
                selection_attempts=selection_attempts,
                task_ids=task_ids,
            ),
            shards=shards,
            db=db,
            benchmark="swebench-live",
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
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
        swebench_split=split,
        swebench_mem_limit=mem_limit,
        swebench_pids_limit=pids_limit,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
        n_samples=n_samples,
        sampling_strategy=sampling,
        sampling_budget=sampling_budget,
        selection_attempts=selection_attempts,
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
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    n_samples: Annotated[
        int,
        typer.Option("--n-samples", min=1, help="Outer attempts or sampling budget"),
    ] = 1,
    sampling: Annotated[
        Literal["none", "rejection", "repair", "multiturn", "sofai"],
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
            limit=limit,
            n_samples=n_samples,
            sampling=sampling,
            sampling_budget=sampling_budget,
            selection_attempts=selection_attempts,
            task_ids=task_ids,
            dataset=dataset,
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _run_bluevela_benchmark(
            command="swebench-lite",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
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
                limit=limit,
                n_samples=n_samples,
                sampling=sampling,
                sampling_budget=sampling_budget,
                selection_attempts=selection_attempts,
                task_ids=task_ids,
                dataset=dataset,
            ),
            shards=shards,
            db=db,
            benchmark="swebench-lite",
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
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
        swebench_split=split,
        swebench_namespace=_optional_str(namespace),
        swebench_arch=None if arch == "auto" else arch,
        swebench_max_workers=max_workers,
        swebench_force_rebuild=force_rebuild,
        swebench_mem_limit=mem_limit,
        swebench_pids_limit=pids_limit,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
        n_samples=n_samples,
        sampling_strategy=sampling,
        sampling_budget=sampling_budget,
        selection_attempts=selection_attempts,
        swebench_dataset=dataset,
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
    on: Annotated[
        str,
        typer.Option("--on", help="Where to run the bench: local or bluevela"),
    ] = "local",
    fetch_db: Annotated[
        bool,
        typer.Option("--fetch-db/--no-fetch-db", help="Rsync DB back when --on bluevela"),
    ] = True,
) -> None:
    """Run the Aider Polyglot benchmark through mcode's harness."""

    from mcode.bench.aider_polyglot import default_benchmark_root, supported_languages
    shards, shard_count, shard_index = _validate_shard_options(
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
    )

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
        )
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _run_bluevela_benchmark(
            command="aider-polyglot",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
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
            ),
            shards=shards,
            db=db,
            benchmark="aider-polyglot",
            backend=backend,
            model=model,
            loop_budget=loop_budget + (0 if no_retry else retry_loop_budget),
            timeout_s=300,
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


@bench_app.command("smoke")
def bench_smoke(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "openai",
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/smoke-16.db"
    ),
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

    if on == "bluevela":
        argv = [
            "--model",
            model,
            "--backend",
            backend,
            "--mem-limit",
            mem_limit,
        ]
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _run_bluevela_benchmark(
            command="smoke",
            argv=argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
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
            shards=shards,
            shard_count=shard_count,
            shard_index=shard_index,
            db=db,
            limit=None,
            n_samples=1,
            task_ids=str(task_ids_file),
            dataset="princeton-nlp/SWE-bench_Verified",
        )
