from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mcode.bench.results import ResultsDB, RunSummary
from mcode.bench.runner import BenchConfig, BenchmarkRunner

app = typer.Typer(add_completion=False, no_args_is_help=True)
bench_app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


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


def _parse_bool(v: str | None) -> bool | None:
    if v is None:
        return None
    lowered = v.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter("Expected a boolean (true/false).")


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


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show Mellea INFO logs")] = False,
) -> None:
    """mCode benchmarking harness."""
    _configure_mellea_logging(verbose)


@app.command("results")
def results(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path")] = Path(
        "experiments/results/results.db"
    ),
    benchmark: Annotated[str | None, typer.Option("--benchmark")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    samples: Annotated[int | None, typer.Option("--samples", min=1)] = None,
    debug_iters: Annotated[int | None, typer.Option("--debug-iters", min=0)] = None,
    timeout_s: Annotated[int | None, typer.Option("--timeout", min=1)] = None,
    compare_samples: Annotated[
        bool,
        typer.Option("--compare-samples", help="Group results by sample count"),
    ] = False,
    retrieval: Annotated[
        str | None,
        typer.Option("--retrieval", help="Filter by retrieval flag (true/false)"),
    ] = None,
) -> None:
    """Query pass rates from the results DB."""
    retrieval_bool = _parse_bool(retrieval)
    rdb = ResultsDB(db)

    if compare_samples:
        rows = rdb.pass_rates_grouped(
            benchmark=benchmark,
            model_id=model,
            backend_name=backend,
            max_debug_iterations=debug_iters,
            timeout_s=timeout_s,
            group_by=("backend_name", "max_debug_iterations", "timeout_s", "samples"),
            retrieval=retrieval_bool,
            samples=samples,
        )
        table = Table(title="Pass rates by samples")
        table.add_column("benchmark")
        table.add_column("backend")
        table.add_column("model")
        table.add_column("debug", justify="right")
        table.add_column("timeout", justify="right")
        table.add_column("samples", justify="right")
        table.add_column("retrieval", justify="center")
        table.add_column("total", justify="right")
        table.add_column("passed", justify="right")
        table.add_column("pass_rate", justify="right")
        for row in rows:
            table.add_row(
                row["benchmark"],
                row["backend_name"],
                row["model_id"],
                str(row["max_debug_iterations"]),
                str(row["timeout_s"]),
                str(row["samples"]),
                "on" if row["retrieval"] else "off",
                str(row["total"]),
                str(row["passed"]),
                f"{row['pass_rate']:.1%}",
            )
        console.print(table)
        return

    rows = rdb.pass_rates_grouped(
        benchmark=benchmark,
        model_id=model,
        backend_name=backend,
        max_debug_iterations=debug_iters,
        timeout_s=timeout_s,
        group_by=(),
        retrieval=retrieval_bool,
        samples=samples,
    )
    table = Table(title="Pass rates (per run)")
    table.add_column("run_id", justify="right")
    table.add_column("timestamp")
    table.add_column("benchmark")
    table.add_column("backend")
    table.add_column("model")
    table.add_column("samples", justify="right")
    table.add_column("debug", justify="right")
    table.add_column("timeout", justify="right")
    table.add_column("retrieval", justify="center")
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
            str(row["samples"]),
            str(row["max_debug_iterations"]),
            str(row["timeout_s"]),
            "on" if row["retrieval"] else "off",
            str(row["total"]),
            str(row["passed"]),
            f"{row['pass_rate']:.1%}",
        )
    console.print(table)


app.add_typer(bench_app, name="bench")


def _print_run_summary(
    *,
    summary: RunSummary,
    benchmark: str,
    backend: str,
    model: str,
    samples: int,
    debug_iters: int,
    timeout_s: int,
    retrieval: bool,
) -> None:
    table = Table(title="Run summary")
    table.add_column("run_id", justify="right")
    table.add_column("benchmark")
    table.add_column("backend")
    table.add_column("model")
    table.add_column("samples", justify="right")
    table.add_column("debug", justify="right")
    table.add_column("timeout", justify="right")
    table.add_column("retrieval", justify="center")
    table.add_column("total", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_row(
        str(summary.run_id),
        benchmark,
        backend,
        model,
        str(samples),
        str(debug_iters),
        str(timeout_s),
        "on" if retrieval else "off",
        str(summary.total),
        str(summary.passed),
        f"{summary.pass_rate:.1%}",
    )
    console.print(table)


def _bench_common(
    benchmark: str,
    backend: str,
    model: str,
    samples: int,
    debug_iters: int,
    timeout_s: int,
    retrieval: bool,
    sandbox: str,
    shard_count: int | None,
    shard_index: int | None,
    db: Path,
    limit: int | None,
) -> None:
    sandbox_name = sandbox.strip().lower()
    if sandbox_name not in {"docker", "process"}:
        raise typer.BadParameter("Unknown --sandbox. Use docker or process.")

    shard_count, shard_index = _validate_shards(shard_count=shard_count, shard_index=shard_index)

    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        samples=samples,
        retrieval=retrieval,
        max_debug_iterations=debug_iters,
        timeout_s=timeout_s,
        sandbox=sandbox_name,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
    )
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db))
    summary = runner.run_benchmark(benchmark, limit=limit)
    _print_run_summary(
        summary=summary,
        benchmark=benchmark,
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=retrieval,
    )


@bench_app.command("humaneval")
def bench_humaneval(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, help="Attempts per task; stop early on pass"),
    ] = 1,
    debug_iters: Annotated[
        int,
        typer.Option("--debug-iters", min=0, help="Fix attempts after a failed run"),
    ] = 0,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per sandbox execution attempt"),
    ] = 60,
    retrieval: Annotated[
        bool,
        typer.Option("--retrieval/--no-retrieval", help="Reserved (no effect yet)"),
    ] = False,
    sandbox: Annotated[
        str,
        typer.Option(
            "--sandbox",
            help="Execution sandbox for code evaluation (docker or process).",
        ),
    ] = "docker",
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Total shards for parallel runs"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Shard index (0..shard-count-1)"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/results.db"
    ),
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
) -> None:
    _bench_common(
        benchmark="humaneval",
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=retrieval,
        sandbox=sandbox,
        shard_count=shard_count,
        shard_index=shard_index,
        db=db,
        limit=limit,
    )


@bench_app.command("mbpp")
def bench_mbpp(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, help="Attempts per task; stop early on pass"),
    ] = 1,
    debug_iters: Annotated[
        int,
        typer.Option("--debug-iters", min=0, help="Fix attempts after a failed run"),
    ] = 0,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per sandbox execution attempt"),
    ] = 60,
    retrieval: Annotated[
        bool,
        typer.Option("--retrieval/--no-retrieval", help="Reserved (no effect yet)"),
    ] = False,
    sandbox: Annotated[
        str,
        typer.Option(
            "--sandbox",
            help="Execution sandbox for code evaluation (docker or process).",
        ),
    ] = "docker",
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Total shards for parallel runs"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Shard index (0..shard-count-1)"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/results.db"
    ),
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
) -> None:
    _bench_common(
        benchmark="mbpp",
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=retrieval,
        sandbox=sandbox,
        shard_count=shard_count,
        shard_index=shard_index,
        db=db,
        limit=limit,
    )


@bench_app.command("swebench-lite")
def bench_swebench_lite(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, help="Attempts per task; stop early on pass"),
    ] = 1,
    debug_iters: Annotated[
        int,
        typer.Option("--debug-iters", min=0, help="Fix attempts after a failed run"),
    ] = 0,
    timeout_s: Annotated[
        int,
        typer.Option("--timeout", min=1, help="Seconds per SWE-bench eval attempt"),
    ] = 1800,
    split: Annotated[str, typer.Option("--split", help="Dataset split (dev/test)")] = "test",
    arch: Annotated[
        str,
        typer.Option(
            "--arch",
            help=(
                "Image arch: auto/x86_64/arm64 (auto prefers x86_64 for prebuilt images)."
            ),
        ),
    ] = "auto",
    namespace: Annotated[
        str,
        typer.Option(
            "--namespace",
            help=(
                "Prebuilt image namespace (default: swebench); set to \"\" to build locally."
            ),
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
    shard_count: Annotated[
        int | None,
        typer.Option("--shard-count", min=1, help="Total shards for parallel runs"),
    ] = None,
    shard_index: Annotated[
        int | None,
        typer.Option("--shard-index", min=0, help="Shard index (0..shard-count-1)"),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite results DB path")] = Path(
        "experiments/results/results.db"
    ),
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
) -> None:
    shard_count, shard_index = _validate_shards(shard_count=shard_count, shard_index=shard_index)

    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        samples=samples,
        retrieval=False,
        max_debug_iterations=debug_iters,
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
    )
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db))
    summary = runner.run_benchmark("swebench-lite", limit=limit)
    _print_run_summary(
        summary=summary,
        benchmark="swebench-lite",
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=False,
    )
