from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from mcode.bench.results import ResultsDB
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


def _parse_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    lowered = v.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter("Expected a boolean (true/false).")


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
    benchmark: Annotated[Optional[str], typer.Option("--benchmark")] = None,
    model: Annotated[Optional[str], typer.Option("--model")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend")] = None,
    samples: Annotated[Optional[int], typer.Option("--samples", min=1)] = None,
    debug_iters: Annotated[Optional[int], typer.Option("--debug-iters", min=0)] = None,
    timeout_s: Annotated[Optional[int], typer.Option("--timeout", min=1)] = None,
    compare_samples: Annotated[bool, typer.Option("--compare-samples")] = False,
    retrieval: Annotated[Optional[str], typer.Option("--retrieval")] = None,
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


def _bench_common(
    benchmark: str,
    backend: str,
    model: str,
    samples: int,
    debug_iters: int,
    timeout_s: int,
    retrieval: bool,
    db: Path,
    limit: Optional[int],
) -> None:
    config = BenchConfig(
        backend_name=backend,
        model_id=model,
        samples=samples,
        retrieval=retrieval,
        max_debug_iterations=debug_iters,
        timeout_s=timeout_s,
    )
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db))
    summary = runner.run_benchmark(benchmark, limit=limit)

    table = Table(title="Run summary")
    table.add_column("run_id", justify="right")
    table.add_column("benchmark")
    table.add_column("model")
    table.add_column("samples", justify="right")
    table.add_column("debug", justify="right")
    table.add_column("retrieval", justify="center")
    table.add_column("total", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_row(
        str(summary.run_id),
        benchmark,
        model,
        str(samples),
        str(debug_iters),
        "on" if retrieval else "off",
        str(summary.total),
        str(summary.passed),
        f"{summary.pass_rate:.1%}",
    )
    console.print(table)


@bench_app.command("humaneval")
def bench_humaneval(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    samples: Annotated[int, typer.Option("--samples", min=1)] = 1,
    debug_iters: Annotated[int, typer.Option("--debug-iters", min=0)] = 0,
    timeout_s: Annotated[int, typer.Option("--timeout", min=1)] = 60,
    retrieval: Annotated[bool, typer.Option("--retrieval/--no-retrieval")] = False,
    db: Annotated[Path, typer.Option("--db")] = Path("experiments/results/results.db"),
    limit: Annotated[Optional[int], typer.Option("--limit", min=1)] = None,
) -> None:
    _bench_common(
        benchmark="humaneval",
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=retrieval,
        db=db,
        limit=limit,
    )


@bench_app.command("mbpp")
def bench_mbpp(
    model: Annotated[str, typer.Option("--model", help="Mellea model id")],
    backend: Annotated[str, typer.Option("--backend", help="Mellea backend name")] = "ollama",
    samples: Annotated[int, typer.Option("--samples", min=1)] = 1,
    debug_iters: Annotated[int, typer.Option("--debug-iters", min=0)] = 0,
    timeout_s: Annotated[int, typer.Option("--timeout", min=1)] = 60,
    retrieval: Annotated[bool, typer.Option("--retrieval/--no-retrieval")] = False,
    db: Annotated[Path, typer.Option("--db")] = Path("experiments/results/results.db"),
    limit: Annotated[Optional[int], typer.Option("--limit", min=1)] = None,
) -> None:
    _bench_common(
        benchmark="mbpp",
        backend=backend,
        model=model,
        samples=samples,
        debug_iters=debug_iters,
        timeout_s=timeout_s,
        retrieval=retrieval,
        db=db,
        limit=limit,
    )


@bench_app.command("swebench-lite")
def bench_swebench_lite() -> None:  # pragma: no cover
    raise typer.BadParameter("SWE-Bench Lite support is deferred; Phase 1 focuses on HumanEval+MBPP.")
