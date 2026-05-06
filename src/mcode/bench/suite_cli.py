from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from mcode.bench.runner import BenchConfig
from mcode.bench.shards import _run_sharded_benchmark
from mcode.bench.suite import SuiteEntry, load_suite_manifest, task_ids_arg
from mcode.cli_shared import (
    _append_option,
    _resolve_artifact_dir,
    _validate_shard_options,
)
from mcode.ui.flags import JsonFlag


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


def _suite_entry_loop_budget(entry: SuiteEntry, *, loop_budget: int, retry_loop_budget: int) -> int:
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
    json_mode: bool = False,
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
    from mcode.bench.cli import _run_single_benchmark

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
        json_mode=json_mode,
    )


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
        from mcode.bench.cli import _run_bluevela_benchmark

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
            json_mode=json_mode,
        )


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
    eval_repair_attempts: Annotated[
        int,
        typer.Option(
            "--eval-repair-attempts",
            min=0,
            help="Retry failed official SWE-bench evaluations with deterministic eval feedback",
        ),
    ] = 0,
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
        if eval_repair_attempts:
            _append_option(argv, "--eval-repair-attempts", eval_repair_attempts)
        _append_option(argv, "--shards", shards)
        _append_option(argv, "--shard-count", shard_count)
        _append_option(argv, "--shard-index", shard_index)
        _append_option(argv, "--cpu-limit", cpu_limit)
        if json_mode:
            argv.append("--json")
        from mcode.bench.cli import _run_bluevela_benchmark

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
        from mcode.bench.cli import bench_swebench_lite

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
            sampling="none",
            sampling_budget=None,
            selection_attempts=1,
            eval_repair_attempts=eval_repair_attempts,
            task_ids=str(task_ids_file),
            dataset="princeton-nlp/SWE-bench_Verified",
            diagnostic_traces=diagnostic_traces,
            json_mode=json_mode,
        )


def register_suite_commands(app: typer.Typer) -> None:
    app.command("suite")(bench_suite)
    app.command("smoke")(bench_smoke)
