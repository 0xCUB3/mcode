from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from mcode.bench.artifacts_cli import register_artifact_commands
from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.bench.shards import (
    SHARDED_INFRA_EXIT_CODE,
    _is_retryable_infra_exception,
    _print_run_summary,
    _run_sharded_benchmark,
)
from mcode.bench.suite_cli import register_suite_commands
from mcode.cli_shared import (
    DEFAULT_DB_PATH,
    _append_option,
    _optional_str,
    _parse_task_ids,
    _resolve_artifact_dir,
    _validate_sampling,
    _validate_shard_options,
)
from mcode.ui.flags import JsonFlag

bench_app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    json_mode: bool = False,
) -> None:
    from mcode.bench import runstate
    from mcode.launch.models import RunStatus, Target

    parsed_task_ids = _parse_task_ids(task_ids)
    runner = BenchmarkRunner(config=config, results_db=ResultsDB(db), json_mode=json_mode)
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
            if _is_polyglot_toolchain_exception(e):
                if json_mode:
                    print(
                        json.dumps(
                            {
                                "kind": "error",
                                "error_type": type(e).__name__,
                                "error": str(e),
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    typer.echo(f"✗ {e}", err=True)
                raise typer.Exit(2) from e
            if benchmark.startswith("swebench") and _is_retryable_infra_exception(e):
                typer.echo(f"✗ retryable infra failure before task loop: {e}", err=True)
                raise typer.Exit(SHARDED_INFRA_EXIT_CODE) from e
            raise
        if json_mode:
            print(
                json.dumps(
                    {
                        "kind": "summary",
                        "data": {
                            "run_id": summary.run_id,
                            "benchmark": benchmark,
                            "backend": backend,
                            "model": model,
                            "loop_budget": loop_budget,
                            "timeout_s": timeout_s,
                            "total": summary.total,
                            "passed": summary.passed,
                            "pass_rate": summary.pass_rate,
                        },
                    },
                    sort_keys=True,
                )
            )
        else:
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


def _is_polyglot_toolchain_exception(exc: object) -> bool:
    return exc.__class__.__name__ == "PolyglotToolchainError"


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


register_artifact_commands(bench_app)
register_suite_commands(bench_app)


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
        json_mode=json_mode,
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
        json_mode=json_mode,
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
        json_mode=json_mode,
    )
