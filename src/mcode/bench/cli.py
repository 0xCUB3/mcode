from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from mcode.bench.artifacts_cli import register_artifact_commands
from mcode.bench.results import ResultsDB, merge_shard_dbs
from mcode.bench.runner import BenchConfig, BenchmarkRunner, NoTasksMatchedError
from mcode.bench.shards import (
    SHARDED_INFRA_EXIT_CODE,
    _full_db_summary,
    _is_retryable_infra_exception,
    _merge_into_results_db,
    _print_run_summary,
    _run_sharded_benchmark,
)
from mcode.bench.suite_cli import register_suite_commands
from mcode.bench.summary import (
    RunPlan,
    print_failure_hints,
    print_run_footer,
    print_run_plan,
    safe_rerun_metadata,
    task_time_ms,
)
from mcode.bench.terminalbench import DEFAULT_DATASET, TerminalBenchConfig
from mcode.cli_shared import (
    DEFAULT_DB_PATH,
    _append_option,
    _optional_str,
    _parse_task_ids,
    _resolve_artifact_dir,
    _validate_sampling,
    _validate_shard_options,
)
from mcode.ui.console import console
from mcode.ui.flags import JsonFlag

bench_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run benchmarks, inspect saved runs, and manage artifacts.",
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
    json_mode: bool = False,
) -> None:
    from mcode.bench import runstate
    from mcode.launch.models import RunStatus, Target

    parsed_task_ids = _parse_task_ids(task_ids)
    run_id = runstate.make_run_id(benchmark)
    runstate.open_run(
        run_id=run_id,
        benchmark=benchmark,
        target=Target.LOCAL_VLLM,
        db_path=db,
        metadata=safe_rerun_metadata(),
    )
    final_status: RunStatus = RunStatus.FAILED
    cancel_reason: str | None = None
    try:
        runner = BenchmarkRunner(
            config=config,
            results_db=ResultsDB(db),
            json_mode=json_mode,
            state_run_id=run_id,
        )
        if not json_mode:
            print_run_plan(
                RunPlan(
                    benchmark=benchmark,
                    backend=backend,
                    model=model,
                    db=db,
                    loop_budget=loop_budget,
                    timeout_s=timeout_s,
                    phase=config.phase,
                    location="local",
                    artifact_dir=config.artifact_dir,
                    limit=limit,
                    task_ids=task_ids,
                    shard_count=config.task_shard_count,
                    shard_index=config.task_shard_index,
                )
            )
        try:
            summary = runner.run_benchmark(benchmark, limit=limit, task_ids=parsed_task_ids)
        except KeyboardInterrupt:
            final_status = RunStatus.STOPPED
            cancel_reason = "interrupt"
            raise
        except NoTasksMatchedError as e:
            if json_mode:
                print(json.dumps({"kind": "error", "error": str(e)}, sort_keys=True))
            else:
                typer.echo(f"✗ {e}", err=True)
            raise typer.Exit(2) from e
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
            print_run_footer(
                db=db,
                summary=summary,
                task_time_ms=task_time_ms(db, summary.run_id),
            )
            print_failure_hints(db=db, run_id=summary.run_id)
        runstate.patch_run(run_id=run_id, metadata={"results_run_id": summary.run_id})
        final_status = RunStatus.DONE
    finally:
        try:
            runstate.close_run(run_id=run_id, status=final_status, cancel_reason=cancel_reason)
        except Exception:
            pass


def _is_polyglot_toolchain_exception(exc: object) -> bool:
    return exc.__class__.__name__ == "PolyglotToolchainError"


def _run_terminal_bench_local(
    *,
    config: TerminalBenchConfig,
    db: Path,
    limit: int | None,
    task_ids: str | None,
    json_mode: bool = False,
) -> None:
    from mcode.bench import runstate
    from mcode.bench.terminalbench import run_terminal_bench
    from mcode.launch.models import RunStatus, Target

    parsed_task_ids = _parse_task_ids(task_ids)
    run_id = runstate.make_run_id("terminal-bench")
    runstate.open_run(
        run_id=run_id,
        benchmark="terminal-bench",
        target=Target.LOCAL_VLLM,
        db_path=db,
        metadata=safe_rerun_metadata(),
    )
    final_status: RunStatus = RunStatus.FAILED
    try:
        if not json_mode:
            print_run_plan(
                RunPlan(
                    benchmark="terminal-bench",
                    backend=config.backend_name,
                    model=config.model_id,
                    db=db,
                    loop_budget=0,
                    timeout_s=int(config.timeout_multiplier * 3600),
                    phase="run",
                    location="local",
                    artifact_dir=config.artifact_dir,
                    limit=limit,
                    task_ids=task_ids,
                )
            )
        result = run_terminal_bench(
            config=config,
            results_db=ResultsDB(db),
            limit=limit,
            task_ids=parsed_task_ids,
            stream_output=not json_mode,
        )
        runstate.patch_run(
            run_id=run_id,
            metadata={
                "results_run_id": result.summary.run_id,
                "harbor_job_dir": str(result.job_dir),
            },
        )
        if json_mode:
            print(
                json.dumps(
                    {
                        "kind": "summary",
                        "data": {
                            "run_id": result.summary.run_id,
                            "benchmark": "terminal-bench",
                            "backend": config.backend_name,
                            "model": config.model_id,
                            "total": result.summary.total,
                            "passed": result.summary.passed,
                            "pass_rate": result.summary.pass_rate,
                            "harbor_returncode": result.returncode,
                            "harbor_job_dir": str(result.job_dir),
                        },
                    },
                    sort_keys=True,
                )
            )
        else:
            _print_run_summary(
                summary=result.summary,
                benchmark="terminal-bench",
                backend=config.backend_name,
                model=config.model_id,
                loop_budget=0,
                timeout_s=int(config.timeout_multiplier * 3600),
            )
            print_run_footer(
                db=db,
                summary=result.summary,
                task_time_ms=task_time_ms(db, result.summary.run_id),
            )
            console.print(f"Harbor job: {result.job_dir}")
            print_failure_hints(db=db, run_id=result.summary.run_id)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
        final_status = RunStatus.DONE
    finally:
        try:
            runstate.close_run(run_id=run_id, status=final_status)
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

    if "--json" not in argv:
        _print_remote_run_plan(command=command, argv=argv, model=model, db=db)
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


def _option_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _print_remote_run_plan(*, command: str, argv: list[str], model: str, db: Path) -> None:
    print_run_plan(
        RunPlan(
            benchmark=command,
            backend=_option_value(argv, "--backend") or "openai",
            model=model,
            db=db,
            loop_budget=int(_option_value(argv, "--loop-budget") or 0),
            timeout_s=int(_option_value(argv, "--timeout") or 0),
            phase=_option_value(argv, "--phase") or "run",
            location="bluevela",
            artifact_dir=_optional_path(_option_value(argv, "--artifact-dir")),
            limit=_optional_int(_option_value(argv, "--limit")),
            task_ids=_option_value(argv, "--task-ids"),
            shards=_optional_int(_option_value(argv, "--shards")),
            shard_count=_optional_int(_option_value(argv, "--shard-count")),
            shard_index=_optional_int(_option_value(argv, "--shard-index")),
        )
    )


def _dispatch_benchmark_run(
    *,
    command: str,
    on: str,
    base_argv: list[str],
    config: BenchConfig,
    db: Path,
    limit: int | None,
    task_ids: str | None,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
    shards: int | None,
    shard_count: int | None,
    shard_index: int | None,
    default_db_path: Path,
    fetch_db: bool,
    fetch_artifacts: bool,
    json_mode: bool,
) -> None:
    if on == "bluevela":
        remote_argv = [*base_argv]
        _append_option(remote_argv, "--shards", shards)
        _append_option(remote_argv, "--shard-count", shard_count)
        _append_option(remote_argv, "--shard-index", shard_index)
        if json_mode:
            remote_argv.append("--json")
        _run_bluevela_benchmark(
            command=command,
            argv=remote_argv,
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
            command=command,
            base_argv=base_argv,
            shards=shards,
            db=db,
            benchmark=command,
            backend=backend,
            model=model,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
            json_mode=json_mode,
        )
        return
    if shard_count and shard_count > 1 and db == default_db_path:
        typer.echo(
            "Note: when running shards in parallel, use a unique --db per shard to avoid SQLite "
            "locks.",
            err=True,
        )

    _run_single_benchmark(
        benchmark=command,
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


def _run_bluevela_benchmark_rc(
    *,
    command: str,
    argv: list[str],
    model: str,
    db: Path,
    fetch_db: bool,
    fetch_artifacts: bool = False,
) -> int:
    from mcode.bench.remote import RemoteBenchError, run_bench_on_bluevela

    try:
        return run_bench_on_bluevela(
            bench_argv=[command, *argv],
            model=model,
            local_db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    except RemoteBenchError as e:
        typer.echo(f"✗ {e}", err=True)
        return 1


def _launch_bluevela_server(
    model: str,
    *,
    tensor_parallel: int | None,
    max_model_len: int | None,
    json_mode: bool,
) -> None:
    from mcode.launch import bluevela, profiles
    from mcode.launch import config as config_mod
    from mcode.launch.models import LaunchError, LaunchSpec, Target
    from mcode.launch.progress import choose as choose_reporter
    from mcode.ui.errors import print_error as print_mcode_error

    cfg = config_mod.load()
    profile = profiles.resolve(model)
    if tensor_parallel is not None or max_model_len is not None:
        profile = profiles.override(
            profile,
            tensor_parallel=tensor_parallel,
            max_model_len=max_model_len,
        )
    spec = LaunchSpec(target=Target.BLUEVELA, model=model, profile=profile)
    reporter = choose_reporter(bluevela.PHASES, json_mode=json_mode)
    try:
        with reporter:
            bluevela.launch(spec, reporter, cfg=cfg)
    except LaunchError as e:
        print_mcode_error(e)
        raise typer.Exit(1) from e


def _ensure_bluevela_server(
    model: str,
    *,
    tensor_parallel: int | None,
    max_model_len: int | None,
    json_mode: bool,
) -> None:
    from mcode.bench.remote import RemoteBenchError, _resolve_endpoint
    from mcode.launch import config as config_mod

    cfg = config_mod.load()
    try:
        _resolve_endpoint(model, cfg=cfg)
        return
    except RemoteBenchError:
        _launch_bluevela_server(
            model,
            tensor_parallel=tensor_parallel,
            max_model_len=max_model_len,
            json_mode=json_mode,
        )


def _db_task_count(db: Path) -> int:
    if not db.exists():
        return 0
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
        try:
            row = conn.execute("SELECT COUNT(*) FROM task_results").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0
    return int(row[0] if row else 0)


def _load_swebench_task_ids(
    *,
    dataset: str,
    split: str,
    limit: int | None,
    task_ids: str | None,
) -> list[str]:
    from mcode.bench.swebench_lite import load_swebench_lite

    parsed_task_ids = _parse_task_ids(task_ids)
    tasks = load_swebench_lite(
        Path(".cache"),
        split=split,
        limit=limit,
        instance_ids=parsed_task_ids,
        dataset_name=dataset,
    )
    return [task.instance_id for task in tasks]


def _run_bluevela_task_chunks(
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
    eval_repair_attempts: int,
    phase: str,
    db: Path,
    shards: int | None,
    fetch_artifacts: bool,
    chunk_size: int,
    relaunch_vllm: bool,
    vllm_tensor_parallel: int | None,
    vllm_max_model_len: int | None,
    json_mode: bool,
) -> None:
    if phase != "run":
        typer.echo("✗ --chunk-size only supports --phase run", err=True)
        raise typer.Exit(2)
    if not shards or shards < 1:
        typer.echo("✗ --chunk-size with --on bluevela requires --shards", err=True)
        raise typer.Exit(2)

    all_task_ids = _load_swebench_task_ids(
        dataset=dataset,
        split=split,
        limit=limit,
        task_ids=task_ids,
    )
    if not all_task_ids:
        typer.echo("✗ no SWE-bench tasks selected", err=True)
        raise typer.Exit(2)

    chunk_root = db.parent / f"{db.stem}-chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_dbs: list[Path] = []
    chunks = [all_task_ids[i : i + chunk_size] for i in range(0, len(all_task_ids), chunk_size)]
    for index, chunk in enumerate(chunks):
        chunk_db = chunk_root / f"chunk-{index:03d}.db"
        expected = len(chunk)
        if _db_task_count(chunk_db) >= expected:
            typer.echo(f"skip chunk {index + 1}/{len(chunks)} rows={expected} db={chunk_db}")
            chunk_dbs.append(chunk_db)
            continue
        if chunk_db.exists():
            chunk_db.unlink()
        if relaunch_vllm:
            _ensure_bluevela_server(
                model,
                tensor_parallel=vllm_tensor_parallel,
                max_model_len=vllm_max_model_len,
                json_mode=json_mode,
            )
        chunk_artifact_dir = _resolve_artifact_dir(chunk_db, None)
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
            limit=None,
            n_samples=n_samples,
            sampling=sampling,
            sampling_budget=sampling_budget,
            selection_attempts=selection_attempts,
            task_ids=",".join(chunk),
            dataset=dataset,
            diagnostic_traces=diagnostic_traces,
            check_image_digests=check_image_digests,
            eval_repair_attempts=eval_repair_attempts,
            phase=phase,
            artifact_dir=chunk_artifact_dir,
        )
        _append_option(argv, "--shards", shards)
        typer.echo(f"run chunk {index + 1}/{len(chunks)} tasks={expected} db={chunk_db}")
        rc = _run_bluevela_benchmark_rc(
            command="swebench-lite",
            argv=argv,
            model=model,
            db=chunk_db,
            fetch_db=True,
            fetch_artifacts=fetch_artifacts,
        )
        rows = _db_task_count(chunk_db)
        if rc != 0 or rows < expected:
            typer.echo(
                f"✗ chunk {index + 1}/{len(chunks)} failed rc={rc} "
                f"rows={rows}/{expected} db={chunk_db}",
                err=True,
            )
            raise typer.Exit(rc or 1)
        chunk_dbs.append(chunk_db)

    if db.exists():
        db.unlink()
    summary = _merge_into_results_db(db=db, shard_paths=chunk_dbs, merge_mode="full_db")
    summary = _full_db_summary(db)
    _print_run_summary(
        summary=summary,
        benchmark="swebench-lite",
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )
    raise typer.Exit(0)


def _terminal_bench_cli_args(
    *,
    model: str,
    backend: str,
    agent: str,
    dataset: str,
    jobs_dir: Path,
    job_name: str | None,
    environment_type: str,
    n_concurrent: int,
    loop_budget: int,
    timeout_multiplier: float,
    harbor_executable: str,
    artifact_dir: Path,
    limit: int | None,
    task_ids: str | None,
    extra_harbor_arg: list[str] | None,
) -> list[str]:
    argv = [
        "--model",
        model,
        "--backend",
        backend,
        "--agent",
        agent,
        "--dataset",
        dataset,
        "--jobs-dir",
        str(jobs_dir),
        "--env",
        environment_type,
        "--n-concurrent",
        str(n_concurrent),
        "--loop-budget",
        str(loop_budget),
        "--timeout-multiplier",
        str(timeout_multiplier),
        "--harbor-executable",
        harbor_executable,
        "--artifact-dir",
        str(artifact_dir),
    ]
    _append_option(argv, "--job-name", job_name)
    _append_option(argv, "--limit", limit)
    _append_option(argv, "--task-ids", task_ids)
    for extra in extra_harbor_arg or []:
        argv.extend(["--harbor-arg", extra])
    return argv


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
    eval_repair_attempts: int,
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
    if eval_repair_attempts:
        _append_option(argv, "--eval-repair-attempts", eval_repair_attempts)
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
    sampling: str,
    sampling_budget: int | None,
    diagnostic_traces: bool,
    selection_attempts: int,
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
    if sampling != "none":
        argv.extend(["--sampling", sampling])
    _append_option(argv, "--sampling-budget", sampling_budget)
    if diagnostic_traces:
        argv.append("--diagnostic-traces")
    if selection_attempts != 1:
        argv.extend(["--selection-attempts", str(selection_attempts)])
    if no_retry:
        argv.append("--no-retry")
    return argv


@bench_app.command("terminal-bench")
def bench_terminal_bench(
    model: Annotated[str, typer.Option("--model", help="Model id passed to the Harbor agent")],
    backend: Annotated[
        str,
        typer.Option("--backend", help="mCode backend name for the mCode Harbor agent"),
    ] = "openai",
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="Harbor agent to run: mcode, oracle, claude-code, codex, openhands, etc.",
        ),
    ] = "mcode",
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Harbor Terminal-Bench dataset id"),
    ] = DEFAULT_DATASET,
    jobs_dir: Annotated[
        Path,
        typer.Option("--jobs-dir", help="Directory where Harbor writes job outputs"),
    ] = Path("experiments/results/terminal-bench-jobs"),
    job_name: Annotated[
        str | None,
        typer.Option("--job-name", help="Stable Harbor job name for resume/inspection"),
    ] = None,
    environment_type: Annotated[
        str,
        typer.Option("--env", help="Harbor environment: docker, daytona, modal, e2b, etc."),
    ] = "docker",
    n_concurrent: Annotated[
        int,
        typer.Option("--n-concurrent", min=1, help="Concurrent Harbor trials"),
    ] = 1,
    loop_budget: Annotated[
        int,
        typer.Option("--loop-budget", min=1, help="mCode terminal-agent ReACT turns"),
    ] = 25,
    timeout_multiplier: Annotated[
        float,
        typer.Option("--timeout-multiplier", min=0.1, help="Scale Harbor task timeouts"),
    ] = 1.0,
    harbor_executable: Annotated[
        str,
        typer.Option("--harbor-executable", help="Harbor executable to run"),
    ] = "harbor",
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite results DB path"),
    ] = Path("experiments/results/terminal-bench.db"),
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", help="Directory for imported Harbor artifacts"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run first N tasks")] = None,
    task_ids: Annotated[
        str | None,
        typer.Option("--task-ids", help="Comma-separated TB2 task IDs or a JSON/text file"),
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
            help="Rsync the artifact directory and Harbor jobs back when --on bluevela",
        ),
    ] = True,
    extra_harbor_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--harbor-arg",
            help="Extra raw argument to append to `harbor run` (repeatable)",
        ),
    ] = None,
    json_mode: JsonFlag = False,
) -> None:
    """Run Terminal-Bench 2.0 through Harbor and import results into mCode."""

    resolved_artifact_dir = _resolve_artifact_dir(db, artifact_dir)
    base_argv = _terminal_bench_cli_args(
        model=model,
        backend=backend,
        agent=agent,
        dataset=dataset,
        jobs_dir=jobs_dir,
        job_name=job_name,
        environment_type=environment_type,
        n_concurrent=n_concurrent,
        loop_budget=loop_budget,
        timeout_multiplier=timeout_multiplier,
        harbor_executable=harbor_executable,
        artifact_dir=resolved_artifact_dir,
        limit=limit,
        task_ids=task_ids,
        extra_harbor_arg=extra_harbor_arg,
    )
    if on == "bluevela":
        if json_mode:
            base_argv.append("--json")
        _run_bluevela_benchmark(
            command="terminal-bench",
            argv=base_argv,
            model=model,
            db=db,
            fetch_db=fetch_db,
            fetch_artifacts=fetch_artifacts,
        )
    if on != "local":
        typer.echo(f"✗ unknown --on target {on!r}; expected local or bluevela", err=True)
        raise typer.Exit(2)
    config = TerminalBenchConfig(
        model_id=model,
        backend_name=backend,
        agent=agent,
        dataset=dataset,
        jobs_dir=jobs_dir,
        job_name=job_name,
        environment_type=environment_type,
        n_concurrent=n_concurrent,
        timeout_multiplier=timeout_multiplier,
        harbor_executable=harbor_executable,
        artifact_dir=resolved_artifact_dir,
        extra_harbor_args=tuple(extra_harbor_arg or ()),
        agent_kwargs={"loop_budget": str(loop_budget)},
    )
    _run_terminal_bench_local(
        config=config,
        db=db,
        limit=limit,
        task_ids=task_ids,
        json_mode=json_mode,
    )


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
    wide: Annotated[
        bool, typer.Option("--wide", help="Show target, shard, artifact, and DB columns")
    ] = False,
) -> None:
    """List saved bench runs from the launch state file."""
    from mcode.bench.cancel import list_runs

    rc = list_runs(
        json_mode=json_mode,
        benchmark=benchmark,
        status=status,
        artifacts_only=artifacts_only,
        limit=limit,
        wide=wide,
    )
    if rc != 0:
        raise typer.Exit(rc)


@bench_app.command("show")
def bench_show(
    run_id: Annotated[str | None, typer.Argument(help="run id (from `mcode bench list`)")] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Show the most recent run")] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Show run details, DB summary, and artifact paths."""
    from mcode.bench.cancel import show_run
    from mcode.ui.errors import handle_errors

    @handle_errors
    def _do() -> None:
        rc = show_run(run_id, latest=latest, json_mode=json_mode)
        if rc != 0:
            raise typer.Exit(rc)

    _do()


@bench_app.command("prune")
def bench_prune(
    json_mode: JsonFlag = False,
    status: Annotated[
        str | None, typer.Option("--status", help="Only prune this run status")
    ] = None,
    older_than: Annotated[
        str | None,
        typer.Option("--older-than", help="Only prune runs older than a duration like 7d or 12h"),
    ] = None,
    missing_db: Annotated[
        bool,
        typer.Option("--missing-db/--any-db", help="Only prune records whose DB path is missing"),
    ] = True,
    yes: Annotated[bool, typer.Option("--yes", help="Actually delete matching records")] = False,
) -> None:
    """Remove stale bench run records from the launch state file."""
    from mcode.bench.cancel import prune_runs
    from mcode.ui.errors import handle_errors

    @handle_errors
    def _do() -> None:
        rc = prune_runs(
            json_mode=json_mode,
            status=status,
            older_than=older_than,
            missing_db=missing_db,
            yes=yes,
        )
        if rc != 0:
            raise typer.Exit(rc)

    _do()


@bench_app.command("cancel")
def bench_cancel(
    run_id: str = typer.Argument(..., help="run id (from `mcode bench list`)"),
) -> None:
    """Cancel a running sharded or Blue Vela bench run."""
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


@bench_app.command("merge-shards")
def bench_merge_shards(
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
        Literal["run", "generate", "evaluate", "prepare"],
        typer.Option("--phase", help="Benchmark phase: run, generate, evaluate, or prepare"),
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
    """Run SWE-bench Live tasks with container-based evaluation."""

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
    base_argv = _swebench_live_cli_args(
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
    _dispatch_benchmark_run(
        command="swebench-live",
        on=on,
        base_argv=base_argv,
        config=config,
        db=db,
        limit=limit,
        task_ids=task_ids,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
        default_db_path=DEFAULT_DB_PATH,
        fetch_db=fetch_db,
        fetch_artifacts=fetch_artifacts,
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
        Literal["run", "generate", "evaluate", "prepare"],
        typer.Option("--phase", help="Benchmark phase: run, generate, evaluate, or prepare"),
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
    eval_repair_attempts: Annotated[
        int,
        typer.Option(
            "--eval-repair-attempts",
            min=0,
            help="Retry failed official SWE-bench evaluations with deterministic eval feedback",
        ),
    ] = 0,
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
    chunk_size: Annotated[
        int | None,
        typer.Option(
            "--chunk-size",
            min=1,
            help="Run Blue Vela in sequential task chunks, writing chunk DBs then merging them",
        ),
    ] = None,
    relaunch_vllm: Annotated[
        bool,
        typer.Option(
            "--relaunch-vllm/--no-relaunch-vllm",
            help="For --chunk-size, launch a fresh Blue Vela vLLM when no healthy server exists",
        ),
    ] = False,
    vllm_tensor_parallel: Annotated[
        int | None,
        typer.Option(
            "--vllm-tensor-parallel",
            min=1,
            help="For --relaunch-vllm, override Blue Vela vLLM tensor parallel size",
        ),
    ] = None,
    vllm_max_model_len: Annotated[
        int | None,
        typer.Option(
            "--vllm-max-model-len",
            min=1,
            help="For --relaunch-vllm, override Blue Vela vLLM max model length",
        ),
    ] = None,
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Run SWE-bench Lite or Verified tasks with container-based evaluation."""
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
    if chunk_size is not None:
        if on != "bluevela":
            typer.echo("✗ --chunk-size currently requires --on bluevela", err=True)
            raise typer.Exit(2)
        if shard_count is not None or shard_index is not None:
            typer.echo("✗ --chunk-size cannot be combined with manual shard mode", err=True)
            raise typer.Exit(2)
        _run_bluevela_task_chunks(
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
            eval_repair_attempts=eval_repair_attempts,
            phase=phase,
            db=db,
            shards=shards,
            fetch_artifacts=fetch_artifacts,
            chunk_size=chunk_size,
            relaunch_vllm=relaunch_vllm,
            vllm_tensor_parallel=vllm_tensor_parallel,
            vllm_max_model_len=vllm_max_model_len,
            json_mode=json_mode,
        )
    base_argv = _swebench_lite_cli_args(
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
        eval_repair_attempts=eval_repair_attempts,
        phase=phase,
        artifact_dir=resolved_artifact_dir,
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
        swebench_eval_repair_attempts=eval_repair_attempts,
        diagnostic_traces=diagnostic_traces,
    )
    _dispatch_benchmark_run(
        command="swebench-lite",
        on=on,
        base_argv=base_argv,
        config=config,
        db=db,
        limit=limit,
        task_ids=task_ids,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
        default_db_path=DEFAULT_DB_PATH,
        fetch_db=fetch_db,
        fetch_artifacts=fetch_artifacts,
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
        Literal["run", "generate", "evaluate", "prepare"],
        typer.Option("--phase", help="Benchmark phase: run, generate, evaluate, or prepare"),
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
    diagnostic_traces: Annotated[
        bool,
        typer.Option(
            "--diagnostic-traces/--no-diagnostic-traces",
            help="Persist compact benchmark diagnostic trace events",
        ),
    ] = False,
    json_mode: JsonFlag = False,
) -> None:
    """Run Aider Polyglot coding exercises through the benchmark harness."""

    from mcode.bench.aider_polyglot import default_benchmark_root, supported_languages

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

    base_argv = _aider_polyglot_cli_args(
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
        sampling=sampling,
        sampling_budget=sampling_budget,
        diagnostic_traces=diagnostic_traces,
        selection_attempts=selection_attempts,
        phase=phase,
        artifact_dir=resolved_artifact_dir,
    )
    effective_loop_budget = loop_budget + (0 if no_retry else retry_loop_budget)
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
        sampling_strategy=sampling,
        sampling_budget=sampling_budget,
        diagnostic_traces=diagnostic_traces,
        selection_attempts=selection_attempts,
    )
    _dispatch_benchmark_run(
        command="aider-polyglot",
        on=on,
        base_argv=base_argv,
        config=config,
        db=db,
        limit=limit,
        task_ids=selected_task_ids,
        backend=backend,
        model=model,
        loop_budget=effective_loop_budget,
        timeout_s=300,
        shards=shards,
        shard_count=shard_count,
        shard_index=shard_index,
        default_db_path=Path("experiments/results/aider-polyglot.db"),
        fetch_db=fetch_db,
        fetch_artifacts=fetch_artifacts,
        json_mode=json_mode,
    )
