from __future__ import annotations

import json
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from mcode.bench.artifacts import read_task_manifest
from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig
from mcode.cli_shared import DEFAULT_DB_PATH
from mcode.ui.console import console
from mcode.ui.flags import JsonFlag


def _resolve_results_run_id(rdb: ResultsDB, run_id: int | None) -> int:
    if run_id is not None:
        return run_id
    row = rdb.conn.execute("SELECT MAX(id) AS run_id FROM runs").fetchone()
    if row is None or row["run_id"] is None:
        raise typer.BadParameter(f"No runs found in {rdb.path}")
    return int(row["run_id"])


def _artifact_replay_config(
    *,
    source_db: Path,
    run_id: int,
    task_id: str,
    candidate_index: int | None,
    benchmark_root: Path | None = None,
    artifact_dir_override: Path | None = None,
    fetch_missing_artifacts: bool = False,
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
        raise typer.BadParameter(f"No artifact manifest for task {task_id!r} in run {run_id}")
    manifest_path = Path(str(row["manifest_path"]))
    if artifact_dir_override is not None:
        manifest_path = artifact_dir_override / str(row["artifact_root"]) / "manifest.json"
    if not manifest_path.exists() and artifact_dir_override is None and fetch_missing_artifacts:
        run = _resolve_artifact_fetch_run(run_id=None, db=source_db)
        _resolved_run_id, _remote_artifact_dir, artifact_dir = _fetch_remote_artifacts_for_run(
            run=run,
            dest=None,
        )
        manifest_path = artifact_dir / str(row["artifact_root"]) / "manifest.json"
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
    table.add_column("selected", justify="right")
    table.add_column("verified")
    table.add_column("patch_bytes", justify="right")
    table.add_column("candidates", justify="right")
    table.add_column("evaluations", justify="right")
    table.add_column("manifest", overflow="fold")
    for row in filtered:
        table.add_row(
            row["task_id"],
            str(row.get("phase") or "-"),
            str(row.get("selected_candidate_index") or "-"),
            "yes" if row.get("selected_verification_succeeded") else "-",
            str(row.get("selected_patch_byte_count") or "-"),
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


def _fetch_remote_artifacts_for_run(*, run, dest: Path | None) -> tuple[str, str, Path]:
    import time
    from dataclasses import replace

    from mcode.launch import state as launch_state
    from mcode.launch.ssh import SshClient
    from mcode.ui.errors import MCodeError

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
    launch_state.update(
        None,
        lambda state: state.upsert_run(
            replace(
                run,
                remote={
                    **run.remote,
                    "local_artifact_dir": str(local_artifact_dir),
                    "artifacts_fetched_at": time.time(),
                },
            )
        ),
    )
    return resolved_run_id, remote_artifact_dir, local_artifact_dir


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
    from mcode.ui.errors import handle_errors

    @handle_errors
    def _do() -> None:
        run = _resolve_artifact_fetch_run(run_id=run_id, db=db)
        resolved_run_id, remote_artifact_dir, local_artifact_dir = _fetch_remote_artifacts_for_run(
            run=run, dest=dest
        )
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
    fetch_missing_artifacts: Annotated[
        bool,
        typer.Option(
            "--fetch-missing-artifacts",
            help="Fetch the recorded remote artifact directory if the local manifest is missing",
        ),
    ] = False,
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
        fetch_missing_artifacts=fetch_missing_artifacts,
    )
    target_db = out_db if out_db is not None else db.with_name(f"{db.stem}-replay.db")
    from mcode.bench.cli import _run_single_benchmark

    _run_single_benchmark(
        benchmark=benchmark,
        config=config,
        db=target_db,
        limit=None,
        task_ids=task_id,
        backend=config.backend_name,
        model=config.model_id,
        loop_budget=config.loop_budget
        + (
            config.aider_polyglot_retry_loop_budget
            if benchmark == "aider-polyglot" and config.aider_polyglot_retry
            else 0
        ),
        timeout_s=config.timeout_s,
        json_mode=False,
    )


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


def register_artifact_commands(app: typer.Typer) -> None:
    app.command("artifacts-list")(bench_artifacts_list)
    app.command("artifacts-fetch")(bench_artifacts_fetch)
    app.command("artifacts-show")(bench_artifacts_show)
    app.command("artifacts-replay")(bench_artifacts_replay)
    app.command("artifacts-patch")(bench_artifacts_patch)
