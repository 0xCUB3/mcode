from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import typer

from mcode.bench.results import ResultsDB, RunSummary, merge_shard_dbs
from mcode.bench.summary import (
    RunPlan,
    print_failure_hints,
    print_run_footer,
    print_run_plan,
    print_run_summary,
    safe_rerun_metadata,
    task_time_ms,
)
from mcode.util import temporary_directory


def _print_run_summary(
    *,
    summary: RunSummary,
    benchmark: str,
    backend: str,
    model: str,
    loop_budget: int,
    timeout_s: int,
) -> None:
    print_run_summary(
        summary=summary,
        benchmark=benchmark,
        backend=backend,
        model=model,
        loop_budget=loop_budget,
        timeout_s=timeout_s,
    )


def _phase_argv(argv: list[str], phase: str) -> list[str]:
    result: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--phase":
            skip_next = True
            continue
        if arg.startswith("--phase="):
            continue
        result.append(arg)
    result.extend(["--phase", phase])
    return result


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
    runstate.open_run(
        run_id=run_id,
        benchmark=benchmark,
        target=Target.LOCAL_VLLM,
        db_path=db,
        metadata=safe_rerun_metadata(),
    )
    runstate.patch_run(run_id=run_id, progress={"current": 0, "total": 0})
    final_status: RunStatus = RunStatus.FAILED
    cancel_reason: str | None = None
    try:
        if not json_mode:
            print_run_plan(
                RunPlan(
                    benchmark=benchmark,
                    backend=backend,
                    model=model,
                    db=db,
                    loop_budget=loop_budget,
                    timeout_s=timeout_s,
                    location="local",
                    shards=shards,
                )
            )
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

            if command in {"swebench-lite", "swebench-live"} and "--dataset" in base_argv:
                prepare_db = run_dir / f"{db.stem}-prepare.db"
                prepare_log = run_dir / f"{db.stem}-prepare.log"
                prepare_argv = _phase_argv(base_argv, "prepare")
                dashboard.post("info", text="▶ preparing SWE-bench images before shard launch")
                with prepare_log.open("w", encoding="utf-8") as log_handle:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "mcode",
                            "bench",
                            command,
                            *prepare_argv,
                            "--db",
                            str(prepare_db),
                        ],
                        cwd=str(Path.cwd()),
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if proc.returncode != 0:
                    failed.append((-1, proc.returncode, prepare_log))
                    raise typer.Exit(proc.returncode)

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
            runstate.patch_run(run_id=run_id, metadata={"results_run_id": summary.run_id})
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
        print_run_footer(db=db, summary=summary, task_time_ms=task_time_ms(db, summary.run_id))
        print_failure_hints(db=db, run_id=summary.run_id)
    finally:
        # Best-effort close so partial state doesn't permanently mark the run
        # RUNNING. Wrapped so a second Ctrl+C during teardown cannot prevent
        # the close.
        try:
            runstate.close_run(run_id=run_id, status=final_status, cancel_reason=cancel_reason)
        except Exception:
            pass
