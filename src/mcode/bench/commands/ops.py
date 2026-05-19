from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from mcode.bench.results import merge_shard_dbs
from mcode.ui.console import console
from mcode.ui.flags import JsonFlag


def _handle_exit(action: Callable[[], int]) -> None:
    from mcode.ui.errors import handle_errors

    @handle_errors
    def _do() -> None:
        rc = action()
        if rc != 0:
            raise typer.Exit(rc)

    _do()


def register_ops_commands(app: typer.Typer) -> None:
    @app.command("list")
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

    @app.command("show")
    def bench_show(
        run_id: Annotated[
            str | None, typer.Argument(help="run id (from `mcode bench list`)")
        ] = None,
        latest: Annotated[bool, typer.Option("--latest", help="Show the most recent run")] = False,
        json_mode: JsonFlag = False,
    ) -> None:
        """Show run details, DB summary, and artifact paths."""
        from mcode.bench.cancel import show_run

        _handle_exit(lambda: show_run(run_id, latest=latest, json_mode=json_mode))

    @app.command("prune")
    def bench_prune(
        json_mode: JsonFlag = False,
        status: Annotated[
            str | None, typer.Option("--status", help="Only prune this run status")
        ] = None,
        older_than: Annotated[
            str | None,
            typer.Option(
                "--older-than", help="Only prune runs older than a duration like 7d or 12h"
            ),
        ] = None,
        missing_db: Annotated[
            bool,
            typer.Option(
                "--missing-db/--any-db", help="Only prune records whose DB path is missing"
            ),
        ] = True,
        yes: Annotated[
            bool, typer.Option("--yes", help="Actually delete matching records")
        ] = False,
    ) -> None:
        """Remove stale bench run records from the launch state file."""
        from mcode.bench.cancel import prune_runs

        _handle_exit(
            lambda: prune_runs(
                json_mode=json_mode,
                status=status,
                older_than=older_than,
                missing_db=missing_db,
                yes=yes,
            )
        )

    @app.command("cancel")
    def bench_cancel(
        run_id: str = typer.Argument(..., help="run id (from `mcode bench list`)"),
    ) -> None:
        """Cancel a running sharded or Blue Vela bench run."""
        from mcode.bench.cancel import cancel_run

        _handle_exit(lambda: cancel_run(run_id))

    @app.command("merge-shards")
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
            f"out={report['out_path']} benchmark={report['benchmark']} "
            f"run_id={report['run_id']} tasks={report['tasks_written']} "
            f"shards_used={report['shards_used']} shards_ignored={report['shards_ignored']}"
        )
