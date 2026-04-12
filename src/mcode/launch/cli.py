"""Typer app for `mcode launch`.

Per-target subcommands, no unified dispatcher. LaunchError → `✗ what / why /
next / logs` formatting. MCODE_DEBUG=1 re-enables tracebacks.

Commands:

    mcode launch bluevela      --model ...
    mcode launch local-vllm    --model ...
    mcode launch local-ollama  --model ...
    mcode launch status        [--json]
    mcode launch logs   <id>
    mcode launch stop   <id> | --all
    mcode launch doctor <target> [--deep]
    mcode launch refresh       (walk state and re-check via target modules)

Output:
- Server launch prints the endpoint as the last line on stdout so callers can
  capture it: `OPENAI_BASE_URL=$(mcode launch local-vllm --model X --json ...)`.
- Errors print to stderr in the formatted layout and exit 1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from mcode.launch import bluevela, local_ollama, local_vllm, profiles, state
from mcode.launch import config as config_mod
from mcode.launch.models import (
    LaunchError,
    LaunchSpec,
    ServerRecord,
    Target,
)
from mcode.launch.progress import choose as choose_reporter

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Launch vLLM + benchmarks on Blue Vela or locally.",
)


# ---------------------------------------------------------------------------
# error formatting
# ---------------------------------------------------------------------------
def _print_error(e: LaunchError) -> None:
    out = sys.stderr
    print(f"\033[31m✗\033[0m {e.what}", file=out)
    if e.why:
        print(f"  why:  {e.why}", file=out)
    if e.next:
        print(f"  next: {e.next}", file=out)
    if e.logs:
        print(f"  logs: {e.logs}", file=out)


def _run(block):
    """Call `block()`; format LaunchError; exit 1 on failure. MCODE_DEBUG=1
    short-circuits to raw traceback for dev."""
    try:
        return block()
    except LaunchError as e:
        if os.environ.get("MCODE_DEBUG"):
            raise
        _print_error(e)
        raise typer.Exit(1) from e


def _build_spec(target: Target, model: str) -> LaunchSpec:
    return LaunchSpec(
        target=target,
        model=model,
        profile=profiles.resolve(model),
    )


def _endpoint_stdout(server: ServerRecord, *, json_mode: bool) -> None:
    if json_mode:
        payload = {
            "id": server.id,
            "target": server.target.value,
            "endpoint": server.endpoint,
            "model": server.model,
            "job_id": server.job_id,
            "status": server.status,
        }
        print(json.dumps(payload))
    else:
        print(server.endpoint)


# ---------------------------------------------------------------------------
# launch bluevela
# ---------------------------------------------------------------------------
@app.command("bluevela")
def cmd_bluevela(
    model: str = typer.Option(..., "--model", "-m", help="HF model id, e.g. Qwen/Qwen3.5-27B"),
    json_mode: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Submit a vLLM server job to Blue Vela LSF."""
    cfg = config_mod.load()
    spec = _build_spec(Target.BLUEVELA, model)
    reporter = choose_reporter(bluevela.PHASES, json_mode=json_mode)

    def block() -> ServerRecord:
        with reporter:
            return bluevela.launch(spec, reporter, cfg=cfg)

    server = _run(block)
    _endpoint_stdout(server, json_mode=json_mode)


# ---------------------------------------------------------------------------
# launch local-vllm
# ---------------------------------------------------------------------------
@app.command("local-vllm")
def cmd_local_vllm(
    model: str = typer.Option(..., "--model", "-m"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Spin up a local vLLM server."""
    cfg = config_mod.load()
    spec = _build_spec(Target.LOCAL_VLLM, model)
    reporter = choose_reporter(local_vllm.PHASES, json_mode=json_mode)

    def block() -> ServerRecord:
        with reporter:
            return local_vllm.launch(spec, reporter, cfg=cfg)

    server = _run(block)
    _endpoint_stdout(server, json_mode=json_mode)


# ---------------------------------------------------------------------------
# launch local-ollama
# ---------------------------------------------------------------------------
@app.command("local-ollama")
def cmd_local_ollama(
    model: str = typer.Option(..., "--model", "-m"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Pull + expose an Ollama model via its OpenAI-compat endpoint."""
    cfg = config_mod.load()
    spec = _build_spec(Target.LOCAL_OLLAMA, model)
    reporter = choose_reporter(local_ollama.PHASES, json_mode=json_mode)

    def block() -> ServerRecord:
        with reporter:
            return local_ollama.launch(spec, reporter, cfg=cfg)

    server = _run(block)
    _endpoint_stdout(server, json_mode=json_mode)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
@app.command("status")
def cmd_status(
    json_mode: bool = typer.Option(False, "--json"),
    raw: bool = typer.Option(False, "--raw", help="include internal LSF state"),
) -> None:
    """List currently-known servers and runs."""
    s = state.load()
    if json_mode:
        payload = {
            "servers": [
                {
                    "id": srv.id,
                    "target": srv.target.value,
                    "endpoint": srv.endpoint,
                    "model": srv.model,
                    "status": srv.status,
                    "job_id": srv.job_id,
                    **({"lsf_state": srv.metadata.get("lsf_state")} if raw else {}),
                }
                for srv in s.servers
            ],
            "runs": [
                {
                    "id": r.id,
                    "target": r.target.value,
                    "status": r.status.value,
                    "benchmark": r.benchmark,
                    "server_id": r.server_id,
                    "shards": len(r.shard_job_ids),
                }
                for r in s.runs
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    if not s.servers and not s.runs:
        print("no servers or runs recorded")
        return
    if s.servers:
        print("servers:")
        for srv in s.servers:
            marker = {"healthy": "✓", "pending": "·", "failed": "✗", "stopped": "—"}.get(
                srv.status, "?"
            )
            print(
                f"  {marker} {srv.id}  [{srv.target.value}]  {srv.model}"
                f"  {srv.endpoint or '(no endpoint yet)'}  ({srv.status})"
            )
    if s.runs:
        print("runs:")
        for r in s.runs:
            print(f"  - {r.id}  [{r.target.value}]  {r.benchmark}  ({r.status.value})")


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
@app.command("logs")
def cmd_logs(id: str = typer.Argument(...)) -> None:
    """Print the path to the log for a run/server (or tail a local log)."""
    s = state.load()
    srv = s.server(id)
    if srv is None:
        _print_error(
            LaunchError(
                what=f"no server with id {id!r}", why="", next="`mcode launch status` to list"
            )
        )
        raise typer.Exit(1)
    if not srv.log_path:
        print("no log path recorded")
        return
    # For Blue Vela, the log is remote — give the user the SSH tail command.
    # For local targets, tail -n 50 works.
    if srv.target == Target.BLUEVELA:
        login = (srv.metadata or {}).get("login", "")
        print(f"ssh {login} tail -n 200 -f {srv.log_path}")
        return
    p = Path(srv.log_path)
    if not p.exists():
        print(f"log file missing: {p}")
        return
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        print(f"could not read log: {e}")
        return
    lines = text.splitlines()[-200:]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
@app.command("stop")
def cmd_stop(
    id: str | None = typer.Argument(None, help="server id; omit with --all"),
    all_: bool = typer.Option(False, "--all", help="stop everything in state"),
) -> None:
    """Stop one server by id, or --all to stop every server. Never uses
    `bkill 0` or `-u` — only the caller's recorded jobs."""
    cfg = config_mod.load()
    s = state.load()
    if all_ and id:
        _print_error(
            LaunchError(what="--all and an id are mutually exclusive", why="", next="pick one")
        )
        raise typer.Exit(1)
    targets: list[ServerRecord] = (
        list(s.servers) if all_ else [srv for srv in s.servers if srv.id == id]
    )
    if not targets:
        if all_:
            print("nothing to stop")
            return
        _print_error(
            LaunchError(what=f"no server with id {id!r}", why="", next="`mcode launch status`")
        )
        raise typer.Exit(1)
    for srv in targets:
        try:
            if srv.target == Target.BLUEVELA:
                bluevela.stop(srv.id, cfg=cfg)
            elif srv.target == Target.LOCAL_VLLM:
                local_vllm.stop(srv.id)
            elif srv.target == Target.LOCAL_OLLAMA:
                local_ollama.stop(srv.id)
            print(f"stopped: {srv.id}")
        except LaunchError as e:
            _print_error(e)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@app.command("doctor")
def cmd_doctor(
    target: str = typer.Argument(...),
    deep: bool = typer.Option(False, "--deep"),
) -> None:
    """Health check for a target."""
    cfg = config_mod.load()
    if target == "bluevela":
        checks = bluevela.doctor(cfg)
    elif target == "local-vllm":
        checks = local_vllm.doctor(cfg)
    elif target == "local-ollama":
        checks = local_ollama.doctor(cfg)
    else:
        _print_error(
            LaunchError(
                what=f"unknown target {target!r}",
                why="valid: bluevela, local-vllm, local-ollama",
                next="pick one of those",
            )
        )
        raise typer.Exit(1)

    any_failed = False
    for c in checks:
        icon = "\033[32m✓\033[0m" if c.ok else "\033[31m✗\033[0m"
        print(f"{icon} {c.name}")
        if c.detail:
            print(f"  {c.detail}")
        if not c.ok and c.next:
            print(f"  next: {c.next}")
        any_failed = any_failed or not c.ok
    if any_failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------
@app.command("refresh")
def cmd_refresh() -> None:
    """Re-query each server/run against its target and persist updated status."""
    cfg = config_mod.load()

    def _update(s: state.State) -> int:
        count = 0
        for srv in list(s.servers):
            try:
                if srv.target == Target.BLUEVELA:
                    updated = bluevela.refresh(srv, cfg=cfg)
                elif srv.target == Target.LOCAL_VLLM:
                    updated = local_vllm.refresh(srv)
                elif srv.target == Target.LOCAL_OLLAMA:
                    updated = local_ollama.refresh(srv, cfg=cfg)
                else:
                    continue
            except LaunchError:
                continue
            if isinstance(updated, ServerRecord):
                s.upsert_server(updated)
                count += 1
        return count

    n = state.update(None, _update)
    print(f"refreshed {n} records")


if __name__ == "__main__":
    app()
