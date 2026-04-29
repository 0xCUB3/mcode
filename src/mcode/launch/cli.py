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
    mcode launch sync bluevela [--dry-run] [--src DIR]
                               (rsync local repo to remote workspace_root)

Output:
- Server launch prints the endpoint as the last line on stdout so callers can
  capture it: `OPENAI_BASE_URL=$(mcode launch local-vllm --model X --json ...)`.
- Errors print to stderr in the formatted layout and exit 1.
"""

from __future__ import annotations

import json
import os
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
from mcode.ui.errors import print_error as _print_error

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Launch vLLM + benchmarks on Blue Vela or locally.",
)


# ---------------------------------------------------------------------------
# error formatting — delegates to mcode.ui.errors so every command in mcode
# uses one formatter; LaunchError is now a subclass of MCodeError.
# ---------------------------------------------------------------------------
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
    from mcode.launch.formatting import format_status_json, format_status_lines

    def load_state():
        try:
            return state.load()
        except Exception as exc:
            raise LaunchError(
                what="could not read launch state",
                why=str(exc),
                next="check MCODE_LAUNCH_STATE or remove the corrupt state file",
            ) from exc

    s = _run(load_state)
    if json_mode:
        print(json.dumps(format_status_json(s, raw=raw), indent=2))
        return
    lines = format_status_lines(s)
    if not lines:
        print("no servers or runs recorded")
        return
    for line in lines:
        print(line)


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
# wait — block until a server is healthy, failed, or times out
# ---------------------------------------------------------------------------
@app.command("wait")
def cmd_wait(
    id: str = typer.Argument(..., help="server id (from `mcode launch status`)"),
    timeout: int = typer.Option(600, "--timeout", min=1, help="give up after N seconds"),
    poll_s: float = typer.Option(2.0, "--poll", min=0.5, help="seconds between polls"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Block until <id> reaches a terminal state. Exits 0 on healthy, 1 on
    failed/stopped, 2 on timeout, 3 on no-such-id."""
    import time

    deadline = time.monotonic() + timeout
    last_status = ""
    while True:
        # Tolerate transient state-file read failures (concurrent partial
        # writes, lock contention). Single corrupt read should not crash
        # a long wait; only deadline expiry does.
        try:
            s = state.load()
        except Exception as e:
            if time.monotonic() >= deadline:
                _print_error(
                    LaunchError(
                        what="failed to read launch state",
                        why=str(e),
                        next="check the state file at $MCODE_LAUNCH_STATE",
                    )
                )
                raise typer.Exit(2) from e
            time.sleep(poll_s)
            continue
        srv = s.server(id)
        if srv is None:
            _print_error(
                LaunchError(
                    what=f"no server with id {id!r}",
                    why="",
                    next="`mcode launch status` to list",
                )
            )
            raise typer.Exit(3)
        last_status = srv.status
        if srv.status == "healthy":
            if json_mode:
                print(json.dumps({"id": srv.id, "status": srv.status, "endpoint": srv.endpoint}))
            else:
                print(f"✓ {srv.id} healthy: {srv.endpoint}")
            return
        if srv.status in ("failed", "stopped"):
            if json_mode:
                print(json.dumps({"id": srv.id, "status": srv.status}))
            else:
                print(f"✗ {srv.id} {srv.status}")
            raise typer.Exit(1)
        if time.monotonic() >= deadline:
            if json_mode:
                print(json.dumps({"id": id, "status": last_status, "timeout_s": timeout}))
            else:
                print(f"⚠ timeout after {timeout}s; last status: {last_status}")
            raise typer.Exit(2)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
@app.command("stop")
def cmd_stop(
    id: str | None = typer.Argument(None, help="server id; omit with --all"),
    all_: bool = typer.Option(False, "--all", help="stop everything in state"),
) -> None:
    """Stop one server by id, or --all to stop every server. Never uses
    `bkill 0` or `-u` — only the caller's recorded jobs.

    Codex verification-pass fixes:
    - Config is loaded lazily, only for Blue Vela targets, so a broken
      [bluevela] TOML never prevents stopping a local server.
    - Honour the return value from bluevela.stop(): False means SSH was
      down and the record was kept as stop-pending. Print a retry hint and
      exit nonzero rather than claiming success.
    """
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

    _cfg: config_mod.LaunchConfig | None = None

    def need_cfg() -> config_mod.LaunchConfig:
        nonlocal _cfg
        if _cfg is None:
            _cfg = _run(config_mod.load)
        return _cfg

    any_failed = False
    for srv in targets:
        try:
            if srv.target == Target.BLUEVELA:
                ok = bluevela.stop(srv.id, cfg=need_cfg())
            elif srv.target == Target.LOCAL_VLLM:
                ok = local_vllm.stop(srv.id)
            elif srv.target == Target.LOCAL_OLLAMA:
                ok = local_ollama.stop(srv.id)
            else:
                ok = False
            if ok:
                print(f"stopped: {srv.id}")
            else:
                any_failed = True
                _print_error(
                    LaunchError(
                        what=f"could not confirm stop of {srv.id}",
                        why="remote kill didn't complete (ssh down?)",
                        next=f"record kept as stop-pending; `mcode launch stop {srv.id}` to retry",
                    )
                )
        except LaunchError as e:
            any_failed = True
            _print_error(e)
    if any_failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@app.command("doctor")
def cmd_doctor(
    target: str = typer.Argument(...),
    deep: bool = typer.Option(False, "--deep"),
    init: bool = typer.Option(False, "--init", help="bootstrap launch.toml for this account"),
    login: str | None = typer.Option(
        None,
        "--login",
        help="user@host for --init (e.g. alice@<your-login-host>)",
    ),
) -> None:
    """Health check for a target. With --init, probe and write launch.toml."""
    # Codex verification-pass fix: don't load existing config for --init. The
    # whole point of --init is to recover a missing/broken launch.toml; eager
    # loading defeats that.
    if init:
        if target != "bluevela":
            _print_error(
                LaunchError(
                    what="--init is only supported for `bluevela`",
                    why=f"target was {target!r}",
                    next="local targets don't need probing — edit launch.toml by hand",
                )
            )
            raise typer.Exit(1)
        if not login:
            login = typer.prompt("Blue Vela login (user@host)")

        def block():
            return bluevela.doctor_init(login=login)

        written = _run(block)
        print(f"wrote {written}")
        print(f"review with `cat {written}` and re-run `mcode launch doctor bluevela`")
        return

    # Health-check path: lazy config load wrapped in _run so a malformed TOML
    # surfaces as a formatted LaunchError, not a traceback.
    cfg = _run(config_mod.load)
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
    """Re-query each server/run against its target and persist updated status.

    Codex final-review fix: config is loaded lazily and only if we have a
    Blue Vela server to refresh. Local records refresh without touching the
    TOML at all.
    """
    _cfg: config_mod.LaunchConfig | None = None

    def need_cfg() -> config_mod.LaunchConfig:
        nonlocal _cfg
        if _cfg is None:
            _cfg = _run(config_mod.load)
        return _cfg

    def _update(s: state.State) -> int:
        count = 0
        for srv in list(s.servers):
            try:
                if srv.target == Target.BLUEVELA:
                    updated = bluevela.refresh(srv, cfg=need_cfg())
                elif srv.target == Target.LOCAL_VLLM:
                    updated = local_vllm.refresh(srv)
                elif srv.target == Target.LOCAL_OLLAMA:
                    # Codex verification-pass fix: pass cfg so custom Ollama
                    # host/port settings are honoured during refresh. Without
                    # this, non-default daemons get refreshed against
                    # 127.0.0.1:11434 and healthy records flip to stopped.
                    updated = local_ollama.refresh(srv, cfg=need_cfg())
                else:
                    continue
            except LaunchError:
                continue
            if isinstance(updated, ServerRecord):
                s.upsert_server(updated)
                count += 1
        return count

    def refresh_state() -> int:
        try:
            return state.update(None, _update)
        except Exception as exc:
            raise LaunchError(
                what="could not refresh launch state",
                why=str(exc),
                next="check MCODE_LAUNCH_STATE or retry after fixing the state file",
            ) from exc

    n = _run(refresh_state)
    print(f"refreshed {n} records")


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
@app.command("sync")
def cmd_sync(
    target: str = typer.Argument(..., help="only `bluevela` is supported"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="preview only"),
    src: Path | None = typer.Option(
        None,
        "--src",
        help="local repo to push (default: cwd's git-tracked root)",
    ),
    bootstrap: bool = typer.Option(
        False,
        "--bootstrap",
        help="allow first sync into a non-empty unmarked remote dir (dangerous with --delete)",
    ),
) -> None:
    """Rsync the local repo to `[bluevela].workspace_root` on the cluster.

    Respects `.gitignore` via `--filter=:- .gitignore`. Use `--dry-run` to
    preview the file list before transferring.
    """
    from mcode.launch.sync import SyncSpec, run_sync

    spec = SyncSpec(target=target, dry_run=dry_run, src=src, bootstrap=bootstrap)

    def block():
        return run_sync(spec)

    result = _run(block)
    if result.rc != 0:
        _print_error(
            LaunchError(
                what=f"rsync exited {result.rc}",
                why="",
                next="check SSH reachability, quotas, and paths",
            )
        )
        raise typer.Exit(result.rc)


if __name__ == "__main__":
    app()
