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
                    "shards": len(r.shard_job_ids) or len(r.shard_pids),
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
    # whole point of --init is to repair a missing/broken launch.toml; eager
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

    n = state.update(None, _update)
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
    import subprocess

    if target != "bluevela":
        _print_error(
            LaunchError(
                what=f"sync only supports target=bluevela (got {target!r})",
                why="local targets don't need a remote push",
                next="use `rsync` directly or skip sync",
            )
        )
        raise typer.Exit(1)

    cfg = _run(config_mod.load)
    bv = cfg.bluevela
    if not bv.login or not bv.workspace_root:
        _print_error(
            LaunchError(
                what="bluevela config incomplete for sync",
                why="need [bluevela].login and [bluevela].workspace_root",
                next="run `mcode launch doctor bluevela --init`",
            )
        )
        raise typer.Exit(1)

    # Default source: repo root detected via git. Codex review fix: fail
    # closed if we can't detect a repo root. Falling back to cwd is
    # dangerous with --delete — a user running sync from the wrong dir
    # could mirror that dir to the remote and wipe the real workspace.
    if src is None:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            _print_error(
                LaunchError(
                    what="cannot determine source repo",
                    why="not inside a git repo (git rev-parse --show-toplevel failed)",
                    next="pass --src /path/to/repo explicitly",
                )
            )
            raise typer.Exit(1)
        src = Path(r.stdout.strip())

    if not src.is_dir():
        _print_error(
            LaunchError(
                what=f"source {src} is not a directory",
                why="",
                next="pass --src /path/to/repo",
            )
        )
        raise typer.Exit(1)

    # Codex review fix: refuse --delete unless the remote destination has
    # our marker file confirming it's a launcher-managed workspace. Missing
    # marker = user either misconfigured workspace_root to a broader dir,
    # or never ran sync before. In both cases, bail and ask them to `touch`
    # the marker explicitly rather than risk wiping unrelated remote data.
    marker = ".mcode-launch-workspace"
    ssh_opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    # Probe returns three pieces: marker presence, directory emptiness, and
    # a count of entries (so we can report something helpful on error).
    # Codex verify-pass fix: marker missing + non-empty remote = we don't
    # know who owns those files. Refuse rather than risk `rsync --delete`
    # nuking them. Only --bootstrap overrides.
    probe_cmd = (
        f"mkdir -p {bv.workspace_root} && "
        f"if test -f {bv.workspace_root}/{marker}; then echo marker; "
        f'elif [ -z "$(ls -A {bv.workspace_root} 2>/dev/null)" ]; then echo empty; '
        f"else echo populated; fi"
    )
    probe = subprocess.run(
        ["ssh", *ssh_opts, bv.login, probe_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        _print_error(
            LaunchError(
                what="ssh to remote failed during sync preflight",
                why=(probe.stderr or "").strip()[:200],
                next="check VPN + ssh keys; try `mcode launch doctor bluevela`",
            )
        )
        raise typer.Exit(1)

    remote_state = probe.stdout.strip()
    if remote_state == "marker":
        pass  # launcher-owned workspace; --delete is safe
    elif remote_state == "empty":
        # First sync into an empty dir — safe; create marker for future runs.
        r = subprocess.run(
            ["ssh", *ssh_opts, bv.login, f"touch {bv.workspace_root}/{marker}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            _print_error(
                LaunchError(
                    what="failed to create workspace marker",
                    why=(r.stderr or "").strip()[:200],
                    next=f"check write permissions on {bv.workspace_root}",
                )
            )
            raise typer.Exit(1)
        print(f"note: created marker {bv.workspace_root}/{marker}")
    elif remote_state == "populated" and bootstrap:
        print(f"⚠ --bootstrap: treating populated {bv.workspace_root} as owned")
        subprocess.run(
            ["ssh", *ssh_opts, bv.login, f"touch {bv.workspace_root}/{marker}"],
            check=False,
        )
    elif remote_state == "populated":
        _print_error(
            LaunchError(
                what=f"{bv.workspace_root} is non-empty and has no launcher marker",
                why=(
                    f"refusing to `rsync --delete` into a directory we don't own. "
                    f"marker {marker} is missing and the dir has files"
                ),
                next=(
                    f"either (a) pass --bootstrap to claim the dir (destructive!), "
                    f"or (b) `ssh {bv.login} touch {bv.workspace_root}/{marker}` "
                    f"if you manually verified it's safe, or (c) point "
                    f"[bluevela].workspace_root at a fresh path"
                ),
            )
        )
        raise typer.Exit(1)
    else:
        _print_error(
            LaunchError(
                what=f"unexpected remote state probe result: {remote_state!r}",
                why="ssh returned something other than marker/empty/populated",
                next="inspect the remote filesystem manually",
            )
        )
        raise typer.Exit(1)

    dest = f"{bv.login}:{bv.workspace_root}/"
    gitignore = src / ".gitignore"
    # Codex review fix: rsync must go over SSH with the same safety options
    # SshClient uses — no interactive prompts, fast failure on transport.
    ssh_cmd = "ssh " + " ".join(ssh_opts)
    argv = [
        "rsync",
        "-az",
        "-e",
        ssh_cmd,
        "--delete",
        "--stats",
        "-v",
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=__pycache__/",
        "--exclude=.pytest_cache/",
        "--exclude=.ruff_cache/",
        "--exclude=node_modules/",
        # Remote-only dirs that the launcher writes and sync must never wipe.
        "--exclude=bench-runs/",
        "--exclude=runs/",
        "--exclude=benchmarks/",
        f"--exclude={marker}",  # never delete our own safety marker
    ]
    if gitignore.exists():
        argv.append("--filter=:- .gitignore")
    if dry_run:
        argv.append("--dry-run")
    argv += [str(src) + "/", dest]

    print(f"{'preview' if dry_run else 'sync'}: {src} → {dest}")
    print(f"  {' '.join(argv)}")
    r = subprocess.run(argv)
    if r.returncode != 0:
        _print_error(
            LaunchError(
                what=f"rsync exited {r.returncode}",
                why="",
                next="check SSH reachability, quotas, and paths",
            )
        )
        raise typer.Exit(r.returncode)


if __name__ == "__main__":
    app()
