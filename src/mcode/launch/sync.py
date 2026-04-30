"""Rsync the local repo to Blue Vela `[bluevela].workspace_root`.

Extracted from `cmd_sync` so the CLI handler is a thin shim and the rsync
machinery is unit-testable on its own.

Safety design:
- Refuses `rsync --delete` into a remote dir that doesn't have our marker
  file unless `bootstrap=True`. Marker prevents accidental wipe of an
  unrelated directory.
- ssh options match SshClient defaults so behavior is consistent across
  every remote operation.
- `.git/`, `.venv/`, caches, plus launcher-owned remote dirs (`runs/`,
  `bench-runs/`, `benchmarks/`) are excluded so a sync never wipes
  in-flight job artifacts.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mcode.launch import config as config_mod
from mcode.launch.models import LaunchError

_MARKER = ".mcode-launch-workspace"
_SSH_OPTS = (
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
)


@dataclass
class SyncSpec:
    target: str  # only "bluevela" supported
    dry_run: bool = False
    src: Path | None = None  # auto-detect via git rev-parse if None
    bootstrap: bool = False


@dataclass
class SyncResult:
    rc: int  # 0 = ok; non-zero = rsync exit code
    src: Path
    dest: str
    dry_run: bool


def run_sync(spec: SyncSpec, *, cfg: config_mod.LaunchConfig | None = None) -> SyncResult:
    """Run sync end-to-end. Raises LaunchError on user-actionable problems
    before invoking rsync. Returns a SyncResult with the rsync exit code so
    the CLI shim can format/exit accordingly."""
    if spec.target != "bluevela":
        raise LaunchError(
            what=f"sync only supports target=bluevela (got {spec.target!r})",
            why="local targets don't need a remote push",
            next="use `rsync` directly or skip sync",
        )

    cfg = cfg or config_mod.load()
    bv = cfg.bluevela
    if not bv.login or not bv.workspace_root:
        raise LaunchError(
            what="bluevela config incomplete for sync",
            why="need [bluevela].login and [bluevela].workspace_root",
            next="run `mcode launch doctor bluevela --init`",
        )

    src = spec.src or _detect_repo_root()
    if not src.is_dir():
        raise LaunchError(
            what=f"source {src} is not a directory",
            why="",
            next="pass --src /path/to/repo",
        )

    _check_remote_marker(login=bv.login, workspace_root=bv.workspace_root, bootstrap=spec.bootstrap)

    dest = f"{bv.login}:{bv.workspace_root}/"
    rc = _rsync(src=src, dest=dest, dry_run=spec.dry_run)
    return SyncResult(rc=rc, src=src, dest=dest, dry_run=spec.dry_run)


def _detect_repo_root() -> Path:
    """Find the git repo root containing cwd. Failure is fatal: falling back
    to cwd is dangerous with `rsync --delete` since the user could mirror an
    unrelated dir to the remote and wipe the real workspace."""
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise LaunchError(
            what="cannot determine source repo",
            why="not inside a git repo (git rev-parse --show-toplevel failed)",
            next="pass --src /path/to/repo explicitly",
        )
    return Path(r.stdout.strip())


def _check_remote_marker(*, login: str, workspace_root: str, bootstrap: bool) -> None:
    """Probe the remote workspace_root and assert it's safe for `rsync --delete`.

    Three branches:
    - marker present: launcher-owned, sync proceeds.
    - empty dir: first sync, create marker for future runs.
    - populated without marker: refuse unless --bootstrap.
    """
    probe_cmd = (
        f"mkdir -p {workspace_root} && "
        f"if test -f {workspace_root}/{_MARKER}; then echo marker; "
        f'elif [ -z "$(ls -A {workspace_root} 2>/dev/null)" ]; then echo empty; '
        f"else echo populated; fi"
    )
    probe = subprocess.run(
        ["ssh", *_SSH_OPTS, login, probe_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise LaunchError(
            what="ssh to remote failed during sync preflight",
            why=(probe.stderr or "").strip()[:200],
            next="check VPN + ssh keys; try `mcode launch doctor bluevela`",
        )
    state = probe.stdout.strip()
    if state == "marker":
        return
    if state == "empty":
        r = subprocess.run(
            ["ssh", *_SSH_OPTS, login, f"touch {workspace_root}/{_MARKER}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            raise LaunchError(
                what="failed to create workspace marker",
                why=(r.stderr or "").strip()[:200],
                next=f"check write permissions on {workspace_root}",
            )
        print(f"note: created marker {workspace_root}/{_MARKER}")
        return
    if state == "populated" and bootstrap:
        print(f"⚠ --bootstrap: treating populated {workspace_root} as owned")
        subprocess.run(
            ["ssh", *_SSH_OPTS, login, f"touch {workspace_root}/{_MARKER}"],
            check=False,
        )
        return
    if state == "populated":
        raise LaunchError(
            what=f"{workspace_root} is non-empty and has no launcher marker",
            why=(
                f"refusing to `rsync --delete` into a directory we don't own. "
                f"marker {_MARKER} is missing and the dir has files"
            ),
            next=(
                f"either (a) pass --bootstrap to claim the dir (destructive!), "
                f"or (b) `ssh {login} touch {workspace_root}/{_MARKER}` "
                f"if you manually verified it's safe, or (c) point "
                f"[bluevela].workspace_root at a fresh path"
            ),
        )
    raise LaunchError(
        what=f"unexpected remote state probe result: {state!r}",
        why="ssh returned something other than marker/empty/populated",
        next="inspect the remote filesystem manually",
    )


def _rsync(*, src: Path, dest: str, dry_run: bool) -> int:
    """Run rsync; return its exit code. Print the command line for debugging."""
    gitignore = src / ".gitignore"
    ssh_cmd = "ssh " + " ".join(_SSH_OPTS)
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
        "--exclude=.uv-cache/",
        "--exclude=.bluevela-reruns/",
        "--exclude=research/",
        "--exclude=experiments/",
        # Remote-only dirs the launcher writes; sync must never wipe these.
        "--exclude=bench-runs/",
        "--exclude=runs/",
        "--exclude=benchmarks/",
        "--exclude=podman-tmp/",
        f"--exclude={_MARKER}",
    ]
    if gitignore.exists():
        argv.append("--filter=:- .gitignore")
    if dry_run:
        argv.append("--dry-run")
    argv += [str(src) + "/", dest]
    print(f"{'preview' if dry_run else 'sync'}: {src} → {dest}")
    print(f"  {' '.join(argv)}")
    return subprocess.run(argv).returncode


__all__ = ["SyncResult", "SyncSpec", "run_sync"]
