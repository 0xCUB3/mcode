from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from mcode.util import temporary_directory


@contextmanager
def repo_snapshot(repo_root: str, *, enabled: bool):
    if not enabled:
        yield None
        return

    root = Path(repo_root)
    with temporary_directory(prefix="mcode-repo-snapshot-") as td:
        snapshot_dir = Path(td) / "snapshot"
        shutil.copytree(root, snapshot_dir, ignore=_ignore_git, symlinks=True)
        yield snapshot_dir


def restore_repo_snapshot(repo_root: str, snapshot_dir: Path) -> None:
    root = Path(repo_root)
    if not root.is_dir() or not snapshot_dir.is_dir():
        return

    for child in root.iterdir():
        if child.name == ".git":
            continue
        _remove_path(child)

    for child in snapshot_dir.iterdir():
        _copy_path(child, root / child.name)


def get_git_diff(repo_root: str) -> str:
    git_dir = os.path.join(repo_root, ".git")
    if not os.path.exists(git_dir):
        print(f"  [diff] no .git in {repo_root}, cannot produce patch", flush=True)
        return ""

    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [diff] git diff failed: {result.stderr[:300]}", flush=True)
    return result.stdout


def _ignore_git(dir_name: str, names: list[str]) -> set[str]:
    del dir_name
    return {".git"} & set(names)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        _rmtree_with_lustre_retry(path)


def _rmtree_with_lustre_retry(path: Path, *, attempts: int = 6) -> None:
    """Remove a directory tree, retrying on ENOTEMPTY.

    Lustre / GPFS metadata can lag the actual unlink: rmtree walks the tree,
    deletes contents, then issues rmdir on the parent — but the network FS
    has not yet synced its dentry cache, and rmdir fails with ENOTEMPTY even
    though the dir is genuinely empty. A short backoff lets the metadata
    catch up.
    """
    import errno
    import time

    last: OSError | None = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            if e.errno != errno.ENOTEMPTY:
                raise
            last = e
            time.sleep(0.5 * (i + 1))
    if last is not None:
        # Final attempt: ignore_errors so we don't crash the task on cleanup.
        # The path is leaked (cleaned by the bench wrapper's run-dir reaper)
        # but the task result is preserved.
        shutil.rmtree(path, ignore_errors=True)


def _copy_path(src: Path, dest: Path) -> None:
    if src.is_symlink():
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = os.readlink(src)
        dest.symlink_to(target)
        return
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
