from __future__ import annotations

import os
import tempfile
from pathlib import Path


def temp_root() -> Path:
    if override := os.environ.get("MCODE_TMPDIR"):
        return Path(override)
    if workspace_tmp := os.environ.get("WORKSPACE_TMP"):
        return Path(workspace_tmp)
    if cache_dir := os.environ.get("MCODE_CACHE_DIR"):
        return Path(cache_dir) / "tmp"
    if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "mcode" / "tmp"
    return Path.home() / ".cache" / "mcode" / "tmp"


def temporary_directory(*, prefix: str) -> tempfile.TemporaryDirectory[str]:
    root = temp_root()
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=root)


def make_temp_dir(*, prefix: str) -> str:
    root = temp_root()
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=root)
