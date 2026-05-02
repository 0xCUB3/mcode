from __future__ import annotations

from pathlib import Path

from mcode.util import make_temp_dir, temp_root, temporary_directory


def test_temp_root_prefers_workspace_tmp(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MCODE_TMPDIR", raising=False)
    monkeypatch.delenv("MCODE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("WORKSPACE_TMP", str(tmp_path / "workspace-tmp"))

    assert temp_root() == tmp_path / "workspace-tmp"


def test_temp_helpers_create_paths_under_workspace_tmp(monkeypatch, tmp_path: Path) -> None:
    workspace_tmp = tmp_path / "workspace-tmp"
    monkeypatch.delenv("MCODE_TMPDIR", raising=False)
    monkeypatch.delenv("MCODE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("WORKSPACE_TMP", str(workspace_tmp))

    created_dir = Path(make_temp_dir(prefix="mcode-testbed-"))
    assert created_dir.parent == workspace_tmp
    assert created_dir.is_dir()

    with temporary_directory(prefix="mcode-snapshot-") as td:
        temp_path = Path(td)
        assert temp_path.parent == workspace_tmp
        assert temp_path.is_dir()
