from __future__ import annotations

from pathlib import Path

import pytest

from mcode import uv_setup


def test_sync_uv_environment_prefers_sibling_mellea_fork(tmp_path):
    project_root = tmp_path / "mcode"
    site_packages = project_root / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='mcode'\n")
    sibling_fork = tmp_path / "mellea-fork"
    sibling_fork.mkdir()
    (sibling_fork / "pyproject.toml").write_text("[project]\nname='mellea'\n")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path) -> None:
        commands.append(cmd)

    selection = uv_setup.sync_uv_environment(project_root, run_command=fake_run)

    override = site_packages / uv_setup.MANAGED_MELLEA_PTH_NAME
    assert selection.source == "local"
    assert selection.local_path == sibling_fork.resolve()
    assert override.read_text() == (
        "# Managed by mcode deps sync. Do not edit by hand.\n"
        f'import sys; sys.path.insert(0, {str(sibling_fork.resolve())!r})\n'
    )
    assert commands[0] == ["uv", "sync"]
    assert commands[1][:4] == ["uv", "run", "python", "-c"]
    assert str(sibling_fork.resolve()) in commands[1][4]


def test_sync_uv_environment_passes_through_uv_sync_args(tmp_path):
    project_root = tmp_path / "mcode"
    site_packages = project_root / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='mcode'\n")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path) -> None:
        commands.append(cmd)

    uv_setup.sync_uv_environment(
        project_root,
        sync_args=["--extra", "dev", "--extra", "swebench"],
        run_command=fake_run,
    )

    assert commands[0] == ["uv", "sync", "--extra", "dev", "--extra", "swebench"]


def test_sync_uv_environment_falls_back_to_github_without_local_fork(tmp_path):
    project_root = tmp_path / "mcode"
    site_packages = project_root / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='mcode'\n")
    managed_override = site_packages / uv_setup.MANAGED_MELLEA_PTH_NAME
    managed_override.write_text(
        "# Managed by mcode deps sync. Do not edit by hand.\n"
        'import sys; sys.path.insert(0, "/tmp/mellea-fork")\n'
    )

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path) -> None:
        commands.append(cmd)

    selection = uv_setup.sync_uv_environment(project_root, run_command=fake_run)

    assert selection.source == "github"
    assert selection.local_path is None
    assert not managed_override.exists()
    assert commands[0] == ["uv", "sync"]
    assert commands[1][:4] == ["uv", "run", "python", "-c"]
    assert "site-packages" in commands[1][4]


def test_sync_uv_environment_refuses_to_overwrite_unmanaged_override(tmp_path):
    project_root = tmp_path / "mcode"
    site_packages = project_root / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='mcode'\n")
    (site_packages / uv_setup.MANAGED_MELLEA_PTH_NAME).write_text("/tmp/elsewhere\n")

    sibling_fork = tmp_path / "mellea-fork"
    sibling_fork.mkdir()
    (sibling_fork / "pyproject.toml").write_text("[project]\nname='mellea'\n")

    with pytest.raises(RuntimeError, match="Refusing to overwrite existing local override"):
        uv_setup.sync_uv_environment(project_root, run_command=lambda cmd, *, cwd: None)


def test_find_local_mellea_uses_env_override(tmp_path):
    project_root = tmp_path / "mcode"
    project_root.mkdir()
    override = tmp_path / "vendor" / "mellea-custom"
    override.mkdir(parents=True)
    (override / "pyproject.toml").write_text("[project]\nname='mellea'\n")

    resolved = uv_setup.find_local_mellea(
        project_root,
        env={"MCODE_MELLEA_PATH": str(override)},
    )

    assert resolved == override.resolve()


def test_sync_uv_environment_uses_empty_sync_args_by_default(tmp_path):
    project_root = tmp_path / "mcode"
    site_packages = project_root / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='mcode'\n")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: Path) -> None:
        commands.append(cmd)

    uv_setup.sync_uv_environment(project_root, run_command=fake_run)

    assert commands[0] == ["uv", "sync"]
