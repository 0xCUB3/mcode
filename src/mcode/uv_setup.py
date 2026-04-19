from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MANAGED_OVERRIDE_HEADER = "# Managed by mcode deps sync. Do not edit by hand."
MANAGED_MELLEA_PTH_NAME = "mcode_local_mellea_override.pth"


@dataclass(frozen=True)
class UvDependencySelection:
    source: str
    local_path: Path | None = None


def find_local_mellea(
    project_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    values = env if env is not None else os.environ
    override = values.get("MCODE_MELLEA_PATH")
    if override:
        candidate = Path(override).expanduser()
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return None


def sync_uv_environment(
    project_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    sync_args: list[str] | None = None,
    run_command=None,
) -> UvDependencySelection:
    project_root = project_root.resolve()
    if not (project_root / "pyproject.toml").is_file():
        raise RuntimeError(f"No pyproject.toml found at {project_root}")

    runner = run_command or _run_command
    local_mellea = find_local_mellea(project_root, env=env)
    selection = UvDependencySelection(source="upstream")
    if local_mellea is not None:
        selection = UvDependencySelection(source="local", local_path=local_mellea)

    runner(["uv", "sync", *(sync_args or [])], cwd=project_root)

    site_packages = _find_site_packages(project_root)
    if local_mellea is not None:
        _write_local_override(site_packages, local_mellea)
    else:
        _remove_local_override(site_packages)

    runner(["uv", "run", "python", "-c", _build_runtime_check(selection)], cwd=project_root)
    return selection


def _build_runtime_check(selection: UvDependencySelection) -> str:
    expected = "site-packages"
    if selection.local_path is not None:
        expected = str(selection.local_path)
    return (
        "import importlib, pathlib; "
        "mellea = importlib.import_module('mellea'); "
        "session = importlib.import_module('mellea.stdlib.session'); "
        "react = importlib.import_module('mellea.stdlib.frameworks.react'); "
        "telemetry = importlib.import_module('mellea.telemetry'); "
        "plugins = importlib.import_module('mellea.plugins.pluginset'); "
        "compare = importlib.import_module('mellea.eval.compare'); "
        "compat = importlib.import_module('mcode.mellea_compat'); "
        "location = str(pathlib.Path(mellea.__file__).resolve()); "
        f"expected = {expected!r}; "
        "assert expected in location, f'expected {expected} in {location}'; "
        "assert session is not None; "
        "assert react is not None; "
        "assert telemetry is not None; "
        "assert plugins is not None; "
        "assert compare is not None; "
        "assert compat.import_requirements() is not None; "
        "assert compat.import_sampling() is not None"
    )


def _find_site_packages(project_root: Path) -> Path:
    windows_path = project_root / ".venv" / "Lib" / "site-packages"
    if windows_path.is_dir():
        return windows_path

    matches = sorted((project_root / ".venv" / "lib").glob("python*/site-packages"))
    if matches:
        return matches[0]

    raise RuntimeError(f"Could not find site-packages under {project_root / '.venv'}")


def _render_local_override(local_mellea: Path) -> str:
    return f"{MANAGED_OVERRIDE_HEADER}\nimport sys; sys.path.insert(0, {str(local_mellea)!r})\n"


def _write_local_override(site_packages: Path, local_mellea: Path) -> None:
    override_path = site_packages / MANAGED_MELLEA_PTH_NAME
    if override_path.exists():
        current = override_path.read_text()
        if not current.startswith(MANAGED_OVERRIDE_HEADER):
            raise RuntimeError(
                "Refusing to overwrite existing local override because it is not managed by mcode."
            )
    override_path.write_text(_render_local_override(local_mellea))


def _remove_local_override(site_packages: Path) -> None:
    override_path = site_packages / MANAGED_MELLEA_PTH_NAME
    if not override_path.exists():
        return
    current = override_path.read_text()
    if not current.startswith(MANAGED_OVERRIDE_HEADER):
        raise RuntimeError(
            "Refusing to overwrite existing local override because it is not managed by mcode."
        )
    override_path.unlink()


def _run_command(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)
