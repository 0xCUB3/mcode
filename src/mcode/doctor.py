"""Top-level `mcode doctor` — system + launch diagnostics.

With no target argument: runs system-level checks (results DB writable,
podman/docker available, mellea importable, ruff installed) plus all three
launch targets.

With a target argument (`bluevela` / `local-vllm` / `local-ollama`):
delegates to the existing per-target doctors in `mcode.launch`. Same
`--init`/`--login` flags as the legacy `mcode launch doctor` (only valid
with `bluevela`).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from mcode.launch.models import Check


def system_checks() -> list[Check]:
    """Checks that aren't tied to any specific launch target."""
    out: list[Check] = []
    out.append(_check_results_dir())
    out.append(_check_container_runtime())
    out.append(_check_mellea_importable())
    out.append(_check_ruff())
    return out


def _check_results_dir() -> Check:
    target = Path(os.environ.get("MCODE_RESULTS_DIR", "experiments/results"))
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return Check(
            name=f"results dir writable ({target})",
            ok=False,
            detail=str(e),
            next=f"check permissions on {target} or set MCODE_RESULTS_DIR to a writable path",
        )
    probe = target / ".mcode-doctor"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        return Check(
            name=f"results dir writable ({target})",
            ok=False,
            detail=str(e),
            next=f"chmod +w {target}",
        )
    return Check(name=f"results dir writable ({target})", ok=True, detail=str(target))


def _check_container_runtime() -> Check:
    podman = shutil.which("podman")
    docker = shutil.which("docker")
    if podman:
        return Check(name="container runtime", ok=True, detail=f"podman → {podman}")
    if docker:
        return Check(name="container runtime", ok=True, detail=f"docker → {docker}")
    return Check(
        name="container runtime",
        ok=False,
        detail="neither podman nor docker on PATH",
        next="install podman (preferred on Linux/macOS) or Docker Desktop",
    )


def _check_mellea_importable() -> Check:
    spec = importlib.util.find_spec("mellea")
    if spec is None:
        return Check(
            name="mellea importable",
            ok=False,
            detail="`import mellea` fails",
            next="run `uv pip install -e '.[dev]'` to refresh deps",
        )
    return Check(name="mellea importable", ok=True, detail="mellea is on the path")


def _check_ruff() -> Check:
    ruff = shutil.which("ruff")
    if ruff is None:
        if shutil.which("uv"):
            return Check(
                name="ruff",
                ok=True,
                detail="not on PATH but available via `uv run ruff`",
            )
        return Check(
            name="ruff",
            ok=False,
            detail="ruff not on PATH and uv not found",
            next="run `uv pip install -e '.[dev]'`",
        )
    return Check(name="ruff", ok=True, detail=ruff)


def render_check_lines(checks: list[Check]) -> tuple[list[str], bool]:
    """Format checks for human-readable output. Returns (lines, any_failed)."""
    from mcode.ui.styles import ANSI_GREEN, ANSI_RED, ANSI_RESET, color_enabled

    lines: list[str] = []
    any_failed = False
    use_color = color_enabled()
    for c in checks:
        if use_color:
            icon = f"{ANSI_GREEN}✓{ANSI_RESET}" if c.ok else f"{ANSI_RED}✗{ANSI_RESET}"
        else:
            icon = "✓" if c.ok else "✗"
        lines.append(f"{icon} {c.name}")
        if c.detail:
            lines.append(f"  {c.detail}")
        if not c.ok and c.next:
            lines.append(f"  next: {c.next}")
        any_failed = any_failed or not c.ok
    return lines, any_failed


__all__ = ["render_check_lines", "system_checks"]
