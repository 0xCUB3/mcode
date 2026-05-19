from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mcode.bench.terminalbench import DEFAULT_DATASET
from mcode.launch.models import Check


def doctor(*, deep: bool = False, harbor_executable: str = "harbor") -> list[Check]:
    checks = [
        _python_check(),
        _harbor_check(harbor_executable),
        _docker_check(),
    ]
    if deep:
        checks.append(_oracle_smoke_check(harbor_executable=harbor_executable))
    else:
        checks.append(
            Check(
                name="terminal-bench oracle smoke",
                ok=True,
                detail="skipped (pass --deep to run one Harbor oracle task)",
            )
        )
    return checks


def _python_check() -> Check:
    version = sys.version_info
    detail = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 12):
        return Check(name="python >= 3.12 for Harbor", ok=True, detail=detail)
    return Check(
        name="python >= 3.12 for Harbor",
        ok=False,
        detail=detail,
        next="run Terminal-Bench through a Python 3.12+ environment or `uv tool install harbor`",
    )


def _harbor_check(harbor_executable: str) -> Check:
    command = _harbor_command(harbor_executable)
    if command is None:
        return Check(
            name="harbor executable",
            ok=False,
            detail=f"{harbor_executable!r} not found on PATH",
            next="install with `uv tool install harbor` or pass --harbor-executable to bench",
        )
    try:
        result = subprocess.run(
            [*command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return Check(
            name="harbor executable",
            ok=False,
            detail=f"{harbor_executable}: {type(exc).__name__}: {exc}",
            next="verify Harbor runs outside mCode",
        )
    output = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        return Check(name="harbor executable", ok=True, detail=output or " ".join(command))
    return Check(
        name="harbor executable",
        ok=False,
        detail=(output or f"exit {result.returncode}"),
        next="reinstall or upgrade Harbor with `uv tool install --force harbor`",
    )


def _docker_check() -> Check:
    docker = shutil.which("docker")
    if not docker:
        return Check(
            name="docker daemon",
            ok=False,
            detail="docker not found on PATH",
            next="install/start Docker, or run Harbor with a cloud --env such as daytona/modal",
        )
    try:
        result = subprocess.run(
            [docker, "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return Check(
            name="docker daemon",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            next="start Docker Desktop/Engine and retry",
        )
    if result.returncode == 0:
        return Check(name="docker daemon", ok=True, detail="docker info ok")
    detail = (result.stderr or result.stdout).strip().splitlines()
    return Check(
        name="docker daemon",
        ok=False,
        detail=detail[0] if detail else f"exit {result.returncode}",
        next="start Docker Desktop/Engine, then run `mcode doctor terminal-bench`",
    )


def _oracle_smoke_check(*, harbor_executable: str) -> Check:
    harbor_command = _harbor_command(harbor_executable)
    if harbor_command is None:
        return Check(
            name="terminal-bench oracle smoke",
            ok=False,
            detail="Harbor is not installed",
            next="install with `uv tool install harbor`",
        )
    with tempfile.TemporaryDirectory(prefix="mcode-tb-doctor-") as td:
        jobs_dir = Path(td) / "jobs"
        cmd = [
            *harbor_command,
            "run",
            "-d",
            DEFAULT_DATASET,
            "--agent",
            "oracle",
            "--env",
            "docker",
            "--n-tasks",
            "1",
            "--n-concurrent",
            "1",
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "doctor-oracle-smoke",
            "--yes",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            return Check(
                name="terminal-bench oracle smoke",
                ok=False,
                detail="timed out after 900s",
                next="try `harbor run -d terminal-bench/terminal-bench-2 -a oracle --n-tasks 1`",
            )
        except Exception as exc:
            return Check(
                name="terminal-bench oracle smoke",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                next="run the Harbor oracle command directly for full logs",
            )
    if result.returncode == 0:
        return Check(name="terminal-bench oracle smoke", ok=True, detail="one oracle task passed")
    output = (result.stderr or result.stdout).strip().splitlines()
    return Check(
        name="terminal-bench oracle smoke",
        ok=False,
        detail=output[-1] if output else f"exit {result.returncode}",
        next="run the Harbor oracle command directly for full logs",
    )


def _harbor_command(harbor_executable: str) -> list[str] | None:
    try:
        command = shlex.split(harbor_executable)
    except ValueError:
        return None
    if not command:
        return None
    resolved = shutil.which(command[0])
    if not resolved:
        return None
    return [resolved, *command[1:]]
