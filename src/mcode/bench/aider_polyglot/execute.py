from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .models import CommandOutcome, PreparedPolyglotTask


def reset_to_baseline(work_dir: Path) -> None:
    subprocess.run(
        ["git", "reset", "--hard", "HEAD"], cwd=work_dir, capture_output=True, check=True
    )
    subprocess.run(["git", "clean", "-fd"], cwd=work_dir, capture_output=True, check=True)


def run_test_commands(task: PreparedPolyglotTask) -> CommandOutcome:
    return run_command_sequence(task.work_dir, task.test_commands, timeout_s=task.timeout_s)


def apply_patch_to_prepared_task(task: PreparedPolyglotTask, patch: str) -> CommandOutcome:
    if not patch.strip():
        return CommandOutcome(
            passed=False,
            output="No patch candidate found.",
            exit_code=None,
            timed_out=False,
        )
    patch_file = task.work_dir / ".mcode-candidate.patch"
    patch_file.write_text(patch, encoding="utf-8")
    try:
        outputs: list[str] = []
        for command in (
            f"git apply --verbose {patch_file.name}",
            f"git apply --verbose --reject {patch_file.name}",
            f"patch --batch --fuzz=5 -p1 -i {patch_file.name}",
        ):
            outcome = run_single_command(task.work_dir, command, timeout_s=task.timeout_s)
            outputs.append(f"$ {command}\n{outcome.output}".strip())
            if outcome.passed:
                return CommandOutcome(
                    passed=True,
                    output="\n\n".join(part for part in outputs if part),
                    exit_code=outcome.exit_code,
                    timed_out=False,
                )
            if outcome.timed_out:
                return CommandOutcome(
                    passed=False,
                    output="\n\n".join(part for part in outputs if part),
                    exit_code=outcome.exit_code,
                    timed_out=True,
                )
        return CommandOutcome(
            passed=False,
            output="\n\n".join(part for part in outputs if part),
            exit_code=1,
            timed_out=False,
        )
    finally:
        if patch_file.exists():
            patch_file.unlink()


def run_command_sequence(
    work_dir: Path,
    commands: tuple[str, ...],
    *,
    timeout_s: int,
) -> CommandOutcome:
    outputs: list[str] = []
    last_exit_code: int | None = None
    for command in commands:
        outcome = run_single_command(work_dir, command, timeout_s=timeout_s)
        last_exit_code = outcome.exit_code
        outputs.append(f"$ {command}\n{outcome.output}".strip())
        if not outcome.passed:
            output = "\n\n".join(part for part in outputs if part)
            if snippets := _failure_report_snippets(work_dir, output):
                output = f"{output}\n\nFailure report snippets:\n{snippets}"
            return CommandOutcome(
                passed=False,
                output=output,
                exit_code=outcome.exit_code,
                timed_out=outcome.timed_out,
            )
    return CommandOutcome(
        passed=True,
        output="\n\n".join(part for part in outputs if part),
        exit_code=last_exit_code,
        timed_out=False,
    )


def _failure_report_snippets(work_dir: Path, output: str = "") -> str:
    from mcode.agent.verification import _collect_failure_artifacts

    return _collect_failure_artifacts(work_dir, output)


def run_single_command(work_dir: Path, command: str, *, timeout_s: int) -> CommandOutcome:
    env = os.environ.copy()
    for extra_bin in (
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/opt/openjdk@21/bin"),
        Path("/opt/homebrew/opt/openjdk/bin"),
    ):
        if extra_bin.is_dir():
            env["PATH"] = f"{extra_bin}:{env.get('PATH', '')}"
    for java_home in (
        Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"),
        Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"),
    ):
        if java_home.is_dir() and not env.get("JAVA_HOME"):
            env["JAVA_HOME"] = str(java_home)
            break
    if command.startswith("cargo "):
        env["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{env.get('PATH', '')}"
    boost_shim = work_dir / ".mcode-boost-shim"
    if boost_shim.is_dir():
        env["BOOST_ROOT"] = str(boost_shim)
        env["BOOST_INCLUDEDIR"] = str(boost_shim / "include")
        env["BOOST_LIBRARYDIR"] = str(boost_shim / "lib")
        env["Boost_NO_SYSTEM_PATHS"] = "ON"
        if command.startswith("cmake -S ") and "BOOST_INCLUDEDIR" not in command:
            boost_config = boost_shim / "lib" / "cmake" / "Boost-1.74.0"
            command = (
                f"{command} -DCMAKE_POLICY_DEFAULT_CMP0167=NEW "
                f"-DBoost_DIR={boost_config} -DCMAKE_PREFIX_PATH={boost_shim} "
                f"-DBOOST_ROOT={boost_shim} "
                f"-DBoost_INCLUDE_DIR={boost_shim / 'include'} "
                f"-DBOOST_INCLUDEDIR={boost_shim / 'include'} "
                f"-DBoost_LIBRARY_DIR_RELEASE={boost_shim / 'lib'} "
                f"-DBoost_DATE_TIME_LIBRARY={boost_shim / 'lib' / 'libboost_date_time.a'} "
                f"-DBOOST_LIBRARYDIR={boost_shim / 'lib'}"
            )
    command_timeout = timeout_s
    if command.startswith("npm install "):
        command_timeout = max(120, timeout_s)

    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=work_dir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=command_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        cleanup_note = _cleanup_after_timed_out_command(work_dir, command)
        output = f"Command timed out after {command_timeout}s"
        if cleanup_note:
            output = f"{output}\n{cleanup_note}"
        return CommandOutcome(
            passed=False,
            output=output,
            exit_code=None,
            timed_out=True,
        )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    return CommandOutcome(
        passed=result.returncode == 0,
        output=output,
        exit_code=result.returncode,
        timed_out=False,
    )


def _cleanup_after_timed_out_command(work_dir: Path, command: str) -> str:
    if "npm install" not in command:
        return ""
    node_modules = work_dir / "node_modules"
    if not node_modules.exists():
        return ""
    shutil.rmtree(node_modules, ignore_errors=True)
    return "Removed partial node_modules after npm install timeout."
