from __future__ import annotations

import hashlib
import io
import tarfile
import threading
import time
from dataclasses import dataclass

from mcode.execution.sandbox import ensure_docker_client, reraise_docker_unavailable
from mcode.execution.swebench import _ensure_image
from mcode.util import make_temp_dir


@dataclass(frozen=True)
class SWEbenchLiveRun:
    resolved: bool
    timed_out: bool
    runtime_s: float
    report: dict
    test_output: str
    patch_sha256: str


def _ms_image_name(instance_id: str) -> str:
    sanitized = instance_id.replace("__", "_1776_").lower()
    return f"docker.io/starryzhang/sweb.eval.x86_64.{sanitized}"


def _parse_pytest_output(output: str) -> dict[str, str]:
    """Parse pytest -rA output matching official SWE-bench-Live logic."""
    _STATUSES = {"FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL"}
    results: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not any(line.startswith(s) for s in _STATUSES):
            continue
        if line.startswith("FAILED"):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) <= 1:
            continue
        results[parts[1]] = parts[0]
    return results


def _check_resolution(
    test_results: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict:
    f2p_results = {}
    for test_id in fail_to_pass:
        status = test_results.get(test_id, "MISSING")
        f2p_results[test_id] = status

    p2p_results = {}
    for test_id in pass_to_pass:
        status = test_results.get(test_id, "MISSING")
        p2p_results[test_id] = status

    all_f2p_pass = all(s == "PASSED" for s in f2p_results.values()) and len(f2p_results) > 0
    # P2P: only tests that actually ran and FAILED/ERROR count as regressions.
    # Tests missing from output (not run) don't count.
    p2p_regressions = [t for t, s in p2p_results.items() if s in ("FAILED", "ERROR")]

    return {
        "resolved": all_f2p_pass and len(p2p_regressions) == 0,
        "fail_to_pass": f2p_results,
        "pass_to_pass": p2p_results,
        "p2p_regressions": p2p_regressions,
    }


class SWEbenchLiveSandbox:
    def __init__(
        self,
        *,
        mem_limit: str = "4g",
        pids_limit: int = 512,
        cpu_limit: float | None = None,
        check_image_digests: bool = False,
    ):
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self.cpu_limit = cpu_limit if (cpu_limit and cpu_limit > 0) else None
        self.check_image_digests = check_image_digests
        self._client = None

    def _cpu_kwargs(self) -> dict[str, int]:
        if self.cpu_limit is None:
            return {}
        # Same floor as SWEbenchSandbox: a near-zero cpu_limit would yield
        # quota=0 (dropped by docker-py) with cpu_period still set, which is
        # corrupted state. Require ≥1 ms slice (1% of one core).
        quota = int(self.cpu_limit * 100_000)
        if quota < 1_000:
            return {}
        return {"cpu_period": 100_000, "cpu_quota": quota}

    def _thread_env(self) -> dict[str, str]:
        """OpenMP/BLAS thread caps. Rootless podman silently no-ops cpu_quota
        on cgroup-v1 clusters, so we constrain at the library level too."""
        if self.cpu_limit is None:
            return {}
        n = max(1, int(self.cpu_limit))
        s = str(n)
        return {
            "OMP_NUM_THREADS": s,
            "OPENBLAS_NUM_THREADS": s,
            "MKL_NUM_THREADS": s,
            "NUMEXPR_NUM_THREADS": s,
            "VECLIB_MAXIMUM_THREADS": s,
            "BLIS_NUM_THREADS": s,
        }

    def _get_client(self):
        self._client = ensure_docker_client(
            self._client,
            scope="SWE-bench Live evaluation",
            from_env_kwargs={"timeout": 600},
        )
        return self._client

    def prepare_images(self, tasks, *, max_workers: int = 4) -> None:
        """Pre-pull Docker images for all tasks sequentially."""
        import os

        if os.environ.get("MCODE_SKIP_IMAGE_PULL"):
            print(
                f"  [images] skipping pull (MCODE_SKIP_IMAGE_PULL set, {len(tasks)} tasks)",
                flush=True,
            )
            return

        client = self._get_client()

        image_names = sorted({_ms_image_name(task.instance_id) for task in tasks})
        print(f"  [images] checking {len(image_names)} task images...", flush=True)
        counts = {"cached": 0, "pulled": 0, "refreshed": 0}
        failed = 0
        for i, image_name in enumerate(image_names):
            try:
                action = _ensure_image(
                    client,
                    image_name,
                    check_image_digests=self.check_image_digests,
                )
                counts[action] += 1
                if action == "cached" and self.check_image_digests:
                    print(
                        f"  [{i + 1}/{len(image_names)}] cached digest match {image_name}",
                        flush=True,
                    )
                elif action == "refreshed":
                    print(
                        f"  [{i + 1}/{len(image_names)}] refreshed moved tag {image_name}",
                        flush=True,
                    )
                elif action == "pulled":
                    print(
                        f"  [{i + 1}/{len(image_names)}] pulled missing image {image_name}",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"  [{i + 1}/{len(image_names)}] FAILED {image_name}: {e}",
                    flush=True,
                )
                failed += 1
                raise
        print(
            f"  [images] done: {counts['cached']} cached, {counts['pulled']} pulled, "
            f"{counts['refreshed']} refreshed, {failed} failed",
            flush=True,
        )

    def repo_context(self, task):
        """Context manager yielding a temp dir with the repo from the task's Docker image."""
        import shutil
        from contextlib import contextmanager
        from pathlib import Path
        from types import SimpleNamespace

        from mcode.agent.tooling import format_tool_result

        @contextmanager
        def _ctx():
            client = self._get_client()
            image_name = _ms_image_name(task.instance_id)
            _ensure_image(
                client,
                image_name,
                check_image_digests=self.check_image_digests,
            )

            dest = make_temp_dir(prefix="mcode-testbed-")
            source_container = client.containers.create(image=image_name, command="true")
            exec_container = None
            try:
                bits, _ = source_container.get_archive("/testbed")
                buf = io.BytesIO()
                for chunk in bits:
                    buf.write(chunk)
                buf.seek(0)
                with tarfile.open(fileobj=buf) as tar:
                    tar.extractall(dest)
                testbed = Path(dest) / "testbed"
                # Normalize permissions: Docker images may have root-only files.
                import stat

                for p in testbed.rglob("*"):
                    try:
                        mode = p.stat().st_mode
                        if p.is_dir():
                            p.chmod(mode | stat.S_IRWXU)
                        else:
                            p.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)
                    except OSError:
                        pass

                # Ensure .git exists so git diff HEAD can produce patches.
                import subprocess

                git_dir = testbed / ".git"
                if not git_dir.exists():
                    print(
                        "  [testbed] .git missing, initializing repo",
                        flush=True,
                    )
                    subprocess.run(
                        ["git", "init"],
                        cwd=str(testbed),
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "add", "-A"],
                        cwd=str(testbed),
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "commit", "-m", "baseline", "--allow-empty"],
                        cwd=str(testbed),
                        capture_output=True,
                        env={
                            **__import__("os").environ,
                            "GIT_AUTHOR_NAME": "mcode",
                            "GIT_AUTHOR_EMAIL": "mcode@test",
                            "GIT_COMMITTER_NAME": "mcode",
                            "GIT_COMMITTER_EMAIL": "mcode@test",
                        },
                    )
                exec_container = client.containers.create(
                    image=image_name,
                    command="tail -f /dev/null",
                    detach=True,
                    volumes={str(testbed): {"bind": "/testbed", "mode": "rw"}},
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    # Network enabled for agent-side setup scripts; hermetic
                    # grading in evaluate_patch still runs with network off.
                    mem_limit=self.mem_limit,
                    pids_limit=self.pids_limit,
                    environment=self._thread_env(),
                    **self._cpu_kwargs(),
                )
                exec_container.start()

                def command_fn(command: str) -> str:
                    shell_command = _build_agent_shell_command(
                        command,
                        host_repo_root=str(testbed),
                    )
                    output, exit_code, timed_out = _exec_agent_command_in_container(
                        exec_container,
                        shell_command,
                        timeout_s=30,
                    )
                    if timed_out:
                        return format_tool_result(shell_command, "TIMEOUT after 30s", "")
                    if exit_code == 0:
                        return format_tool_result(shell_command, "PASSED", output)
                    if exit_code < 0 and output.startswith("Error:"):
                        return format_tool_result(shell_command, "ERROR", output)
                    return format_tool_result(
                        shell_command,
                        f"FAILED (exit {exit_code})",
                        output,
                    )

                yield SimpleNamespace(
                    repo_root=testbed,
                    visible_repo_root="/testbed",
                    command_fn=command_fn,
                )
            except Exception as exc:
                reraise_docker_unavailable(exc, scope="SWE-bench Live evaluation")
                raise
            finally:
                try:
                    source_container.remove(force=True)
                except Exception:
                    pass
                if exec_container is not None:
                    try:
                        exec_container.remove(force=True)
                    except Exception:
                        pass
                shutil.rmtree(dest, ignore_errors=True)

        return _ctx()

    def remove_image(self, task) -> None:
        try:
            client = self._get_client()
            image_name = _ms_image_name(task.instance_id)
            client.images.remove(image_name, force=True)
        except Exception:
            pass

    def evaluate_patch(
        self,
        *,
        task,
        patch: str,
        run_id: str,
        timeout_s: int,
    ) -> SWEbenchLiveRun:
        import uuid

        client = self._get_client()
        image_name = _ms_image_name(task.instance_id)
        patch_sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest()

        # Ensure image is present (and refresh mutable tags when requested).
        _ensure_image(
            client,
            image_name,
            check_image_digests=self.check_image_digests,
        )

        container = None
        start = time.time()
        timed_out = False
        test_output = ""
        try:
            uid = uuid.uuid4().hex[:8]
            container_name = (
                f"mcode-sweb-live-{run_id}.{task.instance_id}.{patch_sha[:8]}.{uid}".replace(
                    "__", "-"
                )
                .replace("/", "-")
                .lower()[:63]
            )
            container = client.containers.create(
                image=image_name,
                name=container_name,
                detach=True,
                command="tail -f /dev/null",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                network_disabled=True,
                mem_limit=self.mem_limit,
                pids_limit=self.pids_limit,
                environment=self._thread_env(),
                **self._cpu_kwargs(),
            )
            container.start()

            patch_dir = "/testbed/.mcode-tmp"
            patch_path = f"{patch_dir}/patch.diff"
            test_patch_path = f"{patch_dir}/test_patch.diff"
            _exec_in_container(container, f"mkdir -p {patch_dir}", workdir="/testbed", timeout_s=30)

            # Apply test patch.
            if task.test_patch:
                _copy_to_container(container, test_patch_path, task.test_patch)
                out, exit_code, apply_timed_out = _exec_in_container(
                    container,
                    f"git apply --verbose {test_patch_path}",
                    workdir="/testbed",
                    timeout_s=60,
                )
                timed_out = timed_out or apply_timed_out
                if exit_code != 0:
                    out2, exit_code2, reject_timed_out = _exec_in_container(
                        container,
                        f"git apply --verbose --reject {test_patch_path}",
                        workdir="/testbed",
                        timeout_s=60,
                    )
                    timed_out = timed_out or reject_timed_out
                    if exit_code2 != 0:
                        runtime_s = time.time() - start
                        return SWEbenchLiveRun(
                            resolved=False,
                            timed_out=timed_out,
                            runtime_s=runtime_s,
                            report={"test_patch_apply_failed": True},
                            test_output=out2,
                            patch_sha256=patch_sha,
                        )

            # Apply solution patch.
            if patch:
                _copy_to_container(container, patch_path, patch)
                apply_cmds = [
                    f"git apply --verbose {patch_path}",
                    f"git apply --verbose --reject {patch_path}",
                    f"patch --batch --fuzz=5 -p1 -i {patch_path}",
                ]
                applied = False
                last_out = ""
                for cmd in apply_cmds:
                    last_out, exit_code, apply_timed_out = _exec_in_container(
                        container,
                        cmd,
                        workdir="/testbed",
                        timeout_s=60,
                    )
                    timed_out = timed_out or apply_timed_out
                    if exit_code == 0:
                        applied = True
                        break
                if not applied:
                    runtime_s = time.time() - start
                    return SWEbenchLiveRun(
                        resolved=False,
                        timed_out=timed_out,
                        runtime_s=runtime_s,
                        report={
                            "patch_successfully_applied": False,
                            "resolved": False,
                        },
                        test_output=last_out,
                        patch_sha256=patch_sha,
                    )

            # Run test commands.
            all_test_output = []
            for cmd in task.test_cmds:
                if cmd.strip():
                    out, _, test_timed_out = _exec_in_container(
                        container,
                        cmd,
                        workdir="/testbed",
                        timeout_s=timeout_s,
                    )
                    timed_out = timed_out or test_timed_out
                    all_test_output.append(out)
            test_output = "\n".join(all_test_output)

            # Parse test output (log_parser is a tag like "pytest", not code).
            test_results = _parse_pytest_output(test_output)

            # Check resolution.
            report = _check_resolution(
                test_results,
                task.fail_to_pass,
                task.pass_to_pass,
            )
            runtime_s = time.time() - start

            return SWEbenchLiveRun(
                resolved=bool(report["resolved"]),
                timed_out=timed_out,
                runtime_s=runtime_s,
                report=report,
                test_output=test_output,
                patch_sha256=patch_sha,
            )
        except Exception as exc:
            reraise_docker_unavailable(exc, scope="SWE-bench Live evaluation")
            runtime_s = time.time() - start
            return SWEbenchLiveRun(
                resolved=False,
                timed_out="timed out" in str(exc).lower(),
                runtime_s=runtime_s,
                report={"error": str(exc)},
                test_output=test_output,
                patch_sha256=patch_sha,
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def _copy_to_container(container, dest_path: str, content: str) -> None:
    data = content.encode("utf-8", errors="replace")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=dest_path.split("/")[-1])
        info.size = len(data)
        info.uid = 0
        info.gid = 0
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    dest_dir = "/".join(dest_path.split("/")[:-1]) or "/"
    container.put_archive(dest_dir, buf)


def _truncate_agent_output(output: str, *, max_chars: int = 10_000) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n[... truncated {len(output) - max_chars} chars ...]\n\n"
        + output[-half:]
    )


def _normalize_agent_command(command: str, *, host_repo_root: str | None = None) -> str:
    normalized = command
    for alias in (host_repo_root, "/home/user/repo", "c:/users/user/tmp/repo"):
        if alias:
            normalized = normalized.replace(alias, "/testbed")
    return normalized


def _build_agent_shell_command(
    command: str,
    *,
    host_repo_root: str | None = None,
) -> str:
    normalized = _normalize_agent_command(command, host_repo_root=host_repo_root)
    preamble = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
    ]
    return "\n".join([*preamble, normalized])


def _exec_agent_command_in_container(
    container,
    cmd: str,
    *,
    workdir: str = "/testbed",
    timeout_s: int = 30,
) -> tuple[str, int, bool]:
    result_box: list[tuple[str, int]] = []

    def _run():
        try:
            val = container.exec_run(
                ["bash", "-o", "pipefail", "-c", cmd],
                workdir=workdir,
            )
            output = (val.output or b"").decode("utf-8", errors="replace")
            result_box.append((_truncate_agent_output(output), val.exit_code))
        except Exception as e:
            result_box.append((f"Error: {type(e).__name__}: {e}", -1))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if not result_box:
        return ("", -1, True)
    output, exit_code = result_box[0]
    return (output, exit_code, False)


def _exec_in_container(
    container,
    cmd: str,
    *,
    workdir: str = "/testbed",
    timeout_s: int = 300,
) -> tuple[str, int, bool]:
    result_box: list[tuple[str, int]] = []
    error_box: list[BaseException] = []

    def _run():
        try:
            val = container.exec_run(
                ["bash", "-c", cmd],
                workdir=workdir,
            )
            output = (val.output or b"").decode("utf-8", errors="replace")
            result_box.append((output, val.exit_code))
        except Exception as e:
            error_box.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if not result_box and not error_box:
        return (f"Command timed out after {timeout_s}s: {cmd}", -1, True)
    if error_box:
        exc = error_box[0]
        reraise_docker_unavailable(exc, scope="SWE-bench Live evaluation")
        return (str(exc), -1, False)
    output, exit_code = result_box[0]
    return (output, exit_code, False)
