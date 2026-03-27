from __future__ import annotations

import hashlib
import platform
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from mcode.execution.sandbox import ensure_docker_client, reraise_docker_unavailable


def _fq_image(name: str) -> str:
    """Fully qualify an image name for podman Docker-compat API."""
    if "/" in name and "." in name.split("/")[0]:
        return name
    return f"docker.io/{name}"


def _copy_to_container_safe(container: object, content: str, dest: str) -> None:
    """Copy content into a container without triggering lchown.

    Rootless podman without subuid ranges fails on the standard
    ``put_archive`` / ``copy_to_container`` because it cannot lchown
    the extracted file.  We build a tar with uid/gid 0 so no ownership
    change is needed.
    """
    import io
    import os
    import tarfile as _tarfile

    data = content.encode("utf-8")
    buf = io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w") as tar:
        info = _tarfile.TarInfo(name=os.path.basename(dest))
        info.size = len(data)
        info.uid = 0
        info.gid = 0
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(os.path.dirname(dest) or "/", buf)


def _ensure_image(client: object, name: str) -> None:
    """Pull *name* if not already present, tolerating podman quirks."""
    import docker

    fq = _fq_image(name)
    try:
        client.images.get(name)
        return
    except docker.errors.ImageNotFound:
        pass
    try:
        client.images.get(fq)
        return
    except docker.errors.ImageNotFound:
        pass
    # Use low-level API; high-level .pull() does a post-pull images.get()
    # that fails on podman due to name normalization differences.
    for line in client.api.pull(fq, stream=True, decode=True):
        if "error" in line:
            raise RuntimeError(line["error"])


def _truncate_command_output(output: str, *, max_chars: int = 10_000) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n[... truncated {len(output) - max_chars} chars ...]\n\n"
        + output[-half:]
    )


def _normalize_agent_command(command: str, *, host_repo_root: str | None = None) -> str:
    if host_repo_root:
        return command.replace(host_repo_root, "/testbed")
    return command


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


def _build_agent_setup_script(eval_script_list: list[str]) -> str:
    setup_commands: list[str] = []
    for raw_command in eval_script_list:
        command = raw_command.strip()
        if not command:
            continue
        if command.startswith("git checkout "):
            break
        if command in {"git status", "git show"}:
            continue
        if command.startswith("git -c core.fileMode=false diff "):
            continue
        if command.startswith("git apply "):
            continue
        if command.startswith(": '>>>>> "):
            continue
        setup_commands.append(command)
    return "\n".join(setup_commands)


def _exec_agent_command_in_container(
    container,
    cmd: str,
    *,
    workdir: str = "/testbed",
    timeout_s: int = 30,
) -> tuple[str, int, bool]:
    result_box: list[tuple[str, int]] = []

    def _run() -> None:
        try:
            val = container.exec_run(
                ["bash", "-o", "pipefail", "-c", cmd],
                workdir=workdir,
            )
            output = (val.output or b"").decode("utf-8", errors="replace")
            result_box.append((_truncate_command_output(output), val.exit_code))
        except Exception as e:
            result_box.append((f"Error: {type(e).__name__}: {e}", -1))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if not result_box:
        return ("", -1, True)
    output, exit_code = result_box[0]
    return (output, exit_code, False)


@dataclass(frozen=True)
class SWEbenchRun:
    resolved: bool
    timed_out: bool
    runtime_s: float
    report: dict
    test_output: str
    patch_sha256: str


class SWEbenchSandbox:
    def __init__(
        self,
        *,
        namespace: str | None = None,
        arch: str | None = None,
        max_workers: int = 4,
        mem_limit: str = "4g",
        pids_limit: int = 512,
        force_rebuild: bool = False,
        base_image_tag: str = "latest",
        env_image_tag: str = "latest",
        instance_image_tag: str = "latest",
    ):
        self.namespace = namespace
        self.arch = arch
        self.max_workers = max_workers
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self.force_rebuild = force_rebuild
        self.base_image_tag = base_image_tag
        self.env_image_tag = env_image_tag
        self.instance_image_tag = instance_image_tag
        self._client = None

    def _get_client(self):
        self._client = ensure_docker_client(self._client, scope="SWE-bench Lite evaluation")
        return self._client

    def _effective_arch(self) -> str:
        if self.arch is not None:
            arch = self.arch.strip().lower()
            if arch not in {"x86_64", "arm64"}:
                raise ValueError(f"Unsupported SWE-bench arch: {self.arch!r}")
            return arch

        # When using prebuilt images (namespace set), prefer x86_64 images for compatibility.
        if self.namespace is not None:
            return "x86_64"

        machine = platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return "arm64"
        return "x86_64"

    @staticmethod
    def _missing_extra_message() -> str:
        return (
            "SWE-bench Lite requires the `swebench` extra. "
            "Install with `uv pip install -e '.[swebench]'`.\n"
            "If you installed `mcode` via `uv tool install ...`, install the extra there too:\n"
            "  `uv tool install -e '.[swebench]'`"
        )

    def prepare_images(self, instances: list[dict]) -> None:
        try:
            from swebench.harness.docker_build import build_env_images
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception as e:  # pragma: no cover
            raise RuntimeError(self._missing_extra_message()) from e

        if self.namespace is not None:
            return

        test_specs = [
            make_test_spec(
                inst,
                namespace=self.namespace,
                base_image_tag=self.base_image_tag,
                env_image_tag=self.env_image_tag,
                instance_image_tag=self.instance_image_tag,
                arch=self._effective_arch(),
            )
            for inst in instances
        ]

        client = self._get_client()
        try:
            build_env_images(
                client,
                test_specs,
                force_rebuild=self.force_rebuild,
                max_workers=self.max_workers,
                namespace=self.namespace,
                instance_image_tag=self.instance_image_tag,
                env_image_tag=self.env_image_tag,
            )
        except Exception as e:
            if self._effective_arch() == "arm64":
                raise RuntimeError(
                    "Failed to build SWE-bench environment images for arm64. "
                    "Some SWE-bench instances pin very old conda packages that aren't available on "
                    "linux-aarch64 (e.g. `setuptools==38.2.4` for Python 3.6).\n"
                    "On Apple Silicon, the easiest workaround is to use prebuilt images:\n"
                    "  `mcode bench swebench-lite --namespace swebench ...`\n"
                    "If you must build locally, try amd64 emulation:\n"
                    "  `mcode bench swebench-lite --namespace none --arch x86_64`\n"
                    "  (add `--max-workers 1` if you hit OOM)"
                ) from e
            raise

    def repo_context(self, instance: dict):
        """Context manager yielding a temp dir with the repo from the task's Docker image."""
        import io
        import shutil
        import tarfile
        from contextlib import contextmanager
        from types import SimpleNamespace

        from mellea.agent.tools.bash import format_tool_result
        from swebench.harness.test_spec.test_spec import make_test_spec

        @contextmanager
        def _ctx():
            client = self._get_client()
            test_spec = make_test_spec(
                instance,
                namespace=self.namespace,
                base_image_tag=self.base_image_tag,
                env_image_tag=self.env_image_tag,
                instance_image_tag=self.instance_image_tag,
                arch=self._effective_arch(),
            )
            _ensure_image(client, test_spec.instance_image_key)

            dest = tempfile.mkdtemp(prefix="mcode-testbed-")
            source_container = client.containers.create(
                image=_fq_image(test_spec.instance_image_key),
                command="true",
            )
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

                exec_container = client.containers.create(
                    image=_fq_image(test_spec.instance_image_key),
                    command="tail -f /dev/null",
                    detach=True,
                    platform=test_spec.platform,
                    volumes={str(testbed): {"bind": "/testbed", "mode": "rw"}},
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    network_disabled=True,
                    mem_limit=self.mem_limit,
                    pids_limit=self.pids_limit,
                )
                exec_container.start()

                setup_script = _build_agent_setup_script(test_spec.eval_script_list)
                if setup_script:
                    setup_output, setup_exit_code, setup_timed_out = (
                        _exec_agent_command_in_container(
                            exec_container,
                            setup_script,
                            timeout_s=300,
                        )
                    )
                    if setup_timed_out:
                        raise RuntimeError("Timed out preparing SWE-bench task environment")
                    if setup_exit_code != 0:
                        raise RuntimeError(
                            "Failed to prepare SWE-bench task environment:\n"
                            f"{setup_output}"
                        )

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
                reraise_docker_unavailable(exc, scope="SWE-bench Lite evaluation")
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

    def evaluate_patch(
        self,
        *,
        instance: dict,
        model_id: str,
        patch: str,
        run_id: str,
        timeout_s: int,
    ) -> SWEbenchRun:
        """
        Apply `patch` to the SWE-bench instance container and execute the official eval script.

        Uses Docker with network disabled during evaluation.
        """
        try:
            from swebench.harness.constants import (
                DOCKER_PATCH,
                DOCKER_USER,
                DOCKER_WORKDIR,
                KEY_INSTANCE_ID,
                KEY_MODEL,
                KEY_PREDICTION,
            )
            from swebench.harness.docker_build import build_instance_image
            from swebench.harness.docker_utils import exec_run_with_timeout
            from swebench.harness.grading import get_eval_report
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception as e:  # pragma: no cover
            raise RuntimeError(self._missing_extra_message()) from e

        client = self._get_client()
        test_spec = make_test_spec(
            instance,
            namespace=self.namespace,
            base_image_tag=self.base_image_tag,
            env_image_tag=self.env_image_tag,
            instance_image_tag=self.instance_image_tag,
            arch=self._effective_arch(),
        )

        patch_sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest()
        pred = {
            KEY_INSTANCE_ID: instance["instance_id"],
            KEY_MODEL: model_id,
            KEY_PREDICTION: patch,
        }

        # Ensure the instance image exists (build locally or pull if namespace provided).
        if test_spec.is_remote_image:
            try:
                _ensure_image(client, test_spec.instance_image_key)
            except Exception as e:  # pragma: no cover
                if test_spec.arch == "arm64":
                    alt_spec = make_test_spec(
                        instance,
                        namespace=self.namespace,
                        base_image_tag=self.base_image_tag,
                        env_image_tag=self.env_image_tag,
                        instance_image_tag=self.instance_image_tag,
                        arch="x86_64",
                    )
                    try:
                        _ensure_image(client, alt_spec.instance_image_key)
                        test_spec = alt_spec
                    except Exception:
                        raise RuntimeError(
                            "Could not pull the SWE-bench prebuilt image "
                            f"{test_spec.instance_image_key!r}. "
                            "Try `--arch x86_64` (recommended) or "
                            "`--namespace none` to build locally."
                        ) from e
                else:
                    raise RuntimeError(
                        "Could not pull the SWE-bench prebuilt image "
                        f"{test_spec.instance_image_key!r}. "
                        "The namespace may not contain images for "
                        "this instance.\n"
                        'Try `--namespace none` (or `--namespace ""`) '
                        "to build locally."
                    ) from e
        else:
            # Build locally (relies on env images produced by `prepare_images`).
            # Keep `nocache=False` for speed; swebench uses this flag name.
            build_instance_image(test_spec, client, logger=None, nocache=False)

        container = None
        start = time.time()
        try:
            # Prevent name collisions across concurrent experiments.
            import uuid

            uid = uuid.uuid4().hex[:8]
            container_name = (
                f"{test_spec.get_instance_container_name(run_id)}.{patch_sha[:8]}.{uid}"
            )
            container = client.containers.create(
                image=_fq_image(test_spec.instance_image_key),
                name=container_name,
                user=DOCKER_USER,
                detach=True,
                command="tail -f /dev/null",
                platform=test_spec.platform,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                network_disabled=True,
                mem_limit=self.mem_limit,
                pids_limit=self.pids_limit,
            )
            container.start()

            # Copy patch to container via exec (avoids put_archive lchown
            # failures in rootless podman without subuid ranges).
            _copy_to_container_safe(container, patch or "", str(DOCKER_PATCH))

            # Apply patch (mirror swebench harness behavior).
            apply_cmds = [
                "git apply --verbose",
                "git apply --verbose --reject",
                "patch --batch --fuzz=5 -p1 -i",
            ]
            applied = False
            last_apply_out = ""
            for cmd in apply_cmds:
                val = container.exec_run(
                    f"{cmd} {DOCKER_PATCH}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                out = (val.output or b"").decode("utf-8", errors="replace")
                last_apply_out = out
                if val.exit_code == 0:
                    applied = True
                    break
            if not applied:
                runtime_s = time.time() - start
                report = {
                    str(instance["instance_id"]): {
                        "patch_is_None": patch is None,
                        "patch_exists": bool(patch),
                        "patch_successfully_applied": False,
                        "resolved": False,
                    }
                }
                return SWEbenchRun(
                    resolved=False,
                    timed_out=False,
                    runtime_s=runtime_s,
                    report=report,
                    test_output=last_apply_out,
                    patch_sha256=patch_sha,
                )

            # Copy eval script and run.
            eval_script = test_spec.eval_script
            _copy_to_container_safe(container, eval_script, "/eval.sh")
            test_output, timed_out, runtime_s = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", timeout_s
            )
            test_output = str(test_output or "")

            # Produce swebench-style report by parsing logs.
            # `get_eval_report` expects a file path; write logs to a temp file.
            with tempfile.TemporaryDirectory(prefix="mcode-swebench-") as td:
                p = Path(td) / "test_output.log"
                p.write_text(test_output, encoding="utf-8", errors="replace")
                report = get_eval_report(
                    test_spec=test_spec,
                    prediction=pred,
                    test_log_path=str(p),
                    include_tests_status=False,
                )

            inst_report = report.get(str(instance["instance_id"]), {})
            resolved = bool(inst_report.get("resolved", False))

            return SWEbenchRun(
                resolved=resolved,
                timed_out=bool(timed_out),
                runtime_s=float(runtime_s),
                report=report,
                test_output=test_output,
                patch_sha256=patch_sha,
            )
        except Exception as exc:
            reraise_docker_unavailable(exc, scope="SWE-bench Lite evaluation")
            raise
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
