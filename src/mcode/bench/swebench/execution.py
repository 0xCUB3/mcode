from __future__ import annotations

import hashlib
import os
import platform
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mcode.bench.swebench.docker import (
    ensure_docker_client,
    is_docker_unavailable_error,
    reraise_docker_unavailable,
)
from mcode.util import make_temp_dir, temporary_directory


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


_RETRYABLE_PODMAN_IMAGE_PATTERNS = (
    "writing blob",
    "adding layer",
    "unpacking failed",
    "chown error detected",
    "insufficient uids or gids",
    "podman system migrate",
    "disk i/o error",
    "database is locked",
    "podman socket did not come up",
    "no such container",
    # Docker Hub upstream timeouts under sharded parallel pulls. Observed when
    # 4 shards race for the same registry endpoint.
    "504 gateway time-out",
    "503 service unavailable",
    "502 bad gateway",
    "fetching blob: received unexpected http status",
    "reading blob",
    "i/o timeout",
)


class RetryablePodmanImageError(RuntimeError):
    pass


def _is_retryable_podman_image_error(exc_or_text: object) -> bool:
    if isinstance(exc_or_text, RetryablePodmanImageError):
        return True
    if isinstance(exc_or_text, BaseException) and is_docker_unavailable_error(exc_or_text):
        return True
    text = str(exc_or_text).lower()
    return any(pattern in text for pattern in _RETRYABLE_PODMAN_IMAGE_PATTERNS)


@contextmanager
def _podman_image_pull_lock():
    import fcntl

    lock_dir = Path(
        os.environ.get("MCODE_PODMAN_LOCK_DIR")
        or (Path.home() / ".cache" / "mcode" / "podman-lock")
    )
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "mcode-podman-images.lock").open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _parse_env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 0/1/true/false/yes/no/on/off (got {raw!r})")


def _container_cpu_kwargs(cpu_limit: float | None) -> dict[str, int]:
    if cpu_limit is None:
        return {}
    # Floor tiny positive limits to a valid quota. int(0.0001 * 100_000) == 0,
    # and docker-py drops the zero quota while keeping cpu_period, leaving a
    # corrupted container config. Require at least a 1 ms slice (1% of one core).
    quota = int(cpu_limit * 100_000)
    if quota < 1_000:
        return {}
    return {"cpu_period": 100_000, "cpu_quota": quota}


def _thread_limit_env(cpu_limit: float | None) -> dict[str, str]:
    if cpu_limit is None:
        return {}
    n = max(1, int(cpu_limit))
    s = str(n)
    return {
        "OMP_NUM_THREADS": s,
        "OPENBLAS_NUM_THREADS": s,
        "MKL_NUM_THREADS": s,
        "NUMEXPR_NUM_THREADS": s,
        "VECLIB_MAXIMUM_THREADS": s,
        "BLIS_NUM_THREADS": s,
    }


def _copy_testbed_from_container(container: object, dest: Path) -> Path:
    import io
    import tarfile

    bits, _ = container.get_archive("/testbed")
    buf = io.BytesIO()
    for chunk in bits:
        buf.write(chunk)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        tar.extractall(dest)
    return dest / "testbed"


def _get_local_image(client: object, name: str, fq: str, docker_module: object) -> object | None:
    for candidate in (name, fq):
        try:
            return client.images.get(candidate)
        except docker_module.errors.ImageNotFound:
            continue
    return None


def _local_repo_digests(image: object) -> set[str]:
    attrs = getattr(image, "attrs", None) or {}
    if not isinstance(attrs, dict):
        return set()
    repo_digests = attrs.get("RepoDigests") or []
    if not isinstance(repo_digests, list):
        return set()
    digests: set[str] = set()
    for item in repo_digests:
        if not isinstance(item, str):
            continue
        digests.add(item)
        if "@" in item:
            digests.add(item.split("@", 1)[1])
    return digests


def _inspect_distribution_digest(client: object, *names: str) -> str:
    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in names:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            response = client.api.inspect_distribution(candidate)
        except Exception as exc:
            last_error = exc
            continue
        descriptor = response.get("Descriptor") or response.get("descriptor")
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"registry digest check returned no descriptor for {candidate}")
        digest = descriptor.get("digest")
        if not isinstance(digest, str) or not digest:
            raise RuntimeError(f"registry digest check returned no digest for {candidate}")
        return digest
    if last_error is not None:
        raise last_error
    raise RuntimeError("registry digest check had no image names to inspect")


def _pull_image_once(client: object, fq: str) -> None:
    # Use low-level API; high-level .pull() does a post-pull images.get()
    # that fails on podman due to name normalization differences.
    for line in client.api.pull(fq, stream=True, decode=True):
        if "error" in line:
            raise RuntimeError(line["error"])


def _ensure_image(
    client: object,
    name: str,
    *,
    check_image_digests: bool | None = None,
) -> str:
    """Ensure *name* is present, optionally refreshing mutable tags by digest.

    Returns one of ``cached``, ``pulled``, or ``refreshed``.
    """
    import docker

    fq = _fq_image(name)
    check_image_digests = (
        _parse_env_bool("MCODE_SWEBENCH_CHECK_IMAGE_DIGESTS", default=False)
        if check_image_digests is None
        else check_image_digests
    )
    attempts_raw = os.environ.get("MCODE_PODMAN_PULL_ATTEMPTS", "2")
    delay_raw = os.environ.get("MCODE_PODMAN_PULL_RETRY_DELAY", "2")
    try:
        attempts = max(1, int(attempts_raw))
    except ValueError:
        raise ValueError(f"MCODE_PODMAN_PULL_ATTEMPTS must be an int (got {attempts_raw!r})")
    try:
        delay = max(0.0, float(delay_raw))
    except ValueError:
        raise ValueError(f"MCODE_PODMAN_PULL_RETRY_DELAY must be a float (got {delay_raw!r})")

    from mcode.util.retry import with_backoff

    def with_podman_backoff(fn):
        return with_backoff(
            fn,
            is_retryable=_is_retryable_podman_image_error,
            max_attempts=attempts,
            base_sleep_s=delay if delay > 0 else 0.001,
            max_sleep_s=max(delay, 30.0),
        )

    with _podman_image_pull_lock():
        try:
            local_image = with_podman_backoff(lambda: _get_local_image(client, name, fq, docker))
            if local_image is not None:
                if not check_image_digests:
                    return "cached"
                try:
                    registry_digest = with_podman_backoff(
                        lambda: _inspect_distribution_digest(client, fq, name)
                    )
                except Exception as last_error:
                    raise RetryablePodmanImageError(
                        f"Retryable podman image digest check failed for {fq}: {last_error}"
                    ) from last_error
                if registry_digest in _local_repo_digests(local_image):
                    return "cached"
                pull_action = "refreshed"
            else:
                pull_action = "pulled"
            with_podman_backoff(lambda: _pull_image_once(client, fq))
            return pull_action
        except RetryablePodmanImageError:
            raise
        except Exception as last_error:
            if _is_retryable_podman_image_error(last_error):
                raise RetryablePodmanImageError(
                    f"Retryable podman image pull failed for {fq}: {last_error}"
                ) from last_error
            raise


def _truncate_command_output(output: str, *, max_chars: int = 10_000) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n[... truncated {len(output) - max_chars} chars ...]\n\n"
        + output[-half:]
    )


def _replace_agent_path_alias(command: str, alias: str, replacement: str = "/testbed") -> str:
    normalized_alias = alias.rstrip("/")
    if not normalized_alias:
        return command
    pattern = re.compile(rf"(?<![A-Za-z0-9_./-]){re.escape(normalized_alias)}(?=$|[\s/'\";)&|])")
    return pattern.sub(replacement, command)


def _normalize_agent_command(command: str, *, host_repo_root: str | None = None) -> str:
    normalized = command
    normalized = re.sub(
        r"(?<![A-Za-z0-9_./-])/home/user/repos/[^\s/'\";)&|]+",
        "/testbed",
        normalized,
    )
    for alias in (host_repo_root, "/home/user/repo", "c:/users/user/tmp/repo"):
        if alias:
            normalized = _replace_agent_path_alias(normalized, alias)
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


def _extract_agent_test_commands(eval_script_list: list[str]) -> list[str]:
    commands: list[str] = []
    in_test_output = False
    for raw_command in eval_script_list:
        command = raw_command.strip()
        if command == ": '>>>>> Start Test Output'":
            in_test_output = True
            continue
        if command == ": '>>>>> End Test Output'":
            break
        if in_test_output and command:
            commands.append(command)
    return commands


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
        cpu_limit: float | None = None,
        force_rebuild: bool = False,
        base_image_tag: str = "latest",
        env_image_tag: str = "latest",
        instance_image_tag: str = "latest",
        check_image_digests: bool = True,
    ):
        self.namespace = namespace
        self.arch = arch
        self.max_workers = max_workers
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        # cpu_limit caps each podman/docker container's CPU. None / 0 / negative
        # → no cap (legacy behavior). Avoids login-node admin auto-killer when
        # rootless containers' in-namespace pytest spikes to >100 cores.
        self.cpu_limit = cpu_limit if (cpu_limit and cpu_limit > 0) else None
        self.force_rebuild = force_rebuild
        self.base_image_tag = base_image_tag
        self.env_image_tag = env_image_tag
        self.instance_image_tag = instance_image_tag
        self.check_image_digests = check_image_digests
        self._client = None

    def _cpu_kwargs(self) -> dict[str, int]:
        return _container_cpu_kwargs(self.cpu_limit)

    def _thread_env(self) -> dict[str, str]:
        return _thread_limit_env(self.cpu_limit)

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

        if self.namespace is not None:
            client = self._get_client()
            image_keys = sorted({spec.instance_image_key for spec in test_specs})
            counts = {"cached": 0, "pulled": 0, "refreshed": 0}
            print(f"  [images] checking {len(image_keys)} prebuilt images...", flush=True)
            for image_key in image_keys:
                action = _ensure_image(
                    client,
                    image_key,
                    check_image_digests=self.check_image_digests,
                )
                counts[action] += 1
                if action == "cached" and self.check_image_digests:
                    print(f"  [images] cached digest match {image_key}", flush=True)
                elif action == "refreshed":
                    print(f"  [images] refreshed moved tag {image_key}", flush=True)
                elif action == "pulled":
                    print(f"  [images] pulled missing image {image_key}", flush=True)
            print(
                f"  [images] done: {counts['cached']} cached, {counts['pulled']} pulled, "
                f"{counts['refreshed']} refreshed",
                flush=True,
            )
            return

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
        import shutil
        from contextlib import contextmanager
        from types import SimpleNamespace

        from swebench.harness.test_spec.test_spec import make_test_spec

        from mcode.agent.tooling import format_tool_result

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
            _ensure_image(
                client,
                test_spec.instance_image_key,
                check_image_digests=self.check_image_digests,
            )

            dest = make_temp_dir(prefix="mcode-testbed-")
            source_container = client.containers.create(
                image=_fq_image(test_spec.instance_image_key),
                command="true",
                network_disabled=True,
            )
            exec_container = None
            try:
                testbed = _copy_testbed_from_container(source_container, Path(dest))

                exec_container = client.containers.create(
                    image=_fq_image(test_spec.instance_image_key),
                    command="tail -f /dev/null",
                    detach=True,
                    platform=test_spec.platform,
                    volumes={str(testbed): {"bind": "/testbed", "mode": "rw"}},
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    # Network enabled: per-task setup scripts (e.g. astropy's
                    # `pip install -e .`) fetch build deps from pypi. The
                    # hermetic grading in evaluate_patch keeps network off.
                    mem_limit=self.mem_limit,
                    pids_limit=self.pids_limit,
                    environment=self._thread_env(),
                    **self._cpu_kwargs(),
                )
                exec_container.start()

                setup_script = _build_agent_setup_script(test_spec.eval_script_list)
                test_commands = _extract_agent_test_commands(test_spec.eval_script_list)
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
                            f"Failed to prepare SWE-bench task environment:\n{setup_output}"
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
                    test_cmds={"test_cmds": test_commands} if test_commands else None,
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
                _ensure_image(
                    client,
                    test_spec.instance_image_key,
                    check_image_digests=self.check_image_digests,
                )
            except Exception as e:  # pragma: no cover
                reraise_docker_unavailable(e, scope="SWE-bench Lite image pull")
                if _is_retryable_podman_image_error(e):
                    raise
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
                        _ensure_image(
                            client,
                            alt_spec.instance_image_key,
                            check_image_digests=self.check_image_digests,
                        )
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
                environment=self._thread_env(),
                **self._cpu_kwargs(),
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
            with temporary_directory(prefix="mcode-swebench-") as td:
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
