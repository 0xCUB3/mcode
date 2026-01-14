from __future__ import annotations

import hashlib
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


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
        if self._client is not None:
            return self._client
        try:
            import docker

            self._client = docker.from_env()
            return self._client
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Docker is required for SWE-bench Lite evaluation; start Docker and retry."
            ) from e

    def _effective_arch(self) -> str:
        if self.arch is not None:
            arch = self.arch.strip().lower()
            if arch not in {"x86_64", "arm64"}:
                raise ValueError(f"Unsupported SWE-bench arch: {self.arch!r}")
            return arch
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
            import docker
            from swebench.harness.constants import (
                DOCKER_PATCH,
                DOCKER_USER,
                DOCKER_WORKDIR,
                KEY_INSTANCE_ID,
                KEY_MODEL,
                KEY_PREDICTION,
            )
            from swebench.harness.docker_build import build_instance_image
            from swebench.harness.docker_utils import copy_to_container, exec_run_with_timeout
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
                client.images.get(test_spec.instance_image_key)
            except docker.errors.ImageNotFound:  # pragma: no cover
                try:
                    client.images.pull(test_spec.instance_image_key)
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        "Could not pull the SWE-bench prebuilt image "
                        f"{test_spec.instance_image_key!r}. "
                        "The namespace may not contain images for this instance.\n"
                        "Try `--namespace none` (or `--namespace \"\"`) to build locally, "
                        "or use a namespace that you know contains the required images."
                    ) from e
        else:
            # Build locally (relies on env images produced by `prepare_images`).
            # Keep `nocache=False` for speed; swebench uses this flag name.
            build_instance_image(test_spec, client, logger=None, nocache=False)

        container = None
        start = time.time()
        try:
            # Prevent name collisions across retries.
            container_name = f"{test_spec.get_instance_container_name(run_id)}.{patch_sha[:8]}"
            container = client.containers.create(
                image=test_spec.instance_image_key,
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

            # Copy patch to container.
            with tempfile.TemporaryDirectory(prefix="mcode-swebench-") as td:
                td_path = Path(td)
                patch_path = td_path / "patch.diff"
                patch_path.write_text(patch or "", encoding="utf-8", errors="replace")
                copy_to_container(container, patch_path, PurePosixPath(str(DOCKER_PATCH)))

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
            with tempfile.TemporaryDirectory(prefix="mcode-swebench-eval-") as td:
                td_path = Path(td)
                eval_path = td_path / "eval.sh"
                eval_path.write_text(eval_script, encoding="utf-8", errors="replace")
                copy_to_container(container, eval_path, PurePosixPath("/eval.sh"))
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
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
