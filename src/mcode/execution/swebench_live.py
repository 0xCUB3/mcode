from __future__ import annotations

import hashlib
import io
import re
import tarfile
import threading
import time
from dataclasses import dataclass


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
    return f"starryzhang/sweb.eval.x86_64.{sanitized}"


def _parse_pytest_output(output: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for line in output.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR)\s+(.+)$", line.strip())
        if m:
            test_id = m.group(2).strip()
            # Strip pytest's " - ErrorMessage" suffix from FAILED lines
            dash_idx = test_id.find(" - ")
            if dash_idx > 0:
                test_id = test_id[:dash_idx].strip()
            results[test_id] = m.group(1)
            continue
        m = re.match(r"^(.+?)\s+(PASSED|FAILED|ERROR)$", line.strip())
        if m:
            results[m.group(1).strip()] = m.group(2)
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
    # P2P: only count as regression if test FAILED or ERROR'd. MISSING is OK
    # since dataset P2P IDs often have truncated parametrize names.
    all_p2p_pass = all(s not in ("FAILED", "ERROR") for s in p2p_results.values())

    return {
        "resolved": all_f2p_pass and all_p2p_pass,
        "fail_to_pass": f2p_results,
        "pass_to_pass": p2p_results,
    }


class SWEbenchLiveSandbox:
    def __init__(
        self,
        *,
        mem_limit: str = "4g",
        pids_limit: int = 512,
    ):
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import docker

            self._client = docker.from_env()
            return self._client
        except Exception as e:
            raise RuntimeError(
                "Docker is required for SWE-bench Live evaluation; start Docker and retry."
            ) from e

    def evaluate_patch(
        self,
        *,
        task,
        patch: str,
        run_id: str,
        timeout_s: int,
    ) -> SWEbenchLiveRun:
        import docker

        client = self._get_client()
        image_name = _ms_image_name(task.instance_id)
        patch_sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest()

        # Pull image if not present.
        try:
            client.images.get(image_name)
        except docker.errors.ImageNotFound:
            client.images.pull(image_name)

        container = None
        start = time.time()
        timed_out = False
        test_output = ""
        try:
            container_name = (
                f"mcode-sweb-live-{run_id}.{task.instance_id}.{patch_sha[:8]}".replace("__", "-")
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
            )
            container.start()

            # Apply test patch.
            if task.test_patch:
                _copy_to_container(container, "/tmp/test_patch.diff", task.test_patch)
                out, exit_code = _exec_in_container(
                    container,
                    "git apply --verbose /tmp/test_patch.diff",
                    workdir="/testbed",
                    timeout_s=60,
                )
                if exit_code != 0:
                    out2, exit_code2 = _exec_in_container(
                        container,
                        "git apply --verbose --reject /tmp/test_patch.diff",
                        workdir="/testbed",
                        timeout_s=60,
                    )
                    if exit_code2 != 0:
                        runtime_s = time.time() - start
                        return SWEbenchLiveRun(
                            resolved=False,
                            timed_out=False,
                            runtime_s=runtime_s,
                            report={"test_patch_apply_failed": True},
                            test_output=out2,
                            patch_sha256=patch_sha,
                        )

            # Apply solution patch.
            if patch:
                _copy_to_container(container, "/tmp/patch.diff", patch)
                apply_cmds = [
                    "git apply --verbose /tmp/patch.diff",
                    "git apply --verbose --reject /tmp/patch.diff",
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                ]
                applied = False
                last_out = ""
                for cmd in apply_cmds:
                    last_out, exit_code = _exec_in_container(
                        container,
                        cmd,
                        workdir="/testbed",
                        timeout_s=60,
                    )
                    if exit_code == 0:
                        applied = True
                        break
                if not applied:
                    runtime_s = time.time() - start
                    return SWEbenchLiveRun(
                        resolved=False,
                        timed_out=False,
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
                    out, _ = _exec_in_container(
                        container,
                        cmd,
                        workdir="/testbed",
                        timeout_s=timeout_s,
                    )
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
        except Exception as e:
            runtime_s = time.time() - start
            return SWEbenchLiveRun(
                resolved=False,
                timed_out="timed out" in str(e).lower(),
                runtime_s=runtime_s,
                report={"error": str(e)},
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
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    dest_dir = "/".join(dest_path.split("/")[:-1]) or "/"
    container.put_archive(dest_dir, buf)


def _exec_in_container(
    container,
    cmd: str,
    *,
    workdir: str = "/testbed",
    timeout_s: int = 300,
) -> tuple[str, int]:
    result_box: list = []

    def _run():
        try:
            val = container.exec_run(
                ["bash", "-c", cmd],
                workdir=workdir,
            )
            output = (val.output or b"").decode("utf-8", errors="replace")
            result_box.append((output, val.exit_code))
        except Exception as e:
            result_box.append((str(e), -1))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if not result_box:
        return (f"Command timed out after {timeout_s}s: {cmd}", -1)

    return result_box[0]
