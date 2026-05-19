from __future__ import annotations

import io
import os
import re
import tarfile
import threading


def copy_to_container(container: object, dest_path: str, content: str) -> None:
    """Copy text into a container without triggering rootless Podman lchown failures."""
    data = content.encode("utf-8", errors="replace")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=os.path.basename(dest_path))
        info.size = len(data)
        info.uid = 0
        info.gid = 0
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(os.path.dirname(dest_path) or "/", buf)


def truncate_output(output: str, *, max_chars: int = 10_000) -> str:
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


def normalize_agent_command(command: str, *, host_repo_root: str | None = None) -> str:
    normalized = re.sub(
        r"(?<![A-Za-z0-9_./-])/home/user/repos/[^\s/'\";)&|]+",
        "/testbed",
        command,
    )
    for alias in (host_repo_root, "/home/user/repo", "c:/users/user/tmp/repo"):
        if alias:
            normalized = _replace_agent_path_alias(normalized, alias)
    return normalized


def build_agent_shell_command(
    command: str,
    *,
    host_repo_root: str | None = None,
) -> str:
    normalized = normalize_agent_command(command, host_repo_root=host_repo_root)
    preamble = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
    ]
    return "\n".join([*preamble, normalized])


def exec_agent_command_in_container(
    container: object,
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
            result_box.append((truncate_output(output), val.exit_code))
        except Exception as e:
            result_box.append((f"Error: {type(e).__name__}: {e}", -1))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if not result_box:
        return ("", -1, True)
    output, exit_code = result_box[0]
    return (output, exit_code, False)
