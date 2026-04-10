from __future__ import annotations

import shlex
import subprocess

from mcode.launch.models import CommandResult, RunHandle, TargetKind


def follow_run_logs(run: RunHandle) -> CommandResult:
    log_paths = [str(path) for path in run.metadata.get("log_paths", []) if path]
    if not log_paths and run.log_path:
        log_paths = [run.log_path]
    if not log_paths:
        return CommandResult(ok=False, message=f"No log paths recorded for {run.id}")
    try:
        if run.target == TargetKind.BLUEVELA.value:
            login = run.metadata.get("login")
            if not login:
                return CommandResult(ok=False, message=f"No login recorded for {run.id}")
            remote_tail = "tail -n 20 -f " + " ".join(shlex.quote(path) for path in log_paths)
            subprocess.run(
                ["ssh", "-n", login, f"bash -lc {shlex.quote(remote_tail)}"],
                check=False,
            )
        else:
            subprocess.run(["tail", "-n", "20", "-f", *log_paths], check=False)
    except KeyboardInterrupt:
        pass
    return CommandResult(ok=True, message=f"Attached to {run.id}", data=run.metadata)
