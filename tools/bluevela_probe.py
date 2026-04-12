#!/usr/bin/env python3
"""Blue Vela cluster probe.

Run this from any Blue Vela login node. It captures the facts the launcher
rewrite needs to set sensible defaults: real queue names, whether the cluster
accepts mode=shared vs mode=exclusive_process GPU reservations, whether spjb
front-end tools (jbmon/jbsub/bugroup) exist, and whether bsub -interactive
works.

Output: JSON on stdout plus a sanitized copy at the path given by --out.
The sanitized copy replaces the real username with "<user>" and real group
names with "<group-N>" so the file is safe to commit as a test fixture.

Usage (from a Blue Vela login node):

    python3 tools/bluevela_probe.py --group <your-lsf-group> \
        --out tests/fixtures/bluevela_probe.json

This script runs small LSF commands. The GPU probe job is killed after it
starts (no billable time). Total wall clock ~30-60 s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @classmethod
    def run(cls, cmd: list[str], timeout: float = 30.0) -> CommandResult:
        start = time.monotonic()
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return cls(
                cmd=shlex.join(cmd),
                returncode=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
                duration_s=round(time.monotonic() - start, 3),
            )
        except subprocess.TimeoutExpired:
            return cls(
                cmd=shlex.join(cmd),
                returncode=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                duration_s=round(time.monotonic() - start, 3),
            )
        except FileNotFoundError:
            return cls(
                cmd=shlex.join(cmd),
                returncode=127,
                stdout="",
                stderr=f"command not found: {cmd[0]}",
                duration_s=round(time.monotonic() - start, 3),
            )


@dataclass
class ProbeReport:
    user: str
    host: str
    timestamp: str
    tool_availability: dict[str, bool] = field(default_factory=dict)
    bugroup: CommandResult | None = None
    bqueues_user: CommandResult | None = None
    queue_details: dict[str, CommandResult] = field(default_factory=dict)
    bhosts: CommandResult | None = None
    gpu_mode_shared: dict[str, CommandResult] = field(default_factory=dict)
    gpu_mode_exclusive: dict[str, CommandResult] = field(default_factory=dict)
    bsub_interactive: CommandResult | None = None
    derived: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        def conv(o):
            if isinstance(o, CommandResult):
                return asdict(o)
            if isinstance(o, ProbeReport):
                return {k: conv(v) for k, v in asdict(o).items()}
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(x) for x in o]
            return o

        return json.dumps(conv(self), indent=2, sort_keys=True)


def _check_tools() -> dict[str, bool]:
    return {
        name: shutil.which(name) is not None
        for name in (
            "bsub",
            "bjobs",
            "bkill",
            "bqueues",
            "bhosts",
            "bugroup",
            "jbmon",
            "jbsub",
            "jbinfo",
            "lsid",
        )
    }


def _parse_bqueues_user(stdout: str) -> list[str]:
    # Skip header; first column is queue name
    queues: list[str] = []
    for line in stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("QUEUE_NAME"):
            continue
        parts = line.split()
        if parts:
            queues.append(parts[0])
    return queues


def _build_probe_job(mode: str) -> list[str]:
    # A trivial GPU job that prints nvidia-smi once and exits.
    # -K makes bsub block until job completes (timeout caps total time).
    return [
        "bsub",
        "-K",
        "-J",
        f"mcode-probe-{mode}",
        "-n",
        "1",
        "-R",
        "span[hosts=1]",
        "-gpu",
        f"num=1:mode={mode}",
        "-o",
        "/tmp/mcode-probe.%J.out",
        "-e",
        "/tmp/mcode-probe.%J.err",
        "bash",
        "-c",
        "nvidia-smi -L; exit 0",
    ]


def probe(group: str, queues_to_probe: list[str] | None) -> ProbeReport:
    user = os.environ.get("USER", "unknown")
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    report = ProbeReport(user=user, host=host, timestamp=ts)
    report.tool_availability = _check_tools()

    # bugroup (optional)
    if report.tool_availability.get("bugroup"):
        report.bugroup = CommandResult.run(["bugroup"], timeout=10)

    # bqueues -u $USER
    report.bqueues_user = CommandResult.run(
        ["bqueues", "-u", user, "-o", "QUEUE_NAME PRIO STATUS NJOBS PEND RUN"],
        timeout=15,
    )

    user_queues: list[str] = []
    if report.bqueues_user.returncode == 0:
        user_queues = _parse_bqueues_user(report.bqueues_user.stdout)

    # bqueues -l <q> for each queue the user can see (or a caller-supplied set)
    probe_set = queues_to_probe if queues_to_probe else user_queues[:5]
    for q in probe_set:
        report.queue_details[q] = CommandResult.run(
            ["bqueues", "-l", q],
            timeout=15,
        )

    # bhosts (host summary; useful to see GPU counts per host)
    report.bhosts = CommandResult.run(["bhosts"], timeout=15)

    # GPU mode probes: pick the first user-visible queue unless the caller
    # passed a probe set. Use a short timeout so we don't sit in a long queue.
    probe_queue = probe_set[0] if probe_set else None
    if probe_queue:
        shared_cmd = ["bsub", "-G", group, "-q", probe_queue] + _build_probe_job("shared")[1:]
        excl_cmd = ["bsub", "-G", group, "-q", probe_queue] + _build_probe_job("exclusive_process")[
            1:
        ]
        # 120 s cap — if the job doesn't even start we record that.
        report.gpu_mode_shared[probe_queue] = CommandResult.run(shared_cmd, timeout=120)
        report.gpu_mode_exclusive[probe_queue] = CommandResult.run(excl_cmd, timeout=120)

    # bsub -interactive echo test
    if probe_queue:
        report.bsub_interactive = CommandResult.run(
            [
                "bsub",
                "-G",
                group,
                "-q",
                probe_queue,
                "-interactive",
                "bash",
                "-c",
                "echo mcode-probe-ok",
            ],
            timeout=60,
        )

    # Derive recommendations
    report.derived = _derive(report, user_queues)
    return report


def _derive(report: ProbeReport, user_queues: list[str]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["queue_order_suggestion"] = user_queues[:5]

    # GPU mode
    shared_ok = any(r.returncode == 0 for r in report.gpu_mode_shared.values())
    excl_ok = any(r.returncode == 0 for r in report.gpu_mode_exclusive.values())
    if excl_ok:
        d["gpu_mode_recommendation"] = "exclusive_process"
    elif shared_ok:
        d["gpu_mode_recommendation"] = "shared"
    else:
        d["gpu_mode_recommendation"] = "unknown"
    d["gpu_mode_shared_ok"] = shared_ok
    d["gpu_mode_exclusive_ok"] = excl_ok

    # spjb tools
    d["has_spjb"] = all(
        report.tool_availability.get(t, False) for t in ("jbmon", "jbsub", "jbinfo")
    )

    # bsub -interactive
    d["bsub_interactive_ok"] = (
        report.bsub_interactive is not None
        and report.bsub_interactive.returncode == 0
        and "mcode-probe-ok" in (report.bsub_interactive.stdout or "")
    )
    return d


def sanitize(report_json: str, user: str) -> str:
    # Replace the real username with <user>; replace any grp_* token we see.
    out = report_json.replace(user, "<user>")
    out = re.sub(r"grp_[A-Za-z0-9_]+", "<group>", out)
    out = re.sub(r"/u/[A-Za-z0-9_\-]+", "/u/<user>", out)
    out = re.sub(r"/proj/[A-Za-z0-9_\-]+", "/proj/<project>", out)
    out = re.sub(r"login\d+\.bluevela\.rmf\.ibm\.com", "<login-host>", out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group", required=True, help="LSF group (from `bugroup`) to use on -G for probe jobs"
    )
    p.add_argument(
        "--queue",
        action="append",
        default=None,
        help="Specific queue to probe; repeatable. Default: first 5 from `bqueues -u $USER`",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="Path to write sanitized JSON (safe to commit)"
    )
    p.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Path to write raw (unsanitized) JSON — local diagnostic only",
    )
    args = p.parse_args(argv)

    report = probe(group=args.group, queues_to_probe=args.queue)
    raw = report.to_json()
    sanitized = sanitize(raw, user=report.user)

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(raw)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(sanitized)

    print(sanitized)
    # Short human summary to stderr for the interactive user
    d = report.derived
    print(
        "\n".join(
            [
                "",
                "--- summary ---",
                f"queues visible:     {len(d.get('queue_order_suggestion', []))}",
                f"queue_order:        {d.get('queue_order_suggestion')}",
                f"gpu_mode shared:    {d.get('gpu_mode_shared_ok')}",
                f"gpu_mode exclusive: {d.get('gpu_mode_exclusive_ok')}",
                f"recommend gpu_mode: {d.get('gpu_mode_recommendation')}",
                f"has spjb tools:     {d.get('has_spjb')}",
                f"bsub -interactive:  {d.get('bsub_interactive_ok')}",
            ]
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
