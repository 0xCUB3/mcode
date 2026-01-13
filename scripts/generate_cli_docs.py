from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_ANSI_RE = re.compile(
    r"""
    \x1B  # ESC
    (?:
        \[[0-?]*[ -/]*[@-~]  # CSI ... Cmd
      | \][^\x07]*(?:\x07|\x1B\\)  # OSC ... BEL or ST
      | [PX^_].*?\x1B\\  # DCS/PM/APC ... ST
    )
    """,
    re.VERBOSE,
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _run(cmd: list[str]) -> str:
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "TERM": "dumb",
        # Make Typer/Rich output more stable in captured logs.
        "COLUMNS": os.environ.get("COLUMNS", "120"),
    }
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    out = (res.stdout or "").replace("\r\n", "\n").rstrip()
    err = (res.stderr or "").rstrip()
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{err}")
    return _strip_ansi(out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_dir = root / "docs" / "wiki"
    docs_dir.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, list[str]]] = [
        ("mcode --help", ["uv", "run", "-q", "mcode", "--help"]),
        ("mcode results --help", ["uv", "run", "-q", "mcode", "results", "--help"]),
        ("mcode bench --help", ["uv", "run", "-q", "mcode", "bench", "--help"]),
        ("mcode bench humaneval --help", ["uv", "run", "-q", "mcode", "bench", "humaneval", "--help"]),
        ("mcode bench mbpp --help", ["uv", "run", "-q", "mcode", "bench", "mbpp", "--help"]),
    ]

    parts: list[str] = ["# CLI Reference", ""]
    for title, cmd in sections:
        parts.append(f"## `{title}`")
        parts.append("")
        parts.append("```text")
        parts.append(_run(cmd))
        parts.append("```")
        parts.append("")

    (docs_dir / "CLI.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
