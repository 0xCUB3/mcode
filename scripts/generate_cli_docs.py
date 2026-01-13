from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (res.stdout or "").rstrip()
    err = (res.stderr or "").rstrip()
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{err}")
    return out


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
