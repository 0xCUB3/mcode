from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import requests

from mcode.bench.tasks import Task

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def load_mbpp(cache_dir: Path) -> Iterable[Task]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "mbpp" / "mbpp.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        _download(MBPP_URL, path)

    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        task_id = f"MBPP/{row['task_id']}"
        prompt = _prompt_from_row(row)
        test_code = _test_code_from_row(row)
        yield Task(
            benchmark="mbpp",
            task_id=task_id,
            prompt=prompt,
            entry_point=None,
            test_code=test_code,
            metadata={"source": "google-research/mbpp", "raw_task_id": row["task_id"]},
        )


def _prompt_from_row(row: dict) -> str:
    tests = "\n".join(row.get("test_list", []))
    setup = row.get("test_setup_code", "").strip()
    setup_block = f"\n\n# Test setup\n{setup}\n" if setup else ""
    return (
        "Write Python code that solves the following problem.\n"
        "Return only Python code.\n\n"
        f"Problem:\n{row['text'].strip()}\n"
        f"{setup_block}\n"
        f"# Tests\n{tests}\n"
    )


def _test_code_from_row(row: dict) -> str:
    setup = row.get("test_setup_code", "").strip()
    tests = row.get("test_list", [])
    lines: list[str] = []
    if setup:
        lines.append(setup)
    lines.extend(tests)
    return "\n".join(lines) + "\n"


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
