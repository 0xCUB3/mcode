from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterable
from pathlib import Path

import requests

from mcode.bench.tasks import Task

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def load_humaneval(cache_dir: Path) -> Iterable[Task]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "humaneval" / "HumanEval.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        _download(HUMANEVAL_URL, path)

    raw = gzip.decompress(path.read_bytes()).decode("utf-8")
    for line in raw.splitlines():
        row = json.loads(line)
        yield Task(
            benchmark="humaneval",
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=row.get("entry_point"),
            test_code=row["test"],
            metadata={"source": "openai/human-eval"},
        )


def _download(url: str, dest: Path) -> None:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except Exception:  # pragma: no cover
            if attempt >= 2:
                raise
            time.sleep(2**attempt)
