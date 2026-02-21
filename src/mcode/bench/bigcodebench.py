from __future__ import annotations

import json
from pathlib import Path

from mcode.bench.tasks import Task

_DATA_FILE = Path(__file__).resolve().parent / "data" / "bigcodebench.json"


def load_bigcodebench(cache_dir, *, variant: str = "complete") -> list[Task]:
    _ = cache_dir
    if variant not in {"complete", "instruct"}:
        raise ValueError(
            f"Unknown BigCodeBench variant: {variant!r}. Expected 'complete' or 'instruct'."
        )

    rows = json.loads(_DATA_FILE.read_text())
    tasks: list[Task] = []
    for row in rows:
        if variant == "complete":
            prompt = row["complete_prompt"]
        else:
            prompt = row["instruct_prompt"]
        tasks.append(
            Task(
                benchmark=f"bigcodebench-{variant}",
                task_id=row["task_id"],
                prompt=prompt,
                entry_point=row["entry_point"],
                test_code=row["test"],
                metadata={
                    "source": "bigcode/bigcodebench",
                    "variant": variant,
                    "libs": row.get("libs", ""),
                },
            )
        )
    return tasks
