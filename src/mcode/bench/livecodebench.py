from __future__ import annotations

import json
from pathlib import Path

from mcode.bench.tasks import Task

_DATA_FILE = Path(__file__).resolve().parent / "data" / "livecodebench.json"


def load_livecodebench(cache_dir, *, cutoff: str | None = None) -> list[Task]:
    _ = cache_dir
    rows = json.loads(_DATA_FILE.read_text())
    tasks: list[Task] = []
    for row in rows:
        contest_date = row.get("contest_date", "")
        if cutoff is not None and contest_date >= cutoff:
            continue
        prompt = _build_prompt(row)
        tasks.append(
            Task(
                benchmark="livecodebench",
                task_id=str(row["question_id"]),
                prompt=prompt,
                entry_point=None,
                test_code=row["test_cases"],
                metadata={
                    "source": "livecodebench",
                    "contest_date": contest_date,
                    "difficulty": row.get("difficulty", ""),
                    "question_title": row.get("question_title", ""),
                },
            )
        )
    return tasks


def _build_prompt(row: dict) -> str:
    content = row["question_content"]
    starter = row.get("starter_code", "") or ""
    prompt = f"Read input from stdin and print output to stdout.\n\n{content}"
    if starter.strip():
        prompt += f"\n\n# Starter code\n{starter}"
    return prompt
