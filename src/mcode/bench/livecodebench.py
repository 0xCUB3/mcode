from __future__ import annotations

import json

from mcode.bench.tasks import Task


def load_livecodebench(cache_dir, *, cutoff: str | None = None) -> list[Task]:
    _ = cache_dir
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "LiveCodeBench requires the `datasets` extra. "
            "Install with `uv pip install -e '.[datasets]'`."
        ) from e

    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="release_v2",
        split="test",
        trust_remote_code=True,
    )
    tasks: list[Task] = []
    for row in dataset:
        if cutoff is not None and row["release_date"] >= cutoff:
            continue
        prompt = _build_prompt(row)
        tasks.append(
            Task(
                benchmark="livecodebench",
                task_id=str(row["question_id"]),
                prompt=prompt,
                entry_point=None,
                test_code=row["input_output"],
                metadata={
                    "source": "livecodebench",
                    "release_date": row["release_date"],
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


def _parse_test_cases(test_code: str) -> dict:
    """Parse the JSON test_code string into a dict with inputs/outputs."""
    try:
        return json.loads(test_code)
    except (json.JSONDecodeError, TypeError):
        return {"inputs": [], "outputs": []}
