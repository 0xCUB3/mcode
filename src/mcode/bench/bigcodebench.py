from __future__ import annotations

from mcode.bench.tasks import Task


def load_bigcodebench(cache_dir, *, variant: str = "complete") -> list[Task]:
    _ = cache_dir
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "BigCodeBench requires the `datasets` extra. "
            "Install with `uv pip install -e '.[datasets]'`."
        ) from e

    if variant not in {"complete", "instruct"}:
        raise ValueError(
            f"Unknown BigCodeBench variant: {variant!r}. Expected 'complete' or 'instruct'."
        )

    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    tasks: list[Task] = []
    for row in dataset:
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
