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
        contest_date = row.get("contest_date", "")
        if cutoff is not None and contest_date >= cutoff:
            continue
        prompt = _build_prompt(row)
        test_code = _merge_test_cases(row)
        tasks.append(
            Task(
                benchmark="livecodebench",
                task_id=str(row["question_id"]),
                prompt=prompt,
                entry_point=None,
                test_code=test_code,
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


def _merge_test_cases(row: dict) -> str:
    public = _load_test_list(row.get("public_test_cases", ""))
    private = _load_test_list(row.get("private_test_cases", ""))
    all_cases = public + private
    inputs = [c.get("input", "") for c in all_cases]
    outputs = [c.get("output", "") for c in all_cases]
    return json.dumps({"inputs": inputs, "outputs": outputs})


def _load_test_list(raw: str) -> list[dict]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, TypeError):
        return _decompress_test_cases(raw)


def _decompress_test_cases(raw: str) -> list[dict]:
    import base64
    import pickle
    import zlib

    try:
        data = base64.b64decode(raw)
        data = zlib.decompress(data)
        loaded = pickle.loads(data)  # noqa: S301
        if isinstance(loaded, str):
            loaded = json.loads(loaded)
        if isinstance(loaded, list):
            return loaded
        return []
    except Exception:
        return []
