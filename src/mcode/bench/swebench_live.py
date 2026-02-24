from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SWEbenchLiveTask:
    benchmark: str
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    test_patch: str
    test_cmds: list[str]
    rebuild_cmds: list[str]
    log_parser: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    difficulty: str
    language: str
    raw_instance: dict = field(repr=False)


def _parse_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [value]
    return []


def load_swebench_live(
    cache_dir: Path | None,
    *,
    split: str = "verified",
    limit: int | None = None,
    instance_ids: list[str] | None = None,
) -> list[SWEbenchLiveTask]:
    _ = cache_dir
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "SWE-bench Live requires the `datasets` extra. "
            "Install with `uv pip install -e '.[datasets]'`."
        ) from e

    ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split=split)
    instances: list[dict] = list(ds)

    if instance_ids is not None:
        id_set = set(instance_ids)
        instances = [inst for inst in instances if inst["instance_id"] in id_set]

    if limit is not None:
        instances = instances[:limit]

    tasks: list[SWEbenchLiveTask] = []
    for inst in instances:
        tasks.append(
            SWEbenchLiveTask(
                benchmark="swebench-live",
                instance_id=str(inst["instance_id"]),
                repo=str(inst["repo"]),
                base_commit=str(inst.get("base_commit", "")),
                problem_statement=str(inst.get("problem_statement", "")),
                hints_text=str(inst.get("hints_text", "")),
                version=str(inst.get("version", "")),
                test_patch=str(inst.get("test_patch", "")),
                test_cmds=_parse_list(inst.get("test_cmd", [])),
                rebuild_cmds=_parse_list(inst.get("install_cmd", [])),
                log_parser=str(inst.get("log_parser", "")),
                fail_to_pass=_parse_list(inst.get("FAIL_TO_PASS", [])),
                pass_to_pass=_parse_list(inst.get("PASS_TO_PASS", [])),
                difficulty=str(inst.get("difficulty", "")),
                language=str(inst.get("language", "")),
                raw_instance=inst,
            )
        )
    return tasks
