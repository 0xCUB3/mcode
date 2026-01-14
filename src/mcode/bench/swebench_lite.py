from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SWEbenchLiteTask:
    benchmark: str
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    raw_instance: dict


def load_swebench_lite(
    cache_dir: Path,
    *,
    split: str = "test",
    limit: int | None = None,
    instance_ids: list[str] | None = None,
) -> list[SWEbenchLiteTask]:
    """
    Load SWE-bench Lite instances via the official `swebench` package.

    Note: SWE-bench uses Hugging Face datasets internally and manages its own caching.
    The `cache_dir` parameter is accepted for API symmetry but is not currently used.
    """
    _ = cache_dir
    try:
        from swebench.harness.utils import load_swebench_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SWE-bench Lite requires the `swebench` extra. "
            "Install with `uv pip install -e '.[swebench]'`.\n"
            "If you installed `mcode` via `uv tool install ...`, install the extra there too:\n"
            "  `uv tool install -e '.[swebench]'`"
        ) from e

    instances: list[dict] = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split, instance_ids)
    if limit is not None:
        instances = instances[:limit]

    tasks: list[SWEbenchLiteTask] = []
    for inst in instances:
        tasks.append(
            SWEbenchLiteTask(
                benchmark="swebench-lite",
                instance_id=str(inst["instance_id"]),
                repo=str(inst["repo"]),
                base_commit=str(inst["base_commit"]),
                problem_statement=str(inst["problem_statement"]),
                hints_text=str(inst.get("hints_text", "")),
                version=str(inst.get("version", "")),
                raw_instance=inst,
            )
        )
    return tasks
