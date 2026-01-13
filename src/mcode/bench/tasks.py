from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Task:
    benchmark: str
    task_id: str
    prompt: str
    entry_point: Optional[str]
    test_code: str
    metadata: dict


def load_benchmark(benchmark: str, cache_dir: Path, limit: Optional[int] = None) -> list[Task]:
    name = benchmark.lower().strip()
    if name in {"humaneval", "human-eval"}:
        from mcode.bench.humaneval import load_humaneval

        return list(_limit(load_humaneval(cache_dir), limit))
    if name in {"mbpp"}:
        from mcode.bench.mbpp import load_mbpp

        return list(_limit(load_mbpp(cache_dir), limit))
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _limit(tasks: Iterable[Task], limit: Optional[int]) -> Iterable[Task]:
    if limit is None:
        yield from tasks
        return
    for i, task in enumerate(tasks):
        if i >= limit:
            return
        yield task

