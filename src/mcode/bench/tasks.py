from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    benchmark: str
    task_id: str
    prompt: str
    entry_point: str | None
    test_code: str
    metadata: dict


def load_benchmark(
    benchmark: str, cache_dir: Path, limit: int | None = None, **kwargs
) -> list[Task]:
    name = benchmark.lower().strip()
    if name in {"humaneval", "human-eval"}:
        from mcode.bench.humaneval import load_humaneval

        return list(_limit(load_humaneval(cache_dir), limit))
    if name == "mbpp":
        from mcode.bench.mbpp import load_mbpp

        return list(_limit(load_mbpp(cache_dir), limit))
    if name == "humaneval+":
        from mcode.bench.evalplus import load_humaneval_plus

        return list(_limit(load_humaneval_plus(cache_dir), limit))
    if name == "mbpp+":
        from mcode.bench.evalplus import load_mbpp_plus

        return list(_limit(load_mbpp_plus(cache_dir), limit))
    if name == "livecodebench":
        from mcode.bench.livecodebench import load_livecodebench

        return list(_limit(load_livecodebench(cache_dir, cutoff=kwargs.get("cutoff")), limit))
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _limit(tasks: Iterable[Task], limit: int | None) -> Iterable[Task]:
    if limit is None:
        yield from tasks
        return
    for i, task in enumerate(tasks):
        if i >= limit:
            return
        yield task
