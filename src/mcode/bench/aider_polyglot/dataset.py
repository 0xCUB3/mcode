from __future__ import annotations

import os
from pathlib import Path

from .constants import BENCHMARK_REPO, LANGUAGE_ORDER
from .models import AiderPolyglotTask


def default_benchmark_root() -> Path:
    override = os.environ.get("MCODE_AIDER_POLYGLOT_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "polyglot-benchmark"


def ensure_benchmark_root(path: str | Path | None = None) -> Path:
    root = Path(path).expanduser() if path is not None else default_benchmark_root()
    if root.is_dir():
        return root
    raise RuntimeError(
        f"Aider Polyglot benchmark not found at {root}. Clone it with: "
        f"git clone {BENCHMARK_REPO} {root}"
    )


def supported_languages() -> tuple[str, ...]:
    return LANGUAGE_ORDER


def load_aider_polyglot(
    root: str | Path | None = None,
    *,
    language: str = "all",
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[AiderPolyglotTask]:
    from .prepare import build_language_descriptors

    benchmark_root = ensure_benchmark_root(root)
    descriptors = build_language_descriptors(benchmark_root)

    if language != "all" and language not in descriptors:
        known = ", ".join(LANGUAGE_ORDER)
        raise ValueError(f"unknown language {language!r}. Expected one of: {known}, all")

    task_filter = set(task_ids or [])
    selected_languages = LANGUAGE_ORDER if language == "all" else (language,)
    tasks: list[AiderPolyglotTask] = []
    for language_name in selected_languages:
        descriptor = descriptors[language_name]
        if not descriptor.practice_dir.is_dir():
            if language == "all":
                continue
            raise RuntimeError(f"practice dir not found: {descriptor.practice_dir}")
        for exercise_dir in sorted(p for p in descriptor.practice_dir.iterdir() if p.is_dir()):
            task = AiderPolyglotTask(
                benchmark="aider-polyglot",
                task_id=f"{language_name}/{exercise_dir.name}",
                language=language_name,
                exercise=exercise_dir.name,
                source_dir=exercise_dir,
            )
            if task_filter and task.task_id not in task_filter:
                continue
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks
