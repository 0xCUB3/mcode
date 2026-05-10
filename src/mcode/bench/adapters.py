from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkAdapter:
    benchmark: str
    load_tasks: Callable[[int | None, list[str] | None], list[object]]
    task_id: Callable[[object], str]
    dataset_metadata: Callable[[], dict[str, object]]
    prepare_environment: Callable[[list[object]], object | None]
    run_task: Callable[[object, object | None, int], dict[str, object] | None]
    cleanup_task: Callable[[object, object | None], None]


def noop_cleanup(_task: object, _environment: object | None) -> None:
    return None


def adapter_for(
    benchmark: str,
    *,
    config: Any,
    run_swebench_task: Callable[[object, object | None, int], dict[str, object] | None],
    run_swebench_live_task: Callable[[object, object | None, int], dict[str, object] | None],
    run_aider_polyglot_task: Callable[[object, object | None, int], dict[str, object] | None],
) -> BenchmarkAdapter:
    name = benchmark.lower().strip()
    if name in {"swebench-lite", "swebench_lite"}:
        return _swebench_lite_adapter(config, run_swebench_task=run_swebench_task)
    if name in {"swebench-live", "swebench_live"}:
        return _swebench_live_adapter(config, run_swebench_live_task=run_swebench_live_task)
    if name in {"aider-polyglot", "aider_polyglot"}:
        return _aider_polyglot_adapter(config, run_aider_polyglot_task=run_aider_polyglot_task)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _swebench_lite_adapter(
    config: Any,
    *,
    run_swebench_task: Callable[[object, object | None, int], dict[str, object] | None],
) -> BenchmarkAdapter:
    from mcode.bench.swebench_lite import load_swebench_lite
    from mcode.execution.swebench import SWEbenchSandbox

    def load_tasks(limit: int | None, task_ids: list[str] | None) -> list[object]:
        return load_swebench_lite(
            config.cache_dir,
            split=config.swebench_split,
            limit=limit,
            instance_ids=task_ids,
            dataset_name=config.swebench_dataset,
        )

    def dataset_metadata() -> dict[str, object]:
        return {
            "name": config.swebench_dataset.split("/")[-1],
            "hf_dataset": config.swebench_dataset,
            "split": config.swebench_split,
        }

    def prepare_environment(tasks: list[object]) -> SWEbenchSandbox:
        sandbox = SWEbenchSandbox(
            namespace=config.swebench_namespace,
            arch=config.swebench_arch,
            max_workers=config.swebench_max_workers,
            mem_limit=config.swebench_mem_limit,
            pids_limit=config.swebench_pids_limit,
            cpu_limit=config.swebench_cpu_limit,
            force_rebuild=config.swebench_force_rebuild,
            check_image_digests=config.swebench_check_image_digests,
        )
        sandbox.prepare_images([task.raw_instance for task in tasks])
        return sandbox

    return BenchmarkAdapter(
        benchmark="swebench-lite",
        load_tasks=load_tasks,
        task_id=lambda task: str(getattr(task, "instance_id")),
        dataset_metadata=dataset_metadata,
        prepare_environment=prepare_environment,
        run_task=run_swebench_task,
        cleanup_task=noop_cleanup,
    )


def _swebench_live_adapter(
    config: Any,
    *,
    run_swebench_live_task: Callable[[object, object | None, int], dict[str, object] | None],
) -> BenchmarkAdapter:
    from mcode.bench.swebench_live import load_swebench_live
    from mcode.execution.swebench_live import SWEbenchLiveSandbox

    def load_tasks(limit: int | None, task_ids: list[str] | None) -> list[object]:
        return load_swebench_live(
            config.cache_dir,
            split=config.swebench_split,
            limit=limit,
            instance_ids=task_ids,
        )

    def prepare_environment(tasks: list[object]) -> SWEbenchLiveSandbox:
        sandbox = SWEbenchLiveSandbox(
            mem_limit=config.swebench_mem_limit,
            pids_limit=config.swebench_pids_limit,
            cpu_limit=config.swebench_cpu_limit,
            check_image_digests=config.swebench_check_image_digests,
        )
        sandbox.prepare_images(tasks)
        return sandbox

    def cleanup_task(task: object, environment: object | None) -> None:
        if os.environ.get("MCODE_KEEP_IMAGES") or environment is None:
            return
        environment.remove_image(task)

    return BenchmarkAdapter(
        benchmark="swebench-live",
        load_tasks=load_tasks,
        task_id=lambda task: str(getattr(task, "instance_id")),
        dataset_metadata=lambda: {
            "name": "SWE-bench-Live",
            "hf_dataset": "SWE-bench-Live/SWE-bench-Live",
            "split": config.swebench_split,
        },
        prepare_environment=prepare_environment,
        run_task=run_swebench_live_task,
        cleanup_task=cleanup_task,
    )


def _aider_polyglot_adapter(
    config: Any,
    *,
    run_aider_polyglot_task: Callable[[object, object | None, int], dict[str, object] | None],
) -> BenchmarkAdapter:
    from mcode.bench.aider_polyglot import load_aider_polyglot
    from mcode.bench.toolchains import ensure_polyglot_toolchains

    def prepare_polyglot_environment(tasks: list[object]) -> None:
        languages = sorted({str(getattr(task, "language")) for task in tasks})
        ensure_polyglot_toolchains(languages)

    return BenchmarkAdapter(
        benchmark="aider-polyglot",
        load_tasks=lambda limit, task_ids: load_aider_polyglot(
            config.aider_polyglot_root,
            language=config.aider_polyglot_language,
            limit=limit,
            task_ids=task_ids,
        ),
        task_id=lambda task: str(getattr(task, "task_id")),
        dataset_metadata=lambda: {
            "name": "Aider Polyglot",
            "root": str(config.aider_polyglot_root) if config.aider_polyglot_root else None,
            "language": config.aider_polyglot_language,
            "retry": config.aider_polyglot_retry,
            "retry_loop_budget": config.aider_polyglot_retry_loop_budget,
        },
        prepare_environment=prepare_polyglot_environment,
        run_task=run_aider_polyglot_task,
        cleanup_task=noop_cleanup,
    )
