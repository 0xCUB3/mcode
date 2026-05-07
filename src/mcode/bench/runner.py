from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

from mcode.bench.artifacts import (
    TaskArtifactManifest,
    TaskArtifactStore,
    VerificationEvidence,
    iso_utc_now,
    make_task_digest,
    read_task_manifest,
)
from mcode.bench.results import ResultsDB, RunSummary
from mcode.execution.sandbox import DockerUnavailableError
from mcode.llm.session import LLMSession
from mcode.ui.task_reporter import choose as choose_task_reporter
from mcode.util.retry import with_backoff


def _default_cache_dir() -> Path:
    override = os.environ.get("MCODE_CACHE_DIR")
    if override:
        return Path(override)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "mcode"
    return Path.home() / ".cache" / "mcode"


@dataclass(frozen=True)
class BenchConfig:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 15
    temperature: float | None = None
    seed: int | None = None
    timeout_s: int = 60
    task_shard_count: int | None = None
    task_shard_index: int | None = None
    cache_dir: Path = field(default_factory=_default_cache_dir)
    phase: str = "run"
    artifact_dir: Path | None = None
    artifact_candidate_index: int | None = None
    swebench_split: str = "test"
    swebench_namespace: str | None = "swebench"
    swebench_arch: str | None = None
    swebench_max_workers: int = 4
    swebench_force_rebuild: bool = False
    swebench_mem_limit: str = "4g"
    swebench_pids_limit: int = 512
    swebench_cpu_limit: float | None = None
    swebench_check_image_digests: bool = False
    swebench_eval_repair_attempts: int = 0
    swebench_dataset: str = "SWE-bench/SWE-bench_Lite"
    aider_polyglot_root: Path | None = None
    aider_polyglot_language: str = "all"
    aider_polyglot_retry: bool = True
    aider_polyglot_retry_loop_budget: int = 8
    n_samples: int = 1
    sampling_strategy: str = "none"
    sampling_budget: int | None = None
    selection_attempts: int = 1
    diagnostic_traces: bool = False
    live_trace: bool = False
    suite_name: str | None = None
    suite_entry_name: str | None = None


@dataclass(frozen=True)
class _RunResume:
    run_id: int
    existing_rows: dict[str, dict[str, object]]
    existing_artifacts: dict[str, dict[str, object]]
    retry_task_ids: set[str]


@dataclass(frozen=True)
class PatchRepoContext:
    repo_root: Path | str
    command_fn: Callable[[str], str] | None = None
    visible_repo_root: str | None = None
    test_cmds: object | None = None


def _coerce_patch_repo_context(repo_context: object) -> PatchRepoContext:
    repo_root = getattr(repo_context, "repo_root", repo_context)
    command_fn = getattr(repo_context, "command_fn", None)
    visible_repo_root = getattr(repo_context, "visible_repo_root", None)
    test_cmds = getattr(repo_context, "test_cmds", None)
    return PatchRepoContext(
        repo_root=repo_root,
        command_fn=command_fn,
        visible_repo_root=visible_repo_root,
        test_cmds=test_cmds,
    )


@dataclass(frozen=True)
class _BenchmarkAdapter:
    benchmark: str
    load_tasks: Callable[[int | None, list[str] | None], list[object]]
    task_id: Callable[[object], str]
    dataset_metadata: Callable[[], dict[str, object]]
    prepare_environment: Callable[[list[object]], object | None]
    run_task: Callable[[object, object | None, int], dict[str, object] | None]
    cleanup_task: Callable[[object, object | None], None]


def _noop_cleanup(_task: object, _environment: object | None) -> None:
    return None


class BenchmarkRunner:
    def __init__(
        self,
        *,
        config: BenchConfig,
        results_db: ResultsDB,
        json_mode: bool = False,
    ) -> None:
        self.config = config
        self.results_db = results_db
        self.json_mode = json_mode
        self._active_reporter = None
        self._live_trace_attempt_label: str | None = None
        self.llm = self._build_llm(loop_budget=config.loop_budget)

    def _build_llm(self, *, loop_budget: int) -> LLMSession:
        return LLMSession(
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            loop_budget=loop_budget,
            temperature=self.config.temperature,
            seed=self.config.seed,
            sampling_strategy=self.config.sampling_strategy,
            sampling_budget=self.config.sampling_budget,
            selection_attempts=self.config.selection_attempts,
            diagnostic_traces=self.config.diagnostic_traces,
            live_event_sink=self._live_trace_event if _live_trace_enabled(self.config) else None,
        )

    def _live_trace_event(self, event_type: str, event: object) -> None:
        if self._active_reporter is None:
            return
        text = _format_live_trace_event(event_type, event)
        if text:
            if self._live_trace_attempt_label and text.startswith("turn "):
                text = f"{self._live_trace_attempt_label}: {text}"
            self._active_reporter.event("info", text)

    def _start_or_resume_run(self, benchmark: str, config: dict) -> _RunResume:
        run_id = self.results_db.find_latest_run_by_config(benchmark, config)
        if run_id is None:
            run_id = self.results_db.start_run(benchmark, config)
            return _RunResume(
                run_id=run_id,
                existing_rows={},
                existing_artifacts={},
                retry_task_ids=set(),
            )
        existing_rows = self.results_db.task_terminal_rows(run_id)
        existing_artifacts = self.results_db.task_artifact_rows(run_id)
        retry_task_ids = {
            task_id for task_id, row in existing_rows.items() if _is_retryable_task_row(row)
        }
        return _RunResume(
            run_id=run_id,
            existing_rows=existing_rows,
            existing_artifacts=existing_artifacts,
            retry_task_ids=retry_task_ids,
        )

    def _should_run_task(self, resume: _RunResume, task_id: str) -> bool:
        if self.config.phase == "generate":
            return task_id not in resume.existing_artifacts
        if self.config.phase == "evaluate":
            return task_id not in resume.existing_rows or task_id in resume.retry_task_ids
        return task_id not in resume.existing_rows or task_id in resume.retry_task_ids

    def _save_task_result(self, run_id: int, result: dict[str, object]) -> None:
        with_backoff(
            lambda: self.results_db.save_task_result(run_id, result),
            is_retryable=_is_retryable_sqlite_lock,
            max_attempts=5,
            base_sleep_s=0.05,
            max_sleep_s=0.5,
        )

    def _save_task_artifact_manifest(
        self,
        run_id: int,
        manifest: TaskArtifactManifest,
        *,
        manifest_path: Path,
    ) -> None:
        with_backoff(
            lambda: self.results_db.save_task_artifact_manifest(
                run_id,
                manifest,
                manifest_path=manifest_path,
            ),
            is_retryable=_is_retryable_sqlite_lock,
            max_attempts=5,
            base_sleep_s=0.05,
            max_sleep_s=0.5,
        )
        if self.config.phase == "generate":
            with_backoff(
                lambda: self.results_db.delete_task_result(run_id, manifest.task.task_id),
                is_retryable=_is_retryable_sqlite_lock,
                max_attempts=5,
                base_sleep_s=0.05,
                max_sleep_s=0.5,
            )

    def _artifact_root(self) -> Path:
        artifact_dir = self.config.artifact_dir
        if artifact_dir is None:
            artifact_dir = self.results_db.path.parent / f"{self.results_db.path.stem}-artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _task_artifact_store(self, *, benchmark: str, task_id: str) -> TaskArtifactStore:
        return TaskArtifactStore.from_task(
            artifact_dir=self._artifact_root(),
            benchmark=benchmark,
            task_id=task_id,
        )

    def _task_metadata(self, task: object) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for name in (
            "repo",
            "base_commit",
            "problem_statement",
            "hints_text",
            "version",
            "language",
            "exercise",
        ):
            value = getattr(task, name, None)
            if value not in (None, ""):
                metadata[name] = value
        raw_instance = getattr(task, "raw_instance", None)
        if raw_instance is not None:
            metadata["raw_instance"] = raw_instance
        test_cmds = getattr(task, "test_cmds", None)
        if test_cmds is not None and "test_cmds" not in metadata:
            metadata["test_cmds"] = test_cmds
        return metadata

    def _task_repo_id(self, *, benchmark: str, task: object, task_id: str) -> str:
        repo = getattr(task, "repo", None)
        if isinstance(repo, str) and repo:
            return repo
        return f"{benchmark}/{task_id}"

    def _run_config_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self.config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _artifact_failure_counters(self, metrics: dict[str, object] | None) -> dict[str, int]:
        counters: dict[str, int] = {}
        if metrics is None:
            return counters
        for key in (
            "malformed_tool_call_recoveries",
            "invalid_tool_call_count",
            "blocked_finalizer_count",
            "repeated_failed_run_test_count",
            "post_edit_exploration_count",
        ):
            value = metrics.get(key)
            if isinstance(value, bool):
                counters[key] = int(value)
            elif isinstance(value, int):
                counters[key] = value
        return counters

    def _verification_evidence(self, items: object) -> list[VerificationEvidence]:
        if not isinstance(items, list):
            return []
        evidence_list: list[VerificationEvidence] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            evidence_list.append(
                VerificationEvidence(
                    verifier_name=str(item.get("verifier_name", "run_tests")),
                    command_label=str(item.get("command_label", "default")),
                    command_digest=str(item.get("command_digest", "")),
                    status=str(item.get("status", "UNKNOWN")),
                    counted_as_verification=bool(item.get("counted_as_verification", False)),
                    output_digest=str(item.get("output_digest", "")),
                    output_preview_path=None,
                    execution_time_ms=_coerce_optional_int(item.get("execution_time_ms")),
                    started_at=_coerce_optional_str(item.get("started_at")),
                    ended_at=_coerce_optional_str(item.get("ended_at")),
                    timed_out=bool(item.get("timed_out", False)),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "verifier_name",
                            "command_label",
                            "command_digest",
                            "status",
                            "counted_as_verification",
                            "output_digest",
                            "execution_time_ms",
                            "started_at",
                            "ended_at",
                            "timed_out",
                        }
                    },
                )
            )
        return evidence_list

    def _candidate_metrics_from_manifest(self, manifest: TaskArtifactManifest) -> dict[str, object]:
        if not manifest.candidates:
            return _scaffold_metrics(None)
        candidate = next(
            (item for item in manifest.candidates if item.selected),
            manifest.candidates[-1],
        )
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        return {
            "terminal_reason": candidate.terminal_reason,
            "turns_to_first_edit": _coerce_optional_int(metadata.get("turns_to_first_edit")),
            "turns_to_first_verification": _coerce_optional_int(
                metadata.get("turns_to_first_verification")
            ),
            "zero_edit": candidate.zero_edit,
            "zero_verification": candidate.zero_verification,
            "verification_succeeded": candidate.verification_succeeded,
            "prompt_snapshot": None,
            "prompt_tokens": candidate.prompt_tokens,
            "completion_tokens": candidate.completion_tokens,
            "total_tokens": candidate.total_tokens,
            "provider": candidate.provider,
            "response_model": candidate.response_model,
            "submission_json": candidate.submission_json,
            "validation_passed_count": candidate.validation_passed_count,
            "validation_failed_count": candidate.validation_failed_count,
            **candidate.failure_counters,
        }

    def _write_generation_manifest(
        self,
        *,
        run_id: int,
        benchmark: str,
        task_id: str,
        task: object,
        patch: str,
        metrics: dict[str, object] | None,
    ) -> TaskArtifactManifest:
        store = self._task_artifact_store(benchmark=benchmark, task_id=task_id)
        metadata = self._task_metadata(task)
        repo_id = self._task_repo_id(benchmark=benchmark, task=task, task_id=task_id)
        task_ref = store.build_task_ref(
            repo_id=repo_id,
            task_digest=make_task_digest(
                benchmark=benchmark,
                task_id=task_id,
                repo_id=repo_id,
                metadata=metadata,
            ),
            metadata=metadata,
        )
        normalized_metrics = _scaffold_metrics(metrics)
        candidate = store.write_candidate(
            candidate_index=0,
            patch=patch,
            terminal_reason=_coerce_optional_str(normalized_metrics.get("terminal_reason")),
            selected=True,
            submission_json=_coerce_optional_str(normalized_metrics.get("submission_json")),
            generation_time_ms=_coerce_optional_int(
                normalized_metrics.get("generation_latency_ms")
            ),
            prompt_tokens=_coerce_optional_int(normalized_metrics.get("prompt_tokens")),
            completion_tokens=_coerce_optional_int(normalized_metrics.get("completion_tokens")),
            total_tokens=_coerce_optional_int(normalized_metrics.get("total_tokens")),
            provider=_coerce_optional_str(normalized_metrics.get("provider")),
            response_model=_coerce_optional_str(normalized_metrics.get("response_model")),
            validation_passed_count=_coerce_optional_int(
                normalized_metrics.get("validation_passed_count")
            ),
            validation_failed_count=_coerce_optional_int(
                normalized_metrics.get("validation_failed_count")
            ),
            zero_edit=bool(normalized_metrics.get("zero_edit", True)),
            zero_verification=bool(normalized_metrics.get("zero_verification", True)),
            verification_succeeded=bool(normalized_metrics.get("verification_succeeded", False)),
            trace_events=(
                normalized_metrics.get("diagnostic_events")
                if isinstance(normalized_metrics.get("diagnostic_events"), list)
                else None
            ),
            verification_evidence=self._verification_evidence(
                normalized_metrics.get("verification_evidence")
            ),
            failure_counters=self._artifact_failure_counters(normalized_metrics),
            metadata={
                "phase": self.config.phase,
                "turns_to_first_edit": normalized_metrics.get("turns_to_first_edit"),
                "turns_to_first_verification": normalized_metrics.get(
                    "turns_to_first_verification"
                ),
                "last_model_output": normalized_metrics.get("last_model_output"),
            },
        )
        manifest = TaskArtifactManifest(
            schema_version=task_ref.artifact_version,
            phase=self.config.phase,
            generated_at=iso_utc_now(),
            run_config_digest=self._run_config_digest(),
            code_sha=_runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=task_ref,
            candidates=(candidate,),
            evaluations=(),
            metadata={"phase": self.config.phase},
        )
        manifest_path = store.write_manifest(manifest)
        self._save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)
        return manifest

    def _load_task_manifest(
        self, *, benchmark: str, task_id: str
    ) -> tuple[TaskArtifactStore, TaskArtifactManifest]:
        store = self._task_artifact_store(benchmark=benchmark, task_id=task_id)
        if not store.manifest_path.exists():
            raise FileNotFoundError(
                f"artifact manifest not found for {benchmark}:{task_id}: {store.manifest_path}"
            )
        return store, read_task_manifest(store.manifest_path)

    def _selected_candidate_patch(
        self, manifest: TaskArtifactManifest, store: TaskArtifactStore
    ) -> tuple[int, str]:
        if not manifest.candidates:
            raise RuntimeError(f"artifact manifest has no candidates for {manifest.task.task_id}")
        if self.config.artifact_candidate_index is not None:
            candidate = next(
                (
                    item
                    for item in manifest.candidates
                    if item.candidate_index == self.config.artifact_candidate_index
                ),
                None,
            )
            if candidate is None:
                raise RuntimeError(
                    "artifact manifest has no candidate index "
                    f"{self.config.artifact_candidate_index} for {manifest.task.task_id}"
                )
            return candidate.candidate_index, (store.task_root / candidate.patch_path).read_text(
                encoding="utf-8"
            )
        for candidate in manifest.candidates:
            if candidate.selected:
                return candidate.candidate_index, (
                    store.task_root / candidate.patch_path
                ).read_text(encoding="utf-8")
        candidate = manifest.candidates[-1]
        return candidate.candidate_index, (store.task_root / candidate.patch_path).read_text(
            encoding="utf-8"
        )

    def _append_generation_candidate(
        self,
        *,
        run_id: int,
        store: TaskArtifactStore,
        manifest: TaskArtifactManifest,
        candidate_index: int,
        patch: str,
        metrics: dict[str, object] | None,
        metadata: dict[str, object] | None = None,
    ) -> TaskArtifactManifest:
        normalized_metrics = _scaffold_metrics(metrics)
        candidate = store.write_candidate(
            candidate_index=candidate_index,
            patch=patch,
            terminal_reason=_coerce_optional_str(normalized_metrics.get("terminal_reason")),
            selected=True,
            submission_json=_coerce_optional_str(normalized_metrics.get("submission_json")),
            generation_time_ms=_coerce_optional_int(
                normalized_metrics.get("generation_latency_ms")
            ),
            prompt_tokens=_coerce_optional_int(normalized_metrics.get("prompt_tokens")),
            completion_tokens=_coerce_optional_int(normalized_metrics.get("completion_tokens")),
            total_tokens=_coerce_optional_int(normalized_metrics.get("total_tokens")),
            provider=_coerce_optional_str(normalized_metrics.get("provider")),
            response_model=_coerce_optional_str(normalized_metrics.get("response_model")),
            validation_passed_count=_coerce_optional_int(
                normalized_metrics.get("validation_passed_count")
            ),
            validation_failed_count=_coerce_optional_int(
                normalized_metrics.get("validation_failed_count")
            ),
            zero_edit=bool(normalized_metrics.get("zero_edit", True)),
            zero_verification=bool(normalized_metrics.get("zero_verification", True)),
            verification_succeeded=bool(normalized_metrics.get("verification_succeeded", False)),
            trace_events=(
                normalized_metrics.get("diagnostic_events")
                if isinstance(normalized_metrics.get("diagnostic_events"), list)
                else None
            ),
            verification_evidence=self._verification_evidence(
                normalized_metrics.get("verification_evidence")
            ),
            failure_counters=self._artifact_failure_counters(normalized_metrics),
            metadata={
                "phase": self.config.phase,
                "turns_to_first_edit": normalized_metrics.get("turns_to_first_edit"),
                "turns_to_first_verification": normalized_metrics.get(
                    "turns_to_first_verification"
                ),
                "last_model_output": normalized_metrics.get("last_model_output"),
                **dict(metadata or {}),
            },
        )
        updated = TaskArtifactManifest(
            schema_version=manifest.schema_version,
            phase=self.config.phase,
            generated_at=iso_utc_now(),
            run_config_digest=self._run_config_digest(),
            code_sha=_runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=manifest.task,
            candidates=tuple(
                [replace(item, selected=False) for item in manifest.candidates] + [candidate]
            ),
            evaluations=manifest.evaluations,
            metadata=dict(manifest.metadata),
        )
        manifest_path = store.write_manifest(updated)
        self._save_task_artifact_manifest(run_id, updated, manifest_path=manifest_path)
        return updated

    def _append_evaluation_manifest(
        self,
        *,
        run_id: int,
        store: TaskArtifactStore,
        manifest: TaskArtifactManifest,
        candidate_index: int,
        evaluation: _TaskEvaluation,
    ) -> TaskArtifactManifest:
        evaluation_artifact = store.write_evaluation(
            source_candidate_index=candidate_index,
            evaluator_name=evaluation.evaluator_name,
            passed=evaluation.passed,
            timed_out=evaluation.timed_out,
            exit_code=evaluation.exit_code,
            report=evaluation.report,
            stdout=evaluation.stdout,
            stderr=evaluation.stderr,
            error_class=evaluation.error,
            runtime_ms=evaluation.runtime_ms,
            metadata={"phase": self.config.phase},
        )
        updated = TaskArtifactManifest(
            schema_version=manifest.schema_version,
            phase=self.config.phase,
            generated_at=iso_utc_now(),
            run_config_digest=self._run_config_digest(),
            code_sha=_runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=manifest.task,
            candidates=manifest.candidates,
            evaluations=tuple([*manifest.evaluations, evaluation_artifact]),
            metadata=dict(manifest.metadata),
        )
        manifest_path = store.write_manifest(updated)
        self._save_task_artifact_manifest(run_id, updated, manifest_path=manifest_path)
        return updated

    def _record_preflight_infra_failure(
        self,
        *,
        resume: _RunResume,
        tasks: list,
        error: Exception,
    ) -> RunSummary:
        for task in tasks:
            task_id = _benchmark_task_id(task)
            if not self._should_run_task(resume, task_id):
                continue
            result = _task_error_result(
                task_id=task_id,
                start_time=time.time(),
                error=error,
                scaffold_metrics=None,
                attempts_used=0,
            )
            self._save_task_result(resume.run_id, result)
        return self.results_db.run_summary(resume.run_id)

    def run_benchmark(
        self,
        benchmark: str,
        *,
        limit: int | None = None,
        task_ids: list[str] | None = None,
    ) -> RunSummary:
        adapter = self._adapter_for(benchmark)
        if self.config.phase in {"run", "generate"}:
            self.llm.check_available()
        return self._run_adapter(adapter, limit=limit, task_ids=task_ids)

    def _adapter_for(self, benchmark: str) -> _BenchmarkAdapter:
        name = benchmark.lower().strip()
        if name in {"swebench-lite", "swebench_lite"}:
            from mcode.bench.swebench_lite import load_swebench_lite
            from mcode.execution.swebench import SWEbenchSandbox

            def load_tasks(limit: int | None, task_ids: list[str] | None) -> list[object]:
                return load_swebench_lite(
                    self.config.cache_dir,
                    split=self.config.swebench_split,
                    limit=limit,
                    instance_ids=task_ids,
                    dataset_name=self.config.swebench_dataset,
                )

            def dataset_metadata() -> dict[str, object]:
                return {
                    "name": self.config.swebench_dataset.split("/")[-1],
                    "hf_dataset": self.config.swebench_dataset,
                    "split": self.config.swebench_split,
                }

            def prepare_environment(tasks: list[object]) -> SWEbenchSandbox:
                sandbox = SWEbenchSandbox(
                    namespace=self.config.swebench_namespace,
                    arch=self.config.swebench_arch,
                    max_workers=self.config.swebench_max_workers,
                    mem_limit=self.config.swebench_mem_limit,
                    pids_limit=self.config.swebench_pids_limit,
                    cpu_limit=self.config.swebench_cpu_limit,
                    force_rebuild=self.config.swebench_force_rebuild,
                    check_image_digests=self.config.swebench_check_image_digests,
                )
                sandbox.prepare_images([task.raw_instance for task in tasks])
                return sandbox

            return _BenchmarkAdapter(
                benchmark="swebench-lite",
                load_tasks=load_tasks,
                task_id=lambda task: str(getattr(task, "instance_id")),
                dataset_metadata=dataset_metadata,
                prepare_environment=prepare_environment,
                run_task=lambda task, sandbox, run_id: self._run_swebench_task(
                    task,
                    swe_sandbox=sandbox,
                    run_id=run_id,
                ),
                cleanup_task=_noop_cleanup,
            )

        if name in {"swebench-live", "swebench_live"}:
            from mcode.bench.swebench_live import load_swebench_live
            from mcode.execution.swebench_live import SWEbenchLiveSandbox

            def load_tasks(limit: int | None, task_ids: list[str] | None) -> list[object]:
                return load_swebench_live(
                    self.config.cache_dir,
                    split=self.config.swebench_split,
                    limit=limit,
                    instance_ids=task_ids,
                )

            def prepare_environment(tasks: list[object]) -> SWEbenchLiveSandbox:
                sandbox = SWEbenchLiveSandbox(
                    mem_limit=self.config.swebench_mem_limit,
                    pids_limit=self.config.swebench_pids_limit,
                    cpu_limit=self.config.swebench_cpu_limit,
                    check_image_digests=self.config.swebench_check_image_digests,
                )
                sandbox.prepare_images(tasks)
                return sandbox

            def cleanup_task(task: object, environment: object | None) -> None:
                if os.environ.get("MCODE_KEEP_IMAGES") or environment is None:
                    return
                environment.remove_image(task)

            return _BenchmarkAdapter(
                benchmark="swebench-live",
                load_tasks=load_tasks,
                task_id=lambda task: str(getattr(task, "instance_id")),
                dataset_metadata=lambda: {
                    "name": "SWE-bench-Live",
                    "hf_dataset": "SWE-bench-Live/SWE-bench-Live",
                    "split": self.config.swebench_split,
                },
                prepare_environment=prepare_environment,
                run_task=lambda task, sandbox, run_id: self._run_swebench_live_task(
                    task,
                    live_sandbox=sandbox,
                    run_id=run_id,
                ),
                cleanup_task=cleanup_task,
            )

        if name in {"aider-polyglot", "aider_polyglot"}:
            from mcode.bench.aider_polyglot import load_aider_polyglot
            from mcode.bench.toolchains import ensure_polyglot_toolchains

            def prepare_polyglot_environment(tasks: list[object]) -> None:
                languages = sorted({str(getattr(task, "language")) for task in tasks})
                ensure_polyglot_toolchains(languages)

            return _BenchmarkAdapter(
                benchmark="aider-polyglot",
                load_tasks=lambda limit, task_ids: load_aider_polyglot(
                    self.config.aider_polyglot_root,
                    language=self.config.aider_polyglot_language,
                    limit=limit,
                    task_ids=task_ids,
                ),
                task_id=lambda task: str(getattr(task, "task_id")),
                dataset_metadata=lambda: {
                    "name": "Aider Polyglot",
                    "root": (
                        str(self.config.aider_polyglot_root)
                        if self.config.aider_polyglot_root
                        else None
                    ),
                    "language": self.config.aider_polyglot_language,
                    "retry": self.config.aider_polyglot_retry,
                    "retry_loop_budget": self.config.aider_polyglot_retry_loop_budget,
                },
                prepare_environment=prepare_polyglot_environment,
                run_task=lambda task, _environment, run_id: self._run_aider_polyglot_task(
                    task,
                    run_id=run_id,
                ),
                cleanup_task=_noop_cleanup,
            )

        raise ValueError(f"Unknown benchmark: {benchmark}")

    def _run_adapter(
        self,
        adapter: _BenchmarkAdapter,
        *,
        limit: int | None,
        task_ids: list[str] | None,
    ) -> RunSummary:
        tasks = adapter.load_tasks(limit, task_ids)
        tasks = _apply_task_shard(
            tasks,
            self.config.task_shard_count,
            self.config.task_shard_index,
        )
        config = _augment_run_config(asdict(self.config))
        config["planned_task_count"] = len(tasks)
        config["dataset"] = adapter.dataset_metadata()
        resume = self._start_or_resume_run(adapter.benchmark, config)
        pending_tasks = [
            task for task in tasks if self._should_run_task(resume, adapter.task_id(task))
        ]
        if not pending_tasks:
            return self.results_db.run_summary(resume.run_id)

        try:
            environment = with_backoff(
                lambda: adapter.prepare_environment(pending_tasks),
                is_retryable=_is_retryable_infra_error,
                max_attempts=2,
                base_sleep_s=0.1,
                max_sleep_s=0.5,
            )
            if self.config.phase == "prepare":
                return self.results_db.run_summary(resume.run_id)
        except Exception as exc:
            if _is_polyglot_toolchain_error(exc):
                raise
            if not _is_retryable_infra_error(exc):
                raise
            return self._record_preflight_infra_failure(
                resume=resume,
                tasks=pending_tasks,
                error=exc,
            )

        with choose_task_reporter(json_mode=self.json_mode) as reporter:
            self._active_reporter = reporter
            try:
                reporter.total(len(pending_tasks))
                for task in pending_tasks:
                    task_id = adapter.task_id(task)
                    reporter.event("info", f"starting {task_id}")
                    result = adapter.run_task(task, environment, resume.run_id)
                    if result is not None:
                        self._save_task_result(resume.run_id, result)
                        detail = f"{task_id} {'ok' if result['passed'] else 'fail'}"
                    else:
                        detail = f"{task_id} generated"
                    adapter.cleanup_task(task, environment)
                    reporter.advance(detail=detail)
            finally:
                self._active_reporter = None

        return self.results_db.run_summary(resume.run_id)

    def _run_aider_polyglot_task(self, task, *, run_id: int) -> dict[str, object] | None:
        from pathlib import Path

        from mcode.bench.aider_polyglot import (
            apply_patch_to_prepared_task,
            cleanup_prepared_task,
            prepare_task,
            reset_to_baseline,
            run_test_commands,
        )
        from mcode.llm.repo_state import get_git_diff, restore_repo_snapshot

        start = time.time()
        prepared = None
        first_metrics: dict[str, object] | None = None
        final_metrics: dict[str, object] | None = None
        first_pass_snapshot = None
        final_pass_snapshot = None
        attempts_used = 0
        try:
            prepared = prepare_task(task, benchmark_root=self.config.aider_polyglot_root)
            if not prepared.stub_paths or not prepared.test_paths:
                raise RuntimeError(
                    f"benchmark task {task.task_id} is missing stubs or tests after preparation"
                )

            if self.config.phase == "evaluate":
                store, manifest = self._load_task_manifest(
                    benchmark=task.benchmark,
                    task_id=task.task_id,
                )
                candidate_index, patch = self._selected_candidate_patch(manifest, store)
                scaffold_result = self._candidate_metrics_from_manifest(manifest)
                apply_outcome = apply_patch_to_prepared_task(prepared, patch)
                if apply_outcome.passed:
                    evaluation = run_test_commands(prepared)
                    error = None if evaluation.passed else "Tests failed"
                    stderr = None
                else:
                    evaluation = apply_outcome
                    error = "Patch did not apply"
                    stderr = None
                eval_detail = _TaskEvaluation(
                    evaluator_name="aider-polyglot",
                    passed=evaluation.passed and apply_outcome.passed,
                    timed_out=evaluation.timed_out,
                    stdout=_truncate(evaluation.output),
                    stderr=stderr,
                    error=error,
                    exit_code=evaluation.exit_code,
                    report={
                        "apply_patch_output": _truncate(apply_outcome.output),
                        "apply_patch_passed": apply_outcome.passed,
                        "tests_output": _truncate(evaluation.output),
                    },
                    runtime_ms=int((time.time() - start) * 1000),
                )
                if eval_detail.passed:
                    scaffold_result["terminal_reason"] = "submitted"
                elif scaffold_result.get("verification_succeeded"):
                    scaffold_result["terminal_reason"] = "wrong_patch_after_verification"
                _append_terminal_diagnostic(scaffold_result, eval_detail)
                self._append_evaluation_manifest(
                    run_id=run_id,
                    store=store,
                    manifest=manifest,
                    candidate_index=candidate_index,
                    evaluation=eval_detail,
                )
                return {
                    "task_id": task.task_id,
                    "passed": eval_detail.passed,
                    "attempts_used": 1,
                    "time_ms": int((time.time() - start) * 1000),
                    "exit_code": eval_detail.exit_code,
                    "timed_out": eval_detail.timed_out,
                    "stdout": eval_detail.stdout,
                    "stderr": eval_detail.stderr,
                    "error": eval_detail.error,
                    "code_sha256": hashlib.sha256(
                        patch.encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if patch
                    else None,
                    **scaffold_result,
                }

            first_metrics, first_pass_snapshot = self._run_aider_polyglot_attempt(
                task=task,
                prepared=prepared,
                prompt=prepared.build_first_prompt(),
                loop_budget=self.config.loop_budget,
                attempt_label="attempt 1/2" if self.config.aider_polyglot_retry else "attempt 1/1",
            )
            attempts_used = 1
            diff = get_git_diff(str(prepared.work_dir))
            if self.config.phase == "generate":
                self._write_generation_manifest(
                    run_id=run_id,
                    benchmark=task.benchmark,
                    task_id=task.task_id,
                    task=task,
                    patch=diff,
                    metrics=first_metrics,
                )
                return None

            evaluation = run_test_commands(prepared)
            if not evaluation.passed and first_pass_snapshot is not None:
                restore_repo_snapshot(
                    str(prepared.work_dir),
                    Path(first_pass_snapshot.name) / "snapshot",
                )
                evaluation = run_test_commands(prepared)

            repair_metrics = None
            if not evaluation.passed and self.config.aider_polyglot_retry:
                retry_output = evaluation.output
                if _should_reset_before_retry(retry_output):
                    reset_to_baseline(prepared.work_dir)
                final_metrics, final_pass_snapshot = self._run_aider_polyglot_attempt(
                    task=task,
                    prepared=prepared,
                    prompt=prepared.build_retry_prompt(retry_output),
                    loop_budget=self.config.aider_polyglot_retry_loop_budget,
                    attempt_label="attempt 2/3",
                )
                evaluation = run_test_commands(prepared)
                if not evaluation.passed and final_pass_snapshot is not None:
                    restore_repo_snapshot(
                        str(prepared.work_dir),
                        Path(final_pass_snapshot.name) / "snapshot",
                    )
                    evaluation = run_test_commands(prepared)
                attempts_used = 2
                if not evaluation.passed and _should_run_final_polyglot_repair(evaluation.output):
                    repair_metrics, repair_pass_snapshot = self._run_aider_polyglot_attempt(
                        task=task,
                        prepared=prepared,
                        prompt=prepared.build_retry_prompt(evaluation.output),
                        loop_budget=self.config.aider_polyglot_retry_loop_budget,
                        attempt_label="attempt 3/3",
                    )
                    evaluation = run_test_commands(prepared)
                    if not evaluation.passed and repair_pass_snapshot is not None:
                        restore_repo_snapshot(
                            str(prepared.work_dir),
                            Path(repair_pass_snapshot.name) / "snapshot",
                        )
                        evaluation = run_test_commands(prepared)
                    attempts_used = 3
            else:
                final_metrics = first_metrics

            terminal_reason = "submitted" if evaluation.passed else None
            metrics = _merge_polyglot_metrics(
                first=first_metrics,
                second=final_metrics if attempts_used > 1 else None,
                first_loop_budget=self.config.loop_budget,
                terminal_reason=terminal_reason,
            )
            if repair_metrics is not None:
                metrics = _merge_polyglot_metrics(
                    first=metrics,
                    second=repair_metrics,
                    first_loop_budget=(
                        self.config.loop_budget + self.config.aider_polyglot_retry_loop_budget
                    ),
                    terminal_reason=terminal_reason,
                )
            diff = get_git_diff(str(prepared.work_dir))
            manifest = self._write_generation_manifest(
                run_id=run_id,
                benchmark=task.benchmark,
                task_id=task.task_id,
                task=task,
                patch=diff,
                metrics=metrics,
            )
            eval_detail = _TaskEvaluation(
                evaluator_name="aider-polyglot",
                passed=evaluation.passed,
                timed_out=evaluation.timed_out,
                stdout=_truncate(evaluation.output),
                stderr=None,
                error=None if evaluation.passed else "Tests failed",
                exit_code=evaluation.exit_code,
                report={"tests_output": _truncate(evaluation.output)},
                runtime_ms=int((time.time() - start) * 1000),
            )
            self._append_evaluation_manifest(
                run_id=run_id,
                store=self._task_artifact_store(benchmark=task.benchmark, task_id=task.task_id),
                manifest=manifest,
                candidate_index=0,
                evaluation=eval_detail,
            )
            sha = (
                hashlib.sha256(diff.encode("utf-8", errors="ignore")).hexdigest() if diff else None
            )
            return {
                "task_id": task.task_id,
                "passed": evaluation.passed,
                "attempts_used": attempts_used,
                "time_ms": int((time.time() - start) * 1000),
                "exit_code": evaluation.exit_code,
                "timed_out": evaluation.timed_out,
                "stdout": _truncate(evaluation.output),
                "stderr": None,
                "error": None if evaluation.passed else "Tests failed",
                "code_sha256": sha,
                **metrics,
            }
        except Exception as e:
            return _task_error_result(
                task_id=task.task_id,
                start_time=start,
                error=e,
                scaffold_metrics=final_metrics or first_metrics,
                attempts_used=attempts_used,
            )
        finally:
            for snapshot in (first_pass_snapshot, final_pass_snapshot):
                if snapshot is not None:
                    snapshot.cleanup()
            if prepared is not None:
                cleanup_prepared_task(prepared)

    def _run_aider_polyglot_attempt(
        self,
        *,
        task,
        prepared,
        prompt: str,
        loop_budget: int,
        attempt_label: str,
    ) -> tuple[dict[str, object] | None, object | None]:
        import shutil

        from mcode.agent.tooling import format_tool_result
        from mcode.bench.aider_polyglot import (
            _failure_report_snippets,
            run_command_sequence,
            run_single_command,
        )
        from mcode.util import temporary_directory

        if self._active_reporter is not None:
            self._active_reporter.event("info", attempt_label)
        llm = self._build_llm(loop_budget=loop_budget)
        pass_snapshot = None
        allowed_commands = tuple(prepared.test_commands)

        def _capture_passing_state() -> None:
            nonlocal pass_snapshot
            if pass_snapshot is not None:
                return
            pass_snapshot = temporary_directory(prefix="mcode-polyglot-pass-")
            snapshot_dir = Path(pass_snapshot.name) / "snapshot"
            shutil.copytree(
                prepared.work_dir,
                snapshot_dir,
                ignore=shutil.ignore_patterns(".git"),
                symlinks=True,
            )

        allowed_commands = _allowed_polyglot_test_commands(prepared)

        def test_fn(test_cmd: str = "default") -> str:
            normalized = test_cmd.strip() or "default"
            if normalized.lower() == "default":
                outcome = run_command_sequence(
                    prepared.work_dir,
                    prepared.test_commands,
                    timeout_s=prepared.timeout_s,
                )
                label = "run_tests default"
            elif normalized in allowed_commands:
                outcome = run_single_command(
                    prepared.work_dir,
                    normalized,
                    timeout_s=prepared.timeout_s,
                )
                label = normalized
            else:
                allowed = ", ".join(allowed_commands)
                return format_tool_result(
                    normalized,
                    "BLOCKED",
                    (
                        "Only `run_tests default` or one of the declared benchmark "
                        f"commands is allowed here: {allowed}"
                    ),
                )
            if outcome.passed:
                _capture_passing_state()
            output = outcome.output
            if (
                not outcome.passed
                and "Failure report snippets:" not in output
                and (snippets := _failure_report_snippets(prepared.work_dir, output))
            ):
                output = f"{output}\n\nFailure report snippets:\n{snippets}"
            status = "PASSED" if outcome.passed else ("TIMEOUT" if outcome.timed_out else "FAILED")
            return format_tool_result(label, status, output)

        previous_attempt_label = self._live_trace_attempt_label
        self._live_trace_attempt_label = attempt_label
        try:
            llm.solve(
                repo=task.repo,
                problem_statement=prompt,
                repo_root=str(prepared.work_dir),
                n_samples=self.config.n_samples,
                test_cmds={"test_cmds": list(prepared.test_commands)},
                test_fn=test_fn,
                visible_repo_root=str(prepared.work_dir),
                editable_paths=prepared.stub_paths,
            )
        except Exception as exc:
            if pass_snapshot is None:
                raise
            metrics = _generation_result(llm) or {}
            metrics.update(
                {
                    "terminal_reason": "verified_after_model_error",
                    "verification_succeeded": True,
                    "zero_edit": False,
                    "zero_verification": False,
                    "last_model_output": {
                        "preview": f"Recovered passing patch after model error: {exc}",
                    },
                }
            )
            return metrics, pass_snapshot
        finally:
            self._live_trace_attempt_label = previous_attempt_label
        return _generation_result(llm), pass_snapshot

    def _generate_task_patch(
        self,
        task,
        *,
        repo_root: Path | str,
        command_fn: Callable[[str], str] | None = None,
        visible_repo_root: str | None = None,
        test_cmds: object | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        verification_metadata = test_cmds
        if verification_metadata is None:
            verification_metadata = getattr(task, "raw_instance", None)
        if verification_metadata is None:
            verification_metadata = getattr(task, "test_cmds", None)

        result = self.llm.solve(
            repo=task.repo,
            problem_statement=task.problem_statement,
            hints_text=task.hints_text or "",
            repo_root=str(repo_root),
            n_samples=self.config.n_samples,
            test_cmds=verification_metadata,
            command_fn=command_fn,
            visible_repo_root=visible_repo_root,
        )
        return result.patch, _generation_result(self.llm)

    def _run_swebench_live_task(self, task, *, live_sandbox, run_id: int) -> dict:
        return self._run_task(
            run_id=run_id,
            task_id=task.instance_id,
            generation_task=task,
            repo_context_factory=lambda: live_sandbox.repo_context(task),
            evaluate_patch=lambda patch: _evaluate_live_patch(
                live_sandbox=live_sandbox,
                task=task,
                patch=patch,
                run_id=run_id,
                timeout_s=self.config.timeout_s,
            ),
        )

    def _run_swebench_task(self, task, *, swe_sandbox, run_id: int) -> dict:
        return self._run_task(
            run_id=run_id,
            task_id=task.instance_id,
            generation_task=task,
            repo_context_factory=lambda: swe_sandbox.repo_context(task.raw_instance),
            evaluate_patch=lambda patch: _evaluate_lite_patch(
                swe_sandbox=swe_sandbox,
                task=task,
                patch=patch,
                run_id=run_id,
                timeout_s=self.config.timeout_s,
                model_id=self.config.model_id,
            ),
        )

    def _run_task(
        self,
        *,
        run_id: int = 0,
        task_id: str,
        generation_task,
        repo_context_factory: Callable[[], object],
        evaluate_patch: Callable[[str], _TaskEvaluation],
    ) -> dict[str, object] | None:
        benchmark = str(getattr(generation_task, "benchmark", ""))
        start = time.time()
        attempts_used = 0
        elapsed_ms = 0

        if self.config.phase == "evaluate":
            try:
                store, manifest = self._load_task_manifest(
                    benchmark=benchmark,
                    task_id=task_id,
                )
                scaffold_result = self._candidate_metrics_from_manifest(manifest)
                candidate_index, patch = self._selected_candidate_patch(manifest, store)
                if patch.strip():
                    eval_detail = evaluate_patch(patch)
                else:
                    eval_detail = _TaskEvaluation(
                        evaluator_name=benchmark or "artifact-eval",
                        passed=False,
                        timed_out=False,
                        stdout=None,
                        stderr=None,
                        error="No patch candidate found",
                    )
                if eval_detail.passed:
                    scaffold_result["terminal_reason"] = "submitted"
                elif scaffold_result.get("verification_succeeded"):
                    scaffold_result["terminal_reason"] = "wrong_patch_after_verification"
                _append_terminal_diagnostic(scaffold_result, eval_detail)
                self._append_evaluation_manifest(
                    run_id=run_id,
                    store=store,
                    manifest=manifest,
                    candidate_index=candidate_index,
                    evaluation=eval_detail,
                )
                elapsed_ms = int((time.time() - start) * 1000)
                return {
                    "task_id": task_id,
                    "passed": eval_detail.passed,
                    "attempts_used": 1,
                    "time_ms": elapsed_ms,
                    "code_sha256": hashlib.sha256(
                        patch.encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if patch
                    else None,
                    **eval_detail.as_result_dict(),
                    **scaffold_result,
                }
            except Exception as e:
                return _task_error_result(
                    task_id=task_id,
                    start_time=start,
                    error=e,
                    scaffold_metrics=None,
                    attempts_used=attempts_used,
                )

        patch = ""
        scaffold_metrics: dict[str, object] | None = None
        scaffold_result: dict[str, object] | None = None
        eval_detail: _TaskEvaluation | None = None
        manifest: TaskArtifactManifest | None = None
        for attempts_used in range(1, 3):
            try:
                with repo_context_factory() as repo_context:
                    patch_context = _coerce_patch_repo_context(repo_context)
                    if scaffold_result is None:
                        patch, scaffold_metrics = self._generate_task_patch(
                            generation_task,
                            repo_root=patch_context.repo_root,
                            command_fn=patch_context.command_fn,
                            visible_repo_root=patch_context.visible_repo_root,
                            test_cmds=patch_context.test_cmds,
                        )
                        elapsed_ms = int((time.time() - start) * 1000)
                        scaffold_result = _scaffold_metrics(scaffold_metrics)
                        manifest = self._write_generation_manifest(
                            run_id=run_id,
                            benchmark=benchmark,
                            task_id=task_id,
                            task=generation_task,
                            patch=patch,
                            metrics=scaffold_metrics,
                        )
                        if self.config.phase == "generate":
                            return None
                    candidate_index = 0
                    if patch.strip():
                        eval_detail = evaluate_patch(patch)
                        scaffold_result = _update_terminal_reason_from_eval(
                            scaffold_result,
                            eval_detail,
                        )
                        if manifest is not None:
                            store = self._task_artifact_store(
                                benchmark=benchmark,
                                task_id=task_id,
                            )
                            manifest = self._append_evaluation_manifest(
                                run_id=run_id,
                                store=store,
                                manifest=manifest,
                                candidate_index=candidate_index,
                                evaluation=eval_detail,
                            )
                        repair_attempt = 0
                        while _should_run_swebench_eval_repair(
                            benchmark=benchmark,
                            patch=patch,
                            eval_detail=eval_detail,
                            repair_attempt=repair_attempt,
                            max_repair_attempts=self.config.swebench_eval_repair_attempts,
                        ):
                            repair_attempt += 1
                            _apply_patch_to_repo(patch_context.repo_root, patch)
                            repair_task = _task_with_eval_repair_feedback(
                                generation_task,
                                eval_detail=eval_detail,
                                repair_attempt=repair_attempt,
                            )
                            patch, scaffold_metrics = self._generate_task_patch(
                                repair_task,
                                repo_root=patch_context.repo_root,
                                command_fn=patch_context.command_fn,
                                visible_repo_root=patch_context.visible_repo_root,
                                test_cmds=patch_context.test_cmds,
                            )
                            candidate_index = repair_attempt
                            elapsed_ms = int((time.time() - start) * 1000)
                            scaffold_result = _scaffold_metrics(scaffold_metrics)
                            if manifest is not None:
                                store = self._task_artifact_store(
                                    benchmark=benchmark,
                                    task_id=task_id,
                                )
                                manifest = self._append_generation_candidate(
                                    run_id=run_id,
                                    store=store,
                                    manifest=manifest,
                                    candidate_index=candidate_index,
                                    patch=patch,
                                    metrics=scaffold_metrics,
                                    metadata={"eval_repair_attempt": repair_attempt},
                                )
                            if not patch.strip():
                                break
                            eval_detail = evaluate_patch(patch)
                            scaffold_result = _update_terminal_reason_from_eval(
                                scaffold_result,
                                eval_detail,
                            )
                            if manifest is not None:
                                store = self._task_artifact_store(
                                    benchmark=benchmark,
                                    task_id=task_id,
                                )
                                manifest = self._append_evaluation_manifest(
                                    run_id=run_id,
                                    store=store,
                                    manifest=manifest,
                                    candidate_index=candidate_index,
                                    evaluation=eval_detail,
                                )
                break
            except Exception as e:
                if _is_retryable_infra_error(e) and attempts_used < 2:
                    continue
                return _task_error_result(
                    task_id=task_id,
                    start_time=start,
                    error=e,
                    scaffold_metrics=scaffold_metrics,
                    attempts_used=attempts_used,
                )

        if scaffold_result is not None:
            _append_terminal_diagnostic(scaffold_result, eval_detail)
        if manifest is not None and eval_detail is not None and not manifest.evaluations:
            store = self._task_artifact_store(benchmark=benchmark, task_id=task_id)
            self._append_evaluation_manifest(
                run_id=run_id,
                store=store,
                manifest=manifest,
                candidate_index=0,
                evaluation=eval_detail,
            )
        sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest() if patch else None
        return {
            "task_id": task_id,
            "passed": eval_detail.passed if eval_detail is not None else False,
            "attempts_used": attempts_used,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **(eval_detail.as_result_dict() if eval_detail is not None else {}),
            **(scaffold_result or _scaffold_metrics(scaffold_metrics)),
        }


@dataclass(frozen=True)
class _TaskEvaluation:
    passed: bool
    timed_out: bool
    stdout: str | None
    stderr: str | None
    error: str | None
    evaluator_name: str = ""
    exit_code: int | None = None
    report: dict[str, object] | None = None
    runtime_ms: int | None = None

    def as_result_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


def _update_terminal_reason_from_eval(
    scaffold_result: dict[str, object],
    eval_detail: _TaskEvaluation,
) -> dict[str, object]:
    if eval_detail.passed:
        scaffold_result["terminal_reason"] = "submitted"
    elif scaffold_result.get("verification_succeeded"):
        scaffold_result["terminal_reason"] = "wrong_patch_after_verification"
    return scaffold_result


def _should_run_swebench_eval_repair(
    *,
    benchmark: str,
    patch: str,
    eval_detail: _TaskEvaluation | None,
    repair_attempt: int,
    max_repair_attempts: int,
) -> bool:
    if benchmark not in {"swebench-lite", "swebench-live"}:
        return False
    if repair_attempt >= max_repair_attempts:
        return False
    if not patch.strip() or eval_detail is None or eval_detail.passed:
        return False
    if eval_detail.timed_out:
        return False
    return True


def _task_with_eval_repair_feedback(
    task: object,
    *,
    eval_detail: _TaskEvaluation,
    repair_attempt: int,
) -> object:
    problem = str(getattr(task, "problem_statement", ""))
    feedback = _format_eval_repair_feedback(eval_detail)
    return SimpleNamespace(
        repo=getattr(task, "repo", ""),
        problem_statement=(
            f"{problem}\n\n"
            f"Previous official SWE-bench evaluation failed after repair attempt "
            f"{repair_attempt - 1}. Continue from the current repository state and "
            "repair only what is needed. Use the deterministic feedback below, then run "
            "the available tests before submitting.\n\n"
            f"{feedback}"
        ),
        hints_text=getattr(task, "hints_text", ""),
        raw_instance=getattr(task, "raw_instance", None),
        benchmark=getattr(task, "benchmark", ""),
        instance_id=getattr(task, "instance_id", None),
    )


def _format_eval_repair_feedback(eval_detail: _TaskEvaluation) -> str:
    sections: list[str] = ["Official evaluation feedback:"]
    if eval_detail.error:
        sections.append(f"Error: {eval_detail.error}")
    if eval_detail.report:
        sections.append(
            "Report JSON:\n" + json.dumps(eval_detail.report, sort_keys=True, default=str)[:4000]
        )
    output = _clean_eval_output(eval_detail.stdout or "")
    if output:
        sections.append("Test output excerpt:\n" + _head_tail_excerpt(output, max_chars=9000))
    return "\n\n".join(sections)


def _apply_patch_to_repo(repo_root: Path | str, patch: str) -> bool:
    if not patch.strip():
        return True
    root = Path(repo_root)
    if not (root / ".git").exists():
        return False
    check = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--check", "-"],
        cwd=root,
        input=patch,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        return False
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=root,
        input=patch,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _evaluate_live_patch(
    *,
    live_sandbox,
    task,
    patch: str,
    run_id: int,
    timeout_s: int,
) -> _TaskEvaluation:
    run = live_sandbox.evaluate_patch(
        task=task,
        patch=patch,
        run_id=f"mcode-{run_id}",
        timeout_s=timeout_s,
    )
    return _TaskEvaluation(
        evaluator_name="swebench-live",
        passed=run.resolved,
        timed_out=run.timed_out,
        stdout=_truncate(run.test_output),
        stderr=json.dumps(run.report, sort_keys=True),
        error=None if run.resolved else _official_eval_error(run.test_output, run.report),
        report=run.report,
        runtime_ms=int(float(getattr(run, "runtime_s", 0.0) or 0.0) * 1000),
    )


def _evaluate_lite_patch(
    *,
    swe_sandbox,
    task,
    patch: str,
    run_id: int,
    timeout_s: int,
    model_id: str,
) -> _TaskEvaluation:
    run = swe_sandbox.evaluate_patch(
        instance=task.raw_instance,
        model_id=model_id,
        patch=patch,
        run_id=f"mcode-{run_id}",
        timeout_s=timeout_s,
    )
    inst_report = run.report.get(task.instance_id, {})
    return _TaskEvaluation(
        evaluator_name="swebench-lite",
        passed=run.resolved,
        timed_out=run.timed_out,
        stdout=_truncate(run.test_output),
        stderr=json.dumps(inst_report, sort_keys=True),
        error=None if run.resolved else _official_eval_error(run.test_output, inst_report),
        report=inst_report,
        runtime_ms=int(float(getattr(run, "runtime_s", 0.0) or 0.0) * 1000),
    )


def _official_eval_error(test_output: object, report: object) -> str:
    text = _clean_eval_output(str(test_output or ""))
    failed_lines = [line.strip() for line in text.splitlines() if _looks_like_failed_test(line)]
    if failed_lines:
        return "Not resolved: " + " | ".join(failed_lines[-3:])[:1200]
    if isinstance(report, Mapping) and report.get("patch_successfully_applied") is False:
        return "Patch did not apply"
    return "Not resolved"


def _looks_like_failed_test(line: str) -> bool:
    lowered = line.lower()
    if not line.strip():
        return False
    return (
        line.lstrip().startswith(("FAILED ", "ERROR "))
        or " failed" in lowered
        or " error" in lowered
        or lowered.startswith(("fail:", "error:"))
    )


def _benchmark_task_id(task: object) -> str:
    task_id = getattr(task, "instance_id", None)
    if task_id is None:
        task_id = getattr(task, "task_id")
    return str(task_id)


def _is_retryable_sqlite_lock(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "database is locked" in str(exc).lower()


def _is_retryable_task_row(row: dict[str, object]) -> bool:
    if row.get("terminal_reason") == "infra_failure":
        return True
    error = row.get("error")
    return isinstance(error, str) and _is_retryable_infra_text(error)


def _is_retryable_infra_error(exc: BaseException) -> bool:
    if isinstance(exc, DockerUnavailableError):
        return True
    return _is_retryable_infra_text(str(exc))


def _is_polyglot_toolchain_error(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "PolyglotToolchainError"


def _is_retryable_infra_text(text: str) -> bool:
    try:
        from mcode.execution.swebench import _is_retryable_podman_image_error

        if _is_retryable_podman_image_error(text):
            return True
    except Exception:
        pass
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in (
            "dockerunavailableerror",
            "docker unavailable",
            "podman socket",
            "docker socket",
            "database is locked",
            "polyglot toolchain unavailable",
            "xml syntax error",
        )
    )


def _append_terminal_diagnostic(
    scaffold_result: dict[str, object],
    eval_detail: _TaskEvaluation | None,
) -> None:
    events = scaffold_result.get("diagnostic_events")
    if not isinstance(events, list):
        return
    events.append(
        {
            "turn": None,
            "event_type": "terminal",
            "payload": {
                "terminal_reason": scaffold_result.get("terminal_reason"),
                "verification_succeeded": scaffold_result.get("verification_succeeded"),
                "turns_to_first_edit": scaffold_result.get("turns_to_first_edit"),
                "turns_to_first_verification": scaffold_result.get("turns_to_first_verification"),
                "official_eval_passed": eval_detail.passed if eval_detail is not None else None,
                "official_eval_timed_out": (
                    eval_detail.timed_out if eval_detail is not None else None
                ),
            },
        }
    )


def _task_error_result(
    *,
    task_id: str,
    start_time: float,
    error: Exception,
    scaffold_metrics: dict[str, object] | None,
    attempts_used: int = 0,
) -> dict[str, object]:
    tb = traceback.format_exc()
    return {
        "task_id": task_id,
        "passed": False,
        "attempts_used": attempts_used,
        "time_ms": int((time.time() - start_time) * 1000),
        "exit_code": None,
        "timed_out": False,
        "stdout": None,
        "stderr": _truncate(tb, max_chars=8000) if tb else None,
        "error": f"{type(error).__name__}: {error}",
        "code_sha256": None,
        **_scaffold_metrics(scaffold_metrics, terminal_reason="infra_failure"),
    }


def _scaffold_metrics(
    metrics: dict[str, object] | None,
    *,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    out = {
        "terminal_reason": None,
        "turns_to_first_edit": None,
        "turns_to_first_verification": None,
        "zero_edit": True,
        "zero_verification": True,
        "verification_succeeded": False,
        "prompt_snapshot": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "provider": None,
        "response_model": None,
        "submission_json": None,
    }
    if metrics:
        out.update(metrics)
    if terminal_reason is not None:
        out["terminal_reason"] = terminal_reason
    return out


def _should_run_final_polyglot_repair(output: str) -> bool:
    if not output.strip():
        return False
    if "TIMEOUT" in output or "Command timed out" in output:
        return _looks_like_repairable_reactive_timeout(output)
    if _should_reset_before_retry(output):
        return False
    return (
        _failed_test_count(output) <= 8
        or "error[E" in output
        or "Compilation failed" in output
        or "compileJava" in output
    )


def _looks_like_repairable_reactive_timeout(output: str) -> bool:
    return bool(
        re.search(
            r"\b(?:Observable|reactive|stream|blockingLast|subscribe)\b",
            output,
            re.IGNORECASE,
        )
    )


def _failed_test_count(output: str) -> int:
    patterns = (
        r"\b(\d+)\s+failed",
        r"\b(\d+)\s+tests? completed,\s+(\d+)\s+failed",
    )
    counts: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, output, flags=re.IGNORECASE):
            groups = [int(group) for group in match.groups() if group is not None]
            if groups:
                counts.append(groups[-1])
    return min(counts) if counts else 999


def _should_reset_before_retry(output: str) -> bool:
    text = output.lower()
    markers = (
        "unclosed delimiter",
        "unclosed string literal",
        "syntaxerror",
        "syntax error",
        "parse error",
        "unexpected eof",
        "unexpected end of file",
        "reached end of file",
        "expected declaration",
    )
    return any(marker in text for marker in markers)


def _merge_polyglot_metrics(
    *,
    first: dict[str, object] | None,
    second: dict[str, object] | None,
    first_loop_budget: int,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    merged = _scaffold_metrics(first)
    if second is None:
        if terminal_reason is not None:
            merged["terminal_reason"] = terminal_reason
        return merged

    second_metrics = _scaffold_metrics(second)
    merged["turns_to_first_edit"] = _merge_polyglot_turn(
        first=merged.get("turns_to_first_edit"),
        second=second_metrics.get("turns_to_first_edit"),
        offset=first_loop_budget,
    )
    merged["turns_to_first_verification"] = _merge_polyglot_turn(
        first=merged.get("turns_to_first_verification"),
        second=second_metrics.get("turns_to_first_verification"),
        offset=first_loop_budget,
    )
    merged["zero_edit"] = bool(
        merged.get("zero_edit", True) and second_metrics.get("zero_edit", True)
    )
    merged["zero_verification"] = bool(
        merged.get("zero_verification", True) and second_metrics.get("zero_verification", True)
    )
    merged["verification_succeeded"] = bool(
        merged.get("verification_succeeded", False)
        or second_metrics.get("verification_succeeded", False)
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = _sum_metric(merged.get(key), second_metrics.get(key))
    for key in (
        "prompt_snapshot",
        "provider",
        "response_model",
        "submission_json",
        "terminal_reason",
    ):
        if second_metrics.get(key) is not None:
            merged[key] = second_metrics[key]
    for key in ("diagnostic_events", "verification_evidence"):
        merged[key] = _merge_list_metric(merged.get(key), second_metrics.get(key))
    merged["validation_passed_count"] = _sum_metric(
        merged.get("validation_passed_count"),
        second_metrics.get("validation_passed_count"),
    )
    merged["validation_failed_count"] = _sum_metric(
        merged.get("validation_failed_count"),
        second_metrics.get("validation_failed_count"),
    )
    if terminal_reason is not None:
        merged["terminal_reason"] = terminal_reason
    return merged


def _merge_polyglot_turn(*, first: object, second: object, offset: int) -> int | None:
    if isinstance(first, int):
        return first
    if isinstance(second, int):
        return offset + second
    return None


def _coerce_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _sum_metric(left: object, right: object) -> int | None:
    total = 0
    seen = False
    for value in (left, right):
        if isinstance(value, int):
            total += value
            seen = True
    return total if seen else None


def _merge_list_metric(left: object, right: object) -> list[object] | None:
    merged: list[object] = []
    if isinstance(left, list):
        merged.extend(left)
    if isinstance(right, list):
        merged.extend(right)
    return merged or None


def _allowed_polyglot_test_commands(prepared) -> set[str]:
    allowed = set(prepared.test_commands)
    if len(prepared.test_commands) > 1:
        allowed.add(" && ".join(prepared.test_commands))
    if prepared.task.language == "python":
        suffix = " -v --tb=short -q"
        prefix = "python -m pytest "
        test_names = [Path(path).name for path in prepared.test_paths]
        if test_names:
            allowed.add(prefix + " ".join(test_names) + suffix)
            allowed.update(prefix + name + suffix for name in test_names)
    if prepared.task.language == "go":
        base = "go test ./..."
        allowed.update(
            {
                base,
                base + " -count=1",
                base + " -v",
                base + " -v -count=1",
                base + " -count=1 -v",
            }
        )
    return allowed


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _truncate(s: str, max_chars: int = 8000) -> str:
    return s if len(s) <= max_chars else s[-max_chars:]


def _clean_eval_output(text: str) -> str:
    return _ANSI_RE.sub("", text).strip()


def _head_tail_excerpt(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars].rstrip()
        + "\n\n... omitted "
        + str(len(text) - max_chars)
        + " chars ...\n\n"
        + text[-tail_chars:].lstrip()
    )


def _live_trace_enabled(config: BenchConfig) -> bool:
    if config.live_trace:
        return True
    raw = os.environ.get("MCODE_LIVE_TRACE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _format_live_trace_event(event_type: str, event: object) -> str | None:
    if not isinstance(event, Mapping):
        return None
    turn = event.get("turn")
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    prefix = f"turn {turn}: " if turn else ""
    if event_type == "turn_start":
        return f"turn {payload.get('turn') or turn}"
    if event_type == "generation":
        calls = payload.get("tool_calls")
        if isinstance(calls, list) and calls:
            names = ", ".join(
                str(call.get("name", "unknown")) for call in calls if isinstance(call, Mapping)
            )
            return f"{prefix}model requested {names or 'tool'}"
        model_output = payload.get("model_output")
        if isinstance(model_output, Mapping):
            preview = str(model_output.get("preview") or "").strip().replace("\n", " ")[:160]
            if preview:
                return f"{prefix}model said {preview!r}"
        return f"{prefix}model produced no tool call"
    if event_type == "read_search_target":
        tool_name = payload.get("tool_name")
        target = payload.get("path") or payload.get("query") or payload.get("pattern") or "."
        return f"{prefix}{tool_name} {target}"
    if event_type == "edit_result":
        return f"{prefix}edit {payload.get('path')} {payload.get('status') or ''}".rstrip()
    if event_type == "run_tests":
        command = payload.get("expanded_command") or payload.get("test_cmd") or "default"
        return f"{prefix}tests {payload.get('status') or 'DONE'}: {command}"
    if event_type == "tool_result":
        status = payload.get("status") or ("ok" if payload.get("success") else "failed")
        return f"{prefix}tool {payload.get('tool_name')} {status}"
    if event_type in {"final_answer", "no_tool_call", "tool_call_filter"}:
        return f"{prefix}{event_type} {dict(payload)}"
    return None


def _apply_task_shard(tasks: list, shard_count: int | None, shard_index: int | None) -> list:
    if shard_count is None and shard_index is None:
        return tasks
    if shard_count is None:
        raise ValueError("task_shard_count is required when task_shard_index is set")
    if shard_count < 1:
        raise ValueError("task_shard_count must be >= 1")
    if shard_index is None:
        shard_index = 0
    if not (0 <= shard_index < shard_count):
        raise ValueError("task_shard_index must be in [0, task_shard_count)")
    if shard_count == 1 and shard_index == 0:
        return tasks
    return tasks[shard_index::shard_count]


def _augment_run_config(config: dict) -> dict:
    out = dict(config)
    out.update(_runtime_metadata())
    return out


def _runtime_metadata() -> dict[str, str]:
    import platform
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    meta: dict[str, str] = {}
    try:
        meta["mcode_version"] = version("mcode")
    except PackageNotFoundError:
        pass
    try:
        meta["mellea_version"] = version("mellea")
    except PackageNotFoundError:
        pass

    sha = os.environ.get("MCODE_GIT_SHA") or os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            repo_root = Path(__file__).resolve().parents[3]
            if (repo_root / ".git").exists():
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if res.returncode == 0:
                    sha = (res.stdout or "").strip() or None
        except Exception:
            sha = None
    if sha:
        meta["mcode_git_sha"] = sha

    meta["python_version"] = sys.version.split()[0]
    meta["platform"] = platform.platform()
    try:
        from mellea.telemetry import is_metrics_enabled

        meta["mellea_metrics_enabled"] = "1" if is_metrics_enabled() else "0"
    except Exception:
        meta["mellea_metrics_enabled"] = "0"
    meta["mellea_requirements_available"] = "1"
    meta["mellea_sampling_available"] = "1"
    return meta


def _generation_result(session: LLMSession) -> dict[str, object] | None:
    solve_result = session.solve_result
    if solve_result is None:
        return None
    result = solve_result.as_metrics_dict()
    if solve_result.submission is not None:
        result["submission_json"] = solve_result.submission.model_dump_json()
    return result
