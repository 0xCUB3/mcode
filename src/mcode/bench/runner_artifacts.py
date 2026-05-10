from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from mcode.bench.artifacts import (
    TaskArtifactManifest,
    TaskArtifactStore,
    VerificationEvidence,
    iso_utc_now,
    make_task_digest,
    read_task_manifest,
)


@dataclass(frozen=True)
class ArtifactRecorder:
    config: Any
    results_db_path: Path
    save_manifest: Callable[[int, TaskArtifactManifest, Path], None]

    def artifact_root(self) -> Path:
        artifact_dir = self.config.artifact_dir
        if artifact_dir is None:
            artifact_dir = self.results_db_path.parent / f"{self.results_db_path.stem}-artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def task_store(self, *, benchmark: str, task_id: str) -> TaskArtifactStore:
        return TaskArtifactStore.from_task(
            artifact_dir=self.artifact_root(),
            benchmark=benchmark,
            task_id=task_id,
        )

    def candidate_metrics_from_manifest(
        self, manifest: TaskArtifactManifest
    ) -> dict[str, object]:
        if not manifest.candidates:
            return scaffold_metrics(None)
        candidate = next(
            (item for item in manifest.candidates if item.selected),
            manifest.candidates[-1],
        )
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        return {
            "terminal_reason": candidate.terminal_reason,
            "turns_to_first_edit": coerce_optional_int(metadata.get("turns_to_first_edit")),
            "turns_to_first_verification": coerce_optional_int(
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

    def write_generation_manifest(
        self,
        *,
        run_id: int,
        benchmark: str,
        task_id: str,
        task: object,
        patch: str,
        metrics: dict[str, object] | None,
    ) -> TaskArtifactManifest:
        store = self.task_store(benchmark=benchmark, task_id=task_id)
        metadata = task_metadata(task)
        repo_id = task_repo_id(benchmark=benchmark, task=task, task_id=task_id)
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
        normalized_metrics = scaffold_metrics(metrics)
        candidate = store.write_candidate(
            candidate_index=0,
            patch=patch,
            terminal_reason=coerce_optional_str(normalized_metrics.get("terminal_reason")),
            selected=True,
            submission_json=coerce_optional_str(normalized_metrics.get("submission_json")),
            generation_time_ms=coerce_optional_int(
                normalized_metrics.get("generation_latency_ms")
            ),
            prompt_tokens=coerce_optional_int(normalized_metrics.get("prompt_tokens")),
            completion_tokens=coerce_optional_int(normalized_metrics.get("completion_tokens")),
            total_tokens=coerce_optional_int(normalized_metrics.get("total_tokens")),
            provider=coerce_optional_str(normalized_metrics.get("provider")),
            response_model=coerce_optional_str(normalized_metrics.get("response_model")),
            validation_passed_count=coerce_optional_int(
                normalized_metrics.get("validation_passed_count")
            ),
            validation_failed_count=coerce_optional_int(
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
            verification_evidence=verification_evidence(
                normalized_metrics.get("verification_evidence")
            ),
            failure_counters=artifact_failure_counters(normalized_metrics),
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
            run_config_digest=self.run_config_digest(),
            code_sha=runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=task_ref,
            candidates=(candidate,),
            evaluations=(),
            metadata={"phase": self.config.phase},
        )
        self._write_and_save(run_id, store, manifest)
        return manifest

    def load_task_manifest(
        self, *, benchmark: str, task_id: str
    ) -> tuple[TaskArtifactStore, TaskArtifactManifest]:
        store = self.task_store(benchmark=benchmark, task_id=task_id)
        if not store.manifest_path.exists():
            raise FileNotFoundError(
                f"artifact manifest not found for {benchmark}:{task_id}: {store.manifest_path}"
            )
        return store, read_task_manifest(store.manifest_path)

    def selected_candidate_patch(
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

    def append_generation_candidate(
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
        normalized_metrics = scaffold_metrics(metrics)
        candidate = store.write_candidate(
            candidate_index=candidate_index,
            patch=patch,
            terminal_reason=coerce_optional_str(normalized_metrics.get("terminal_reason")),
            selected=True,
            submission_json=coerce_optional_str(normalized_metrics.get("submission_json")),
            generation_time_ms=coerce_optional_int(
                normalized_metrics.get("generation_latency_ms")
            ),
            prompt_tokens=coerce_optional_int(normalized_metrics.get("prompt_tokens")),
            completion_tokens=coerce_optional_int(normalized_metrics.get("completion_tokens")),
            total_tokens=coerce_optional_int(normalized_metrics.get("total_tokens")),
            provider=coerce_optional_str(normalized_metrics.get("provider")),
            response_model=coerce_optional_str(normalized_metrics.get("response_model")),
            validation_passed_count=coerce_optional_int(
                normalized_metrics.get("validation_passed_count")
            ),
            validation_failed_count=coerce_optional_int(
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
            verification_evidence=verification_evidence(
                normalized_metrics.get("verification_evidence")
            ),
            failure_counters=artifact_failure_counters(normalized_metrics),
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
            run_config_digest=self.run_config_digest(),
            code_sha=runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=manifest.task,
            candidates=tuple(
                [replace(item, selected=False) for item in manifest.candidates] + [candidate]
            ),
            evaluations=manifest.evaluations,
            metadata=dict(manifest.metadata),
        )
        self._write_and_save(run_id, store, updated)
        return updated

    def append_evaluation_manifest(
        self,
        *,
        run_id: int,
        store: TaskArtifactStore,
        manifest: TaskArtifactManifest,
        candidate_index: int,
        evaluation: object,
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
            run_config_digest=self.run_config_digest(),
            code_sha=runtime_metadata().get("mcode_git_sha"),
            model_id=self.config.model_id,
            backend_name=self.config.backend_name,
            task=manifest.task,
            candidates=manifest.candidates,
            evaluations=tuple([*manifest.evaluations, evaluation_artifact]),
            metadata=dict(manifest.metadata),
        )
        self._write_and_save(run_id, store, updated)
        return updated

    def run_config_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self.config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _write_and_save(
        self,
        run_id: int,
        store: TaskArtifactStore,
        manifest: TaskArtifactManifest,
    ) -> None:
        manifest_path = store.write_manifest(manifest)
        self.save_manifest(run_id, manifest, manifest_path)


def task_metadata(task: object) -> dict[str, object]:
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


def task_repo_id(*, benchmark: str, task: object, task_id: str) -> str:
    repo = getattr(task, "repo", None)
    if isinstance(repo, str) and repo:
        return repo
    return f"{benchmark}/{task_id}"


def artifact_failure_counters(metrics: dict[str, object] | None) -> dict[str, int]:
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


def verification_evidence(items: object) -> list[VerificationEvidence]:
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
                execution_time_ms=coerce_optional_int(item.get("execution_time_ms")),
                started_at=coerce_optional_str(item.get("started_at")),
                ended_at=coerce_optional_str(item.get("ended_at")),
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


def scaffold_metrics(
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


def coerce_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def runtime_metadata() -> dict[str, str]:
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
