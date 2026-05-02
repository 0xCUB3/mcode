from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SAFE_PATH_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class PatchStats:
    touched_files: tuple[str, ...]
    added_lines: int
    deleted_lines: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class TaskArtifactRef:
    benchmark: str
    task_id: str
    task_digest: str
    repo_id: str
    artifact_version: int
    artifact_root: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationEvidence:
    verifier_name: str
    command_label: str
    command_digest: str
    status: str
    counted_as_verification: bool
    output_digest: str
    output_preview_path: str | None
    execution_time_ms: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_index: int
    patch_path: str
    patch_stats: PatchStats
    terminal_reason: str | None
    selected: bool
    submission_json: str | None
    generation_time_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    provider: str | None
    response_model: str | None
    validation_passed_count: int | None
    validation_failed_count: int | None
    zero_edit: bool
    zero_verification: bool
    verification_succeeded: bool
    trace_path: str | None
    failure_counters: dict[str, int] = field(default_factory=dict)
    verification_evidence: tuple[VerificationEvidence, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationArtifact:
    evaluator_name: str
    passed: bool
    timed_out: bool
    exit_code: int | None
    report_path: str | None
    stdout_preview_path: str | None
    stderr_preview_path: str | None
    error_class: str | None
    source_candidate_index: int
    runtime_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskArtifactManifest:
    schema_version: int
    phase: str
    generated_at: str
    run_config_digest: str
    code_sha: str | None
    model_id: str | None
    backend_name: str | None
    task: TaskArtifactRef
    candidates: tuple[CandidateArtifact, ...] = ()
    evaluations: tuple[EvaluationArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifactManifest:
    schema_version: int
    benchmark: str
    phase: str
    generated_at: str
    run_config_digest: str
    code_sha: str | None
    model_id: str | None
    backend_name: str | None
    task_manifest_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskArtifactStore:
    artifact_dir: Path
    benchmark: str
    task_id: str

    @classmethod
    def from_task(cls, *, artifact_dir: Path, benchmark: str, task_id: str) -> TaskArtifactStore:
        return cls(artifact_dir=artifact_dir, benchmark=benchmark, task_id=task_id)

    @property
    def task_root(self) -> Path:
        return self.artifact_dir / self.benchmark / _safe_task_path(self.task_id)

    @property
    def manifest_path(self) -> Path:
        return self.task_root / _MANIFEST_NAME

    def build_task_ref(
        self,
        *,
        repo_id: str,
        task_digest: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskArtifactRef:
        return TaskArtifactRef(
            benchmark=self.benchmark,
            task_id=self.task_id,
            task_digest=task_digest,
            repo_id=repo_id,
            artifact_version=SCHEMA_VERSION,
            artifact_root=str(self.task_root.relative_to(self.artifact_dir)),
            metadata=dict(metadata or {}),
        )

    def write_candidate(
        self,
        *,
        candidate_index: int,
        patch: str,
        terminal_reason: str | None,
        selected: bool,
        submission_json: str | None,
        generation_time_ms: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        provider: str | None,
        response_model: str | None,
        validation_passed_count: int | None,
        validation_failed_count: int | None,
        zero_edit: bool,
        zero_verification: bool,
        verification_succeeded: bool,
        trace_events: list[dict[str, Any]] | None,
        verification_evidence: list[VerificationEvidence] | None,
        failure_counters: dict[str, int] | None,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateArtifact:
        candidate_root = self.task_root / f"candidate-{candidate_index}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        patch_path = candidate_root / "patch.diff"
        patch_path.write_text(patch, encoding="utf-8")
        trace_path = None
        if trace_events:
            trace_path = candidate_root / "trace.json"
            trace_path.write_text(
                json.dumps(trace_events, indent=2, sort_keys=True), encoding="utf-8"
            )
        normalized_evidence: list[VerificationEvidence] = []
        for evidence_index, evidence in enumerate(verification_evidence or []):
            preview_path = None
            preview_text = str(evidence.metadata.get("output_preview", "")).strip()
            if preview_text:
                preview_file = candidate_root / f"verification-{evidence_index}.txt"
                preview_file.write_text(preview_text, encoding="utf-8")
                preview_path = str(preview_file.relative_to(self.task_root))
            normalized_evidence.append(
                VerificationEvidence(
                    verifier_name=evidence.verifier_name,
                    command_label=evidence.command_label,
                    command_digest=evidence.command_digest,
                    status=evidence.status,
                    counted_as_verification=evidence.counted_as_verification,
                    output_digest=evidence.output_digest,
                    output_preview_path=preview_path,
                    execution_time_ms=evidence.execution_time_ms,
                    started_at=evidence.started_at,
                    ended_at=evidence.ended_at,
                    timed_out=evidence.timed_out,
                    metadata={
                        key: value
                        for key, value in evidence.metadata.items()
                        if key != "output_preview"
                    },
                )
            )
        submission_value = submission_json
        if submission_value is not None:
            (candidate_root / "submission.json").write_text(submission_value, encoding="utf-8")
        return CandidateArtifact(
            candidate_index=candidate_index,
            patch_path=str(patch_path.relative_to(self.task_root)),
            patch_stats=compute_patch_stats(patch),
            terminal_reason=terminal_reason,
            selected=selected,
            submission_json=submission_value,
            generation_time_ms=generation_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            provider=provider,
            response_model=response_model,
            validation_passed_count=validation_passed_count,
            validation_failed_count=validation_failed_count,
            zero_edit=zero_edit,
            zero_verification=zero_verification,
            verification_succeeded=verification_succeeded,
            trace_path=(
                str(trace_path.relative_to(self.task_root)) if trace_path is not None else None
            ),
            failure_counters=dict(failure_counters or {}),
            verification_evidence=tuple(normalized_evidence),
            metadata=dict(metadata or {}),
        )

    def write_evaluation(
        self,
        *,
        source_candidate_index: int,
        evaluator_name: str,
        passed: bool,
        timed_out: bool,
        exit_code: int | None,
        report: dict[str, Any] | None,
        stdout: str | None,
        stderr: str | None,
        error_class: str | None,
        runtime_ms: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationArtifact:
        candidate_root = self.task_root / f"candidate-{source_candidate_index}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        report_path = None
        if report is not None:
            report_file = candidate_root / "evaluation-report.json"
            report_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            report_path = str(report_file.relative_to(self.task_root))
        stdout_path = None
        if stdout:
            stdout_file = candidate_root / "evaluation-stdout.txt"
            stdout_file.write_text(stdout, encoding="utf-8")
            stdout_path = str(stdout_file.relative_to(self.task_root))
        stderr_path = None
        if stderr:
            stderr_file = candidate_root / "evaluation-stderr.txt"
            stderr_file.write_text(stderr, encoding="utf-8")
            stderr_path = str(stderr_file.relative_to(self.task_root))
        return EvaluationArtifact(
            evaluator_name=evaluator_name,
            passed=passed,
            timed_out=timed_out,
            exit_code=exit_code,
            report_path=report_path,
            stdout_preview_path=stdout_path,
            stderr_preview_path=stderr_path,
            error_class=error_class,
            source_candidate_index=source_candidate_index,
            runtime_ms=runtime_ms,
            metadata=dict(metadata or {}),
        )

    def write_manifest(self, manifest: TaskArtifactManifest) -> Path:
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                task_manifest_to_dict(manifest),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return self.manifest_path

    def read_manifest(self) -> TaskArtifactManifest:
        return read_task_manifest(self.manifest_path)


def compute_patch_stats(patch: str) -> PatchStats:
    touched_files: list[str] = []
    added_lines = 0
    deleted_lines = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                touched_files.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added_lines += 1
        elif line.startswith("-"):
            deleted_lines += 1
    patch_bytes = patch.encode("utf-8", errors="ignore")
    return PatchStats(
        touched_files=tuple(sorted(set(touched_files))),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        byte_count=len(patch_bytes),
        sha256=sha256(patch_bytes).hexdigest(),
    )


def digest_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def digest_text(value: str) -> str:
    return sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def iso_utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_task_digest(
    *, benchmark: str, task_id: str, repo_id: str, metadata: dict[str, Any]
) -> str:
    return digest_json(
        {
            "benchmark": benchmark,
            "task_id": task_id,
            "repo_id": repo_id,
            "metadata": metadata,
        }
    )


def task_manifest_to_dict(manifest: TaskArtifactManifest) -> dict[str, Any]:
    return asdict(manifest)


def run_manifest_to_dict(manifest: RunArtifactManifest) -> dict[str, Any]:
    return asdict(manifest)


def read_task_manifest(path: Path) -> TaskArtifactManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    task_raw = raw.get("task") or {}
    candidate_raw = raw.get("candidates") or []
    evaluation_raw = raw.get("evaluations") or []
    return TaskArtifactManifest(
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        phase=str(raw.get("phase", "run")),
        generated_at=str(raw.get("generated_at", "")),
        run_config_digest=str(raw.get("run_config_digest", "")),
        code_sha=_optional_str(raw.get("code_sha")),
        model_id=_optional_str(raw.get("model_id")),
        backend_name=_optional_str(raw.get("backend_name")),
        task=TaskArtifactRef(
            benchmark=str(task_raw.get("benchmark", "")),
            task_id=str(task_raw.get("task_id", "")),
            task_digest=str(task_raw.get("task_digest", "")),
            repo_id=str(task_raw.get("repo_id", "")),
            artifact_version=int(task_raw.get("artifact_version", SCHEMA_VERSION)),
            artifact_root=str(task_raw.get("artifact_root", "")),
            metadata=dict(task_raw.get("metadata") or {}),
        ),
        candidates=tuple(_candidate_from_dict(item) for item in candidate_raw),
        evaluations=tuple(_evaluation_from_dict(item) for item in evaluation_raw),
        metadata=dict(raw.get("metadata") or {}),
    )


def iter_task_manifests(
    artifact_dir: Path,
    *,
    benchmark: str | None = None,
) -> list[TaskArtifactManifest]:
    roots = [artifact_dir / benchmark] if benchmark else [artifact_dir]
    manifests: list[TaskArtifactManifest] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(_MANIFEST_NAME)):
            manifests.append(read_task_manifest(path))
    return manifests


def _candidate_from_dict(raw: dict[str, Any]) -> CandidateArtifact:
    patch_stats_raw = raw.get("patch_stats") or {}
    return CandidateArtifact(
        candidate_index=int(raw.get("candidate_index", 0)),
        patch_path=str(raw.get("patch_path", "")),
        patch_stats=PatchStats(
            touched_files=tuple(str(item) for item in patch_stats_raw.get("touched_files") or ()),
            added_lines=int(patch_stats_raw.get("added_lines", 0)),
            deleted_lines=int(patch_stats_raw.get("deleted_lines", 0)),
            byte_count=int(patch_stats_raw.get("byte_count", 0)),
            sha256=str(patch_stats_raw.get("sha256", "")),
        ),
        terminal_reason=_optional_str(raw.get("terminal_reason")),
        selected=bool(raw.get("selected", False)),
        submission_json=_optional_str(raw.get("submission_json")),
        generation_time_ms=_optional_int(raw.get("generation_time_ms")),
        prompt_tokens=_optional_int(raw.get("prompt_tokens")),
        completion_tokens=_optional_int(raw.get("completion_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
        provider=_optional_str(raw.get("provider")),
        response_model=_optional_str(raw.get("response_model")),
        validation_passed_count=_optional_int(raw.get("validation_passed_count")),
        validation_failed_count=_optional_int(raw.get("validation_failed_count")),
        zero_edit=bool(raw.get("zero_edit", True)),
        zero_verification=bool(raw.get("zero_verification", True)),
        verification_succeeded=bool(raw.get("verification_succeeded", False)),
        trace_path=_optional_str(raw.get("trace_path")),
        failure_counters={
            str(key): int(value) for key, value in dict(raw.get("failure_counters") or {}).items()
        },
        verification_evidence=tuple(
            VerificationEvidence(
                verifier_name=str(item.get("verifier_name", "")),
                command_label=str(item.get("command_label", "")),
                command_digest=str(item.get("command_digest", "")),
                status=str(item.get("status", "")),
                counted_as_verification=bool(item.get("counted_as_verification", False)),
                output_digest=str(item.get("output_digest", "")),
                output_preview_path=_optional_str(item.get("output_preview_path")),
                execution_time_ms=_optional_int(item.get("execution_time_ms")),
                started_at=_optional_str(item.get("started_at")),
                ended_at=_optional_str(item.get("ended_at")),
                timed_out=bool(item.get("timed_out", False)),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw.get("verification_evidence") or []
        ),
        metadata=dict(raw.get("metadata") or {}),
    )


def _evaluation_from_dict(raw: dict[str, Any]) -> EvaluationArtifact:
    return EvaluationArtifact(
        evaluator_name=str(raw.get("evaluator_name", "")),
        passed=bool(raw.get("passed", False)),
        timed_out=bool(raw.get("timed_out", False)),
        exit_code=_optional_int(raw.get("exit_code")),
        report_path=_optional_str(raw.get("report_path")),
        stdout_preview_path=_optional_str(raw.get("stdout_preview_path")),
        stderr_preview_path=_optional_str(raw.get("stderr_preview_path")),
        error_class=_optional_str(raw.get("error_class")),
        source_candidate_index=int(raw.get("source_candidate_index", 0)),
        runtime_ms=_optional_int(raw.get("runtime_ms")),
        metadata=dict(raw.get("metadata") or {}),
    )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_task_path(task_id: str) -> Path:
    parts = [part for part in task_id.split("/") if part]
    if not parts:
        parts = [task_id or "task"]
    safe_parts = [_safe_path_part(part) for part in parts]
    return Path(*safe_parts)


def _safe_path_part(part: str) -> str:
    cleaned = _SAFE_PATH_CHARS_RE.sub("-", part.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "task"
