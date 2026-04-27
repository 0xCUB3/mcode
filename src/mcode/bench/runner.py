from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mcode.bench.results import ResultsDB, RunSummary
from mcode.execution.sandbox import DockerUnavailableError
from mcode.llm.session import LLMSession
from mcode.mellea_compat import requirements_available, sampling_available
from mcode.ui.task_reporter import choose as choose_task_reporter


def _default_cache_dir() -> Path:
    override = os.environ.get("MCODE_CACHE_DIR")
    if override:
        return Path(override)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "mcode"
    return Path("/tmp/mcode-cache")


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
    swebench_split: str = "test"
    swebench_namespace: str | None = "swebench"
    swebench_arch: str | None = None
    swebench_max_workers: int = 4
    swebench_force_rebuild: bool = False
    swebench_mem_limit: str = "4g"
    swebench_pids_limit: int = 512
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


@dataclass(frozen=True)
class PatchRepoContext:
    repo_root: Path | str
    command_fn: Callable[[str], str] | None = None
    visible_repo_root: str | None = None


def _coerce_patch_repo_context(repo_context: object) -> PatchRepoContext:
    repo_root = getattr(repo_context, "repo_root", repo_context)
    command_fn = getattr(repo_context, "command_fn", None)
    visible_repo_root = getattr(repo_context, "visible_repo_root", None)
    return PatchRepoContext(
        repo_root=repo_root,
        command_fn=command_fn,
        visible_repo_root=visible_repo_root,
    )


class BenchmarkRunner:
    def __init__(self, *, config: BenchConfig, results_db: ResultsDB):
        self.config = config
        self.results_db = results_db
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
        )

    def run_benchmark(
        self,
        benchmark: str,
        *,
        limit: int | None = None,
        task_ids: list[str] | None = None,
    ) -> RunSummary:
        name = benchmark.lower().strip()
        if name in {"swebench-lite", "swebench_lite"}:
            self.llm.check_available()
            return self._run_swebench_lite(limit=limit, task_ids=task_ids)
        if name in {"swebench-live", "swebench_live"}:
            self.llm.check_available()
            return self._run_swebench_live(limit=limit, task_ids=task_ids)
        if name in {"aider-polyglot", "aider_polyglot"}:
            self.llm.check_available()
            return self._run_aider_polyglot(limit=limit, task_ids=task_ids)
        raise ValueError(f"Unknown benchmark: {benchmark}")

    def _run_swebench_lite(
        self,
        *,
        limit: int | None,
        task_ids: list[str] | None = None,
    ) -> RunSummary:
        from mcode.bench.swebench_lite import load_swebench_lite
        from mcode.execution.swebench import SWEbenchSandbox

        tasks = load_swebench_lite(
            self.config.cache_dir,
            split=self.config.swebench_split,
            limit=limit,
            instance_ids=task_ids,
            dataset_name=self.config.swebench_dataset,
        )
        tasks = _apply_task_shard(tasks, self.config.task_shard_count, self.config.task_shard_index)
        config = _augment_run_config(asdict(self.config))
        config["planned_task_count"] = len(tasks)
        config["dataset"] = {
            "name": self.config.swebench_dataset.split("/")[-1],
            "hf_dataset": self.config.swebench_dataset,
            "split": self.config.swebench_split,
        }

        swe_sandbox = SWEbenchSandbox(
            namespace=self.config.swebench_namespace,
            arch=self.config.swebench_arch,
            max_workers=self.config.swebench_max_workers,
            mem_limit=self.config.swebench_mem_limit,
            pids_limit=self.config.swebench_pids_limit,
            force_rebuild=self.config.swebench_force_rebuild,
        )
        swe_sandbox.prepare_images([t.raw_instance for t in tasks])
        run_id = self.results_db.start_run("swebench-lite", config)

        passed = 0
        total = 0
        with choose_task_reporter() as reporter:
            reporter.total(len(tasks))
            for task in tasks:
                total += 1
                result = self._run_swebench_task(task, swe_sandbox=swe_sandbox, run_id=run_id)
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                detail = f"{task.instance_id} {'ok' if result['passed'] else 'fail'}"
                reporter.advance(detail=detail)

        return RunSummary(run_id=run_id, total=total, passed=passed)

    def _run_swebench_live(
        self,
        *,
        limit: int | None,
        task_ids: list[str] | None = None,
    ) -> RunSummary:
        from mcode.bench.swebench_live import load_swebench_live
        from mcode.execution.swebench_live import SWEbenchLiveSandbox

        tasks = load_swebench_live(
            self.config.cache_dir,
            split=self.config.swebench_split,
            limit=limit,
            instance_ids=task_ids,
        )
        tasks = _apply_task_shard(tasks, self.config.task_shard_count, self.config.task_shard_index)
        config = _augment_run_config(asdict(self.config))
        config["planned_task_count"] = len(tasks)
        config["dataset"] = {
            "name": "SWE-bench-Live",
            "hf_dataset": "SWE-bench-Live/SWE-bench-Live",
            "split": self.config.swebench_split,
        }

        live_sandbox = SWEbenchLiveSandbox(
            mem_limit=self.config.swebench_mem_limit,
            pids_limit=self.config.swebench_pids_limit,
        )
        live_sandbox.prepare_images(tasks)
        run_id = self.results_db.start_run("swebench-live", config)

        passed = 0
        total = 0
        with choose_task_reporter() as reporter:
            reporter.total(len(tasks))
            for task in tasks:
                total += 1
                result = self._run_swebench_live_task(
                    task,
                    live_sandbox=live_sandbox,
                    run_id=run_id,
                )
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                if not os.environ.get("MCODE_KEEP_IMAGES"):
                    live_sandbox.remove_image(task)
                detail = f"{task.instance_id} {'ok' if result['passed'] else 'fail'}"
                reporter.advance(detail=detail)

        return RunSummary(run_id=run_id, total=total, passed=passed)

    def _run_aider_polyglot(
        self,
        *,
        limit: int | None,
        task_ids: list[str] | None = None,
    ) -> RunSummary:
        from mcode.bench.aider_polyglot import load_aider_polyglot

        tasks = load_aider_polyglot(
            self.config.aider_polyglot_root,
            language=self.config.aider_polyglot_language,
            limit=limit,
            task_ids=task_ids,
        )
        tasks = _apply_task_shard(tasks, self.config.task_shard_count, self.config.task_shard_index)
        config = _augment_run_config(asdict(self.config))
        config["planned_task_count"] = len(tasks)
        config["dataset"] = {
            "name": "Aider Polyglot",
            "root": (
                str(self.config.aider_polyglot_root) if self.config.aider_polyglot_root else None
            ),
            "language": self.config.aider_polyglot_language,
            "retry": self.config.aider_polyglot_retry,
            "retry_loop_budget": self.config.aider_polyglot_retry_loop_budget,
        }
        run_id = self.results_db.start_run("aider-polyglot", config)

        passed = 0
        total = 0
        with choose_task_reporter() as reporter:
            reporter.total(len(tasks))
            for task in tasks:
                total += 1
                result = self._run_aider_polyglot_task(task)
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                detail = f"{task.task_id} {'ok' if result['passed'] else 'fail'}"
                reporter.advance(detail=detail)

        return RunSummary(run_id=run_id, total=total, passed=passed)

    def _run_aider_polyglot_task(self, task) -> dict[str, object]:
        from pathlib import Path

        from mcode.bench.aider_polyglot import (
            cleanup_prepared_task,
            prepare_task,
            run_test_commands,
        )
        from mcode.llm.repo_state import get_git_diff, restore_repo_snapshot

        start = time.time()
        prepared = None
        first_metrics: dict[str, object] | None = None
        final_metrics: dict[str, object] | None = None
        first_pass_snapshot = None
        final_pass_snapshot = None
        evaluation = None
        attempts_used = 0
        try:
            prepared = prepare_task(task, benchmark_root=self.config.aider_polyglot_root)
            if not prepared.stub_paths or not prepared.test_paths:
                raise RuntimeError(
                    f"benchmark task {task.task_id} is missing stubs or tests after preparation"
                )

            first_metrics, first_pass_snapshot = self._run_aider_polyglot_attempt(
                task=task,
                prepared=prepared,
                prompt=prepared.build_first_prompt(),
                loop_budget=self.config.loop_budget,
            )
            evaluation = run_test_commands(prepared)
            if not evaluation.passed and first_pass_snapshot is not None:
                restore_repo_snapshot(
                    str(prepared.work_dir),
                    Path(first_pass_snapshot.name) / "snapshot",
                )
                evaluation = run_test_commands(prepared)
            attempts_used = 1

            if not evaluation.passed and self.config.aider_polyglot_retry:
                final_metrics, final_pass_snapshot = self._run_aider_polyglot_attempt(
                    task=task,
                    prepared=prepared,
                    prompt=prepared.build_retry_prompt(evaluation.output),
                    loop_budget=self.config.aider_polyglot_retry_loop_budget,
                )
                evaluation = run_test_commands(prepared)
                if not evaluation.passed and final_pass_snapshot is not None:
                    restore_repo_snapshot(
                        str(prepared.work_dir),
                        Path(final_pass_snapshot.name) / "snapshot",
                    )
                    evaluation = run_test_commands(prepared)
                attempts_used = 2
            else:
                final_metrics = first_metrics

            terminal_reason = None
            if evaluation.passed:
                terminal_reason = "submitted"
            metrics = _merge_polyglot_metrics(
                first=first_metrics,
                second=final_metrics if attempts_used > 1 else None,
                first_loop_budget=self.config.loop_budget,
                terminal_reason=terminal_reason,
            )
            diff = get_git_diff(str(prepared.work_dir))
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
    ) -> tuple[dict[str, object] | None, object | None]:
        import shutil
        from tempfile import TemporaryDirectory

        from mcode.agent.tooling import format_tool_result
        from mcode.bench.aider_polyglot import run_command_sequence, run_single_command

        llm = self._build_llm(loop_budget=loop_budget)
        pass_snapshot = None
        allowed_commands = tuple(prepared.test_commands)

        def _capture_passing_state() -> None:
            nonlocal pass_snapshot
            if pass_snapshot is not None:
                return
            pass_snapshot = TemporaryDirectory(prefix="mcode-polyglot-pass-")
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
            status = "PASSED" if outcome.passed else ("TIMEOUT" if outcome.timed_out else "FAILED")
            return format_tool_result(label, status, outcome.output)

        llm.solve(
            repo=task.repo,
            problem_statement=prompt,
            repo_root=str(prepared.work_dir),
            n_samples=self.config.n_samples,
            test_cmds={"test_cmds": list(prepared.test_commands)},
            test_fn=test_fn,
            visible_repo_root=str(prepared.work_dir),
        )
        return _generation_result(llm), pass_snapshot

    def _generate_task_patch(
        self,
        task,
        *,
        repo_root: Path | str,
        command_fn: Callable[[str], str] | None = None,
        visible_repo_root: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
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
        task_id: str,
        generation_task,
        repo_context_factory: Callable[[], object],
        evaluate_patch: Callable[[str], _TaskEvaluation],
    ) -> dict:
        start = time.time()
        patch = ""
        scaffold_metrics: dict[str, object] | None = None
        scaffold_result: dict[str, object] | None = None
        eval_detail: _TaskEvaluation | None = None
        attempts_used = 0
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
                        )
                        elapsed_ms = int((time.time() - start) * 1000)
                        scaffold_result = _scaffold_metrics(scaffold_metrics)
                    if patch.strip():
                        eval_detail = evaluate_patch(patch)
                        if eval_detail.passed:
                            scaffold_result["terminal_reason"] = "submitted"
                        elif scaffold_result.get("verification_succeeded"):
                            scaffold_result["terminal_reason"] = "wrong_patch_after_verification"
                break
            except DockerUnavailableError as e:
                if attempts_used == 2:
                    return _task_error_result(
                        task_id=task_id,
                        start_time=start,
                        error=e,
                        scaffold_metrics=scaffold_metrics,
                        attempts_used=attempts_used,
                    )
            except Exception as e:
                return _task_error_result(
                    task_id=task_id,
                    start_time=start,
                    error=e,
                    scaffold_metrics=scaffold_metrics,
                    attempts_used=attempts_used,
                )

        if scaffold_result is not None:
            _append_terminal_diagnostic(scaffold_result, eval_detail)
        sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest() if patch else None
        return {
            "task_id": task_id,
            "passed": eval_detail.passed if eval_detail is not None else False,
            "attempts_used": attempts_used,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **(eval_detail.as_result_dict() if eval_detail is not None else {}),
            **scaffold_result,
        }


@dataclass(frozen=True)
class _TaskEvaluation:
    passed: bool
    timed_out: bool
    stdout: str | None
    stderr: str | None
    error: str | None

    def as_result_dict(self) -> dict[str, object]:
        return {
            "exit_code": None,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


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
        passed=run.resolved,
        timed_out=run.timed_out,
        stdout=_truncate(run.test_output),
        stderr=json.dumps(run.report, sort_keys=True),
        error=None if run.resolved else "Not resolved",
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
        passed=run.resolved,
        timed_out=run.timed_out,
        stdout=_truncate(run.test_output),
        stderr=json.dumps(inst_report, sort_keys=True),
        error=None if run.resolved else "Not resolved",
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
    if terminal_reason is not None:
        merged["terminal_reason"] = terminal_reason
    return merged


def _merge_polyglot_turn(*, first: object, second: object, offset: int) -> int | None:
    if isinstance(first, int):
        return first
    if isinstance(second, int):
        return offset + second
    return None


def _sum_metric(left: object, right: object) -> int | None:
    total = 0
    seen = False
    for value in (left, right):
        if isinstance(value, int):
            total += value
            seen = True
    return total if seen else None


def _allowed_polyglot_test_commands(prepared) -> set[str]:
    allowed = set(prepared.test_commands)
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


def _truncate(s: str, max_chars: int = 8000) -> str:
    return s if len(s) <= max_chars else s[-max_chars:]


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
    meta["mellea_requirements_available"] = "1" if requirements_available() else "0"
    meta["mellea_sampling_available"] = "1" if sampling_available() else "0"
    return meta


def _generation_result(session: LLMSession) -> dict[str, object] | None:
    result = session.last_solve_result
    if result is None:
        return None
    if session.last_submission:
        result["submission_json"] = json.dumps(session.last_submission, sort_keys=True)
    return result
