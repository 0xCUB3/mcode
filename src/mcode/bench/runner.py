from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.progress import Progress

from mcode.bench.results import ResultsDB, RunSummary
from mcode.execution.sandbox import DockerUnavailableError
from mcode.llm.session import LLMSession
from mcode.mellea_compat import requirements_available, sampling_available


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
    n_samples: int = 1
    sampling_strategy: str = "none"
    sampling_budget: int | None = None


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
        self.llm = LLMSession(
            model_id=config.model_id,
            backend_name=config.backend_name,
            loop_budget=config.loop_budget,
            temperature=config.temperature,
            seed=config.seed,
            sampling_strategy=config.sampling_strategy,
            sampling_budget=config.sampling_budget,
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
        with Progress() as progress:
            t = progress.add_task("[bold]Running swebench-lite[/bold]", total=len(tasks))
            for task in tasks:
                total += 1
                result = self._run_swebench_task(task, swe_sandbox=swe_sandbox, run_id=run_id)
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                progress.advance(t, 1)

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
        with Progress() as progress:
            t = progress.add_task("[bold]Running swebench-live[/bold]", total=len(tasks))
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
                progress.advance(t, 1)

        return RunSummary(run_id=run_id, total=total, passed=passed)

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
