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
    loop_budget: int = 3
    temperature: float | None = None
    seed: int | None = None
    strategy: str = "repair"
    s2_model_id: str | None = None
    s2_backend_name: str = "ollama"
    s2_solver_mode: str = "best_attempt"
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
            strategy_name=config.strategy,
            s2_model_id=config.s2_model_id,
            s2_backend_name=config.s2_backend_name,
            s2_solver_mode=config.s2_solver_mode,
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
            dataset_name=self.config.swebench_dataset,
        )
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.instance_id in id_set]
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
                try:
                    result = self._run_swebench_task(task, swe_sandbox=swe_sandbox, run_id=run_id)
                except DockerUnavailableError:
                    raise
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
        )
        if task_ids:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.instance_id in id_set]
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
                try:
                    result = self._run_swebench_live_task(
                        task,
                        live_sandbox=live_sandbox,
                        run_id=run_id,
                    )
                except DockerUnavailableError:
                    raise
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
    ) -> str:
        verification_metadata = getattr(task, "raw_instance", None)
        if verification_metadata is None:
            verification_metadata = getattr(task, "test_cmds", None)

        with self.llm.open():
            return self.llm.generate_patch(
                repo=task.repo,
                problem_statement=task.problem_statement,
                hints_text=task.hints_text or "",
                repo_root=str(repo_root),
                n_samples=self.config.n_samples,
                test_cmds=verification_metadata,
                command_fn=command_fn,
                visible_repo_root=visible_repo_root,
            )

    def _run_swebench_live_task(self, task, *, live_sandbox, run_id: int) -> dict:
        start = time.time()
        try:
            with live_sandbox.repo_context(task) as repo_context:
                patch_context = _coerce_patch_repo_context(repo_context)
                try:
                    patch = self._generate_task_patch(
                        task,
                        repo_root=patch_context.repo_root,
                        command_fn=patch_context.command_fn,
                        visible_repo_root=patch_context.visible_repo_root,
                    )
                except DockerUnavailableError:
                    raise
                except Exception as e:
                    elapsed_ms = int((time.time() - start) * 1000)
                    tb = traceback.format_exc()
                    return {
                        "task_id": task.instance_id,
                        "passed": False,
                        "attempts_used": 0,
                        "time_ms": elapsed_ms,
                        "exit_code": None,
                        "timed_out": False,
                        "stdout": None,
                        "stderr": (_truncate(tb, max_chars=8000) if tb else None),
                        "error": f"{type(e).__name__}: {e}",
                        "code_sha256": None,
                    }
                elapsed_ms = int((time.time() - start) * 1000)

                has_patch = bool(patch and patch.strip())
                last_detail: dict = {}
                if has_patch:
                    run = live_sandbox.evaluate_patch(
                        task=task,
                        patch=patch,
                        run_id=f"mcode-{run_id}",
                        timeout_s=self.config.timeout_s,
                    )
                    last_detail = {
                        "exit_code": None,
                        "timed_out": run.timed_out,
                        "stdout": _truncate(run.test_output),
                        "stderr": json.dumps(run.report, sort_keys=True),
                        "error": None if run.resolved else "Not resolved",
                    }
        except DockerUnavailableError:
            raise
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            tb = traceback.format_exc()
            return {
                "task_id": task.instance_id,
                "passed": False,
                "attempts_used": 0,
                "time_ms": elapsed_ms,
                "exit_code": None,
                "timed_out": False,
                "stdout": None,
                "stderr": (_truncate(tb, max_chars=8000) if tb else None),
                "error": f"{type(e).__name__}: {e}",
                "code_sha256": None,
            }

        sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest() if patch else None

        return {
            "task_id": task.instance_id,
            "passed": last_detail.get("error") is None if last_detail else False,
            "attempts_used": 1,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **last_detail,
        }

    def _run_swebench_task(self, task, *, swe_sandbox, run_id: int) -> dict:
        start = time.time()
        try:
            with swe_sandbox.repo_context(task.raw_instance) as repo_context:
                patch_context = _coerce_patch_repo_context(repo_context)
                try:
                    patch = self._generate_task_patch(
                        task,
                        repo_root=patch_context.repo_root,
                        command_fn=patch_context.command_fn,
                        visible_repo_root=patch_context.visible_repo_root,
                    )
                except DockerUnavailableError:
                    raise
                except Exception as e:
                    elapsed_ms = int((time.time() - start) * 1000)
                    tb = traceback.format_exc()
                    return {
                        "task_id": task.instance_id,
                        "passed": False,
                        "attempts_used": 0,
                        "time_ms": elapsed_ms,
                        "exit_code": None,
                        "timed_out": False,
                        "stdout": None,
                        "stderr": _truncate(tb, max_chars=8000) if tb else None,
                        "error": f"{type(e).__name__}: {e}",
                        "code_sha256": None,
                    }
                elapsed_ms = int((time.time() - start) * 1000)

                has_patch = bool(patch and patch.strip())
                last_detail: dict = {}
                if has_patch:
                    run = swe_sandbox.evaluate_patch(
                        instance=task.raw_instance,
                        model_id=self.config.model_id,
                        patch=patch,
                        run_id=f"mcode-{run_id}",
                        timeout_s=self.config.timeout_s,
                    )
                    inst_report = run.report.get(task.instance_id, {})
                    last_detail = {
                        "exit_code": None,
                        "timed_out": run.timed_out,
                        "stdout": _truncate(run.test_output),
                        "stderr": json.dumps(inst_report, sort_keys=True),
                        "error": None if run.resolved else "Not resolved",
                    }
        except DockerUnavailableError:
            raise
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            tb = traceback.format_exc()
            return {
                "task_id": task.instance_id,
                "passed": False,
                "attempts_used": 0,
                "time_ms": elapsed_ms,
                "exit_code": None,
                "timed_out": False,
                "stdout": None,
                "stderr": _truncate(tb, max_chars=8000) if tb else None,
                "error": f"{type(e).__name__}: {e}",
                "code_sha256": None,
            }

        sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest() if patch else None

        return {
            "task_id": task.instance_id,
            "passed": last_detail.get("error") is None if last_detail else False,
            "attempts_used": 1,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **last_detail,
        }


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
    return meta
