from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.progress import Progress

from mcode.bench.results import ResultsDB, RunSummary
from mcode.bench.tasks import Task, load_benchmark
from mcode.execution.sandbox import DockerSandbox
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
    samples: int = 1
    retrieval: bool = False
    max_debug_iterations: int = 0
    timeout_s: int = 60
    sandbox: str = "docker"
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


class BenchmarkRunner:
    def __init__(self, *, config: BenchConfig, results_db: ResultsDB):
        self.config = config
        self.results_db = results_db
        self.llm = LLMSession(model_id=config.model_id, backend_name=config.backend_name)
        self.sandbox = _make_sandbox(config)

    def run_benchmark(self, benchmark: str, *, limit: int | None = None) -> RunSummary:
        name = benchmark.lower().strip()
        if name in {"swebench-lite", "swebench_lite"}:
            self.llm.check_available()
            return self._run_swebench_lite(limit=limit)

        self.sandbox.check_available()
        self.sandbox.ensure_image()
        self.llm.check_available()

        tasks = load_benchmark(name, cache_dir=self.config.cache_dir, limit=limit)
        tasks = _apply_task_shard(tasks, self.config.task_shard_count, self.config.task_shard_index)
        config = _augment_run_config(asdict(self.config))
        config["dataset"] = _dataset_metadata(name, cache_dir=self.config.cache_dir) or {}
        run_id = self.results_db.start_run(name, config)

        passed = 0
        total = 0
        with self.llm.open(), Progress() as progress:
            t = progress.add_task(f"[bold]Running {name}[/bold]", total=len(tasks))
            for task in tasks:
                total += 1
                result = self.run_task(task)
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                progress.advance(t, 1)

        return RunSummary(run_id=run_id, total=total, passed=passed)

    def run_task(self, task: Task) -> dict:
        start = time.time()

        def evaluate(code: str) -> tuple[bool, dict]:
            combined = _combine_for_eval(task, code)
            run = self.sandbox.run_python(combined, timeout_s=self.config.timeout_s)
            ok = run.success
            detail = {
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "error": run.error,
            }
            return ok, detail

        last_error_detail: dict | None = None
        samples_generated = 0
        debug_iterations_used = 0
        final_code = ""

        for _ in range(self.config.samples):
            samples_generated += 1
            code = self.llm.generate_code(task=task)
            final_code = code

            ok, detail = evaluate(code)
            last_error_detail = detail
            if ok:
                break

            for _ in range(self.config.max_debug_iterations):
                debug_iterations_used += 1
                code = self.llm.debug_code(task=task, code=code, error=detail.get("stderr") or "")
                final_code = code
                ok, detail = evaluate(code)
                last_error_detail = detail
                if ok:
                    break
            if ok:
                break

        elapsed_ms = int((time.time() - start) * 1000)
        passed = bool(
            last_error_detail
            and (last_error_detail.get("exit_code") == 0)
            and not last_error_detail.get("timed_out")
        )

        sha = (
            hashlib.sha256(final_code.encode("utf-8", errors="ignore")).hexdigest()
            if final_code
            else None
        )

        return {
            "task_id": task.task_id,
            "passed": passed,
            "samples_generated": samples_generated,
            "debug_iterations_used": debug_iterations_used,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **(last_error_detail or {}),
        }

    def _run_swebench_lite(self, *, limit: int | None) -> RunSummary:
        from mcode.bench.swebench_lite import load_swebench_lite
        from mcode.execution.swebench import SWEbenchSandbox

        tasks = load_swebench_lite(
            self.config.cache_dir,
            split=self.config.swebench_split,
            limit=limit,
        )
        tasks = _apply_task_shard(tasks, self.config.task_shard_count, self.config.task_shard_index)
        config = _augment_run_config(asdict(self.config))
        config["dataset"] = {
            "name": "SWE-bench_Lite",
            "hf_dataset": "SWE-bench/SWE-bench_Lite",
            "split": self.config.swebench_split,
        }
        run_id = self.results_db.start_run("swebench-lite", config)

        swe_sandbox = SWEbenchSandbox(
            namespace=self.config.swebench_namespace,
            arch=self.config.swebench_arch,
            max_workers=self.config.swebench_max_workers,
            mem_limit=self.config.swebench_mem_limit,
            pids_limit=self.config.swebench_pids_limit,
            force_rebuild=self.config.swebench_force_rebuild,
        )
        swe_sandbox.prepare_images([t.raw_instance for t in tasks])

        passed = 0
        total = 0
        with self.llm.open(), Progress() as progress:
            t = progress.add_task("[bold]Running swebench-lite[/bold]", total=len(tasks))
            for task in tasks:
                total += 1
                result = self._run_swebench_task(task, swe_sandbox=swe_sandbox, run_id=run_id)
                if result["passed"]:
                    passed += 1
                self.results_db.save_task_result(run_id, result)
                progress.advance(t, 1)

        return RunSummary(run_id=run_id, total=total, passed=passed)

    def _run_swebench_task(self, task, *, swe_sandbox, run_id: int) -> dict:
        import json as _json
        import time as _time

        start = _time.time()
        samples_generated = 0
        debug_iterations_used = 0

        last_detail: dict | None = None
        final_patch = ""

        def _truncate(s: str, max_chars: int = 8000) -> str:
            if len(s) <= max_chars:
                return s
            return s[-max_chars:]

        for _ in range(self.config.samples):
            samples_generated += 1
            patch = self.llm.generate_patch(
                repo=task.repo,
                problem_statement=task.problem_statement,
                hints_text=task.hints_text,
            )
            final_patch = patch

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
                "stderr": _json.dumps(inst_report, sort_keys=True),
                "error": None if run.resolved else "Not resolved",
            }
            if run.resolved and not run.timed_out:
                break

            for _ in range(self.config.max_debug_iterations):
                debug_iterations_used += 1
                patch = self.llm.debug_patch(
                    repo=task.repo,
                    problem_statement=task.problem_statement,
                    hints_text=task.hints_text,
                    previous_patch=patch,
                    failure_output=_truncate(run.test_output, max_chars=16000),
                )
                final_patch = patch
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
                    "stderr": _json.dumps(inst_report, sort_keys=True),
                    "error": None if run.resolved else "Not resolved",
                }
                if run.resolved and not run.timed_out:
                    break
            if run.resolved and not run.timed_out:
                break

        elapsed_ms = int((_time.time() - start) * 1000)
        passed = bool(
            last_detail
            and last_detail.get("error") is None
            and not bool(last_detail.get("timed_out", False))
        )
        import hashlib as _hashlib

        sha = (
            _hashlib.sha256(final_patch.encode("utf-8", errors="ignore")).hexdigest()
            if final_patch
            else None
        )

        return {
            "task_id": task.instance_id,
            "passed": passed,
            "samples_generated": samples_generated,
            "debug_iterations_used": debug_iterations_used,
            "time_ms": elapsed_ms,
            "code_sha256": sha,
            **(last_detail or {}),
        }


def _combine_for_eval(task: Task, code: str) -> str:
    if task.benchmark == "humaneval":
        entry = task.entry_point
        if not entry:
            raise ValueError(f"HumanEval task missing entry_point: {task.task_id}")
        return (
            f"{code}\n\n"
            f"{task.test_code}\n\n"
            "def __mcode_main():\n"
            f"    check({entry})\n\n"
            "if __name__ == '__main__':\n"
            "    __mcode_main()\n"
        )

    if task.benchmark == "mbpp":
        return (
            f"{code}\n\n"
            "# --- mbpp tests ---\n"
            f"{task.test_code}\n"
        )

    raise ValueError(f"Unsupported benchmark for eval: {task.benchmark}")


def _make_sandbox(config: BenchConfig):
    name = config.sandbox.strip().lower()
    if name == "docker":
        return DockerSandbox()
    if name == "process":
        from mcode.execution.process_sandbox import ProcessSandbox

        return ProcessSandbox()
    raise ValueError(f"Unknown sandbox: {config.sandbox!r}")


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
    import os
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


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_metadata(benchmark: str, *, cache_dir: Path) -> dict[str, str | None] | None:
    name = benchmark.lower().strip()
    if name in {"humaneval", "human-eval"}:
        from mcode.bench.humaneval import HUMANEVAL_URL

        path = cache_dir / "humaneval" / "HumanEval.jsonl.gz"
        return {
            "name": "HumanEval",
            "url": HUMANEVAL_URL,
            "sha256": _sha256_path(path),
        }
    if name == "mbpp":
        from mcode.bench.mbpp import MBPP_URL

        path = cache_dir / "mbpp" / "mbpp.jsonl"
        return {
            "name": "MBPP",
            "url": MBPP_URL,
            "sha256": _sha256_path(path),
        }
    return None
