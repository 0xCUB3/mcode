from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.progress import Progress

from mcode.bench.results import ResultsDB, RunSummary
from mcode.bench.tasks import Task, load_benchmark
from mcode.execution.sandbox import DockerSandbox
from mcode.llm.session import LLMSession


@dataclass(frozen=True)
class BenchConfig:
    model_id: str
    backend_name: str = "ollama"
    samples: int = 1
    retrieval: bool = False
    max_debug_iterations: int = 0
    timeout_s: int = 60
    cache_dir: Path = Path.home() / ".cache" / "mcode"
    swebench_split: str = "test"
    swebench_namespace: str | None = None
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
        self.sandbox = DockerSandbox()

    def run_benchmark(self, benchmark: str, *, limit: int | None = None) -> RunSummary:
        self.sandbox.check_available()
        self.sandbox.ensure_image()
        self.llm.check_available()

        name = benchmark.lower().strip()
        if name in {"swebench-lite", "swebench_lite"}:
            return self._run_swebench_lite(limit=limit)

        tasks = load_benchmark(name, cache_dir=self.config.cache_dir, limit=limit)
        run_id = self.results_db.start_run(name, asdict(self.config))

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
        run_id = self.results_db.start_run("swebench-lite", asdict(self.config))

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
