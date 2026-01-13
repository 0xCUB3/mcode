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

        tasks = load_benchmark(benchmark, cache_dir=self.config.cache_dir, limit=limit)
        run_id = self.results_db.start_run(benchmark, asdict(self.config))

        passed = 0
        total = 0
        with self.llm.open(), Progress() as progress:
            t = progress.add_task(f"[bold]Running {benchmark}[/bold]", total=len(tasks))
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
