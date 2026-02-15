from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


class PatchOutput(BaseModel):
    patch: str = Field(..., description="A unified diff patch (git apply compatible), no markdown.")


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 3
    temperature: float | None = None
    seed: int | None = None
    _m: object | None = field(default=None, repr=False)

    def _model_options(self, *, system_prompt: str) -> dict:
        from mellea.backends import ModelOption

        opts: dict = {ModelOption.SYSTEM_PROMPT: system_prompt}
        if self.temperature is not None:
            opts[ModelOption.TEMPERATURE] = self.temperature
        if self.seed is not None:
            opts[ModelOption.SEED] = self.seed
        raw = os.environ.get("MCODE_MAX_NEW_TOKENS")
        if raw:
            opts[ModelOption.MAX_NEW_TOKENS] = int(raw)
        return opts

    def _strategy(self):
        from mellea.stdlib.sampling import RepairTemplateStrategy

        return RepairTemplateStrategy(loop_budget=max(1, self.loop_budget))

    def check_available(self) -> None:
        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        try:
            with mellea.start_session(
                backend_name=self.backend_name,
                model_id=self.model_id,
            ):
                return
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Could not start a Mellea session (backend={self.backend_name!r}, "
                f"model_id={self.model_id!r}). "
                "Ensure the backend is running and accessible (e.g. Ollama server) and retry."
            ) from e

    @contextmanager
    def open(self):
        if self._m is not None:
            yield self
            return

        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        with mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
        ) as m:
            self._m = m
            try:
                yield self
            finally:
                self._m = None

    def generate_code(self, *, task: Task, requirements: list | None = None):
        system_prompt = _code_system_prompt(task)
        return self._m.instruct(
            task.prompt,
            format=CodeOutput,
            strategy=self._strategy(),
            requirements=requirements or [],
            return_sampling_results=True,
            model_options=self._model_options(system_prompt=system_prompt),
        )

    def generate_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        requirements: list | None = None,
    ):
        system_prompt = (
            "You are an expert software engineer.\n"
            "Given a GitHub issue and a repository name, produce a single unified diff patch.\n"
            "The patch must fix the issue.\n"
            "The patch must apply cleanly with `git apply` from the repository root."
        )
        hints_block = f"\n\nHints:\n{hints_text.strip()}" if hints_text.strip() else ""
        description = (
            f"Repository: {repo}\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{hints_block}"
        )
        return self._m.instruct(
            description,
            format=PatchOutput,
            strategy=self._strategy(),
            requirements=requirements or [],
            return_sampling_results=True,
            model_options=self._model_options(system_prompt=system_prompt),
        )


def _code_system_prompt(task: Task) -> str:
    if task.benchmark == "humaneval":
        return (
            "You are an expert Python programmer.\n"
            "Complete the function defined in the prompt.\n"
            "Keep the function name and signature exactly the same."
        )
    return "You are an expert Python programmer."
