from __future__ import annotations

import os
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task

_T = TypeVar("_T")


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


class PatchOutput(BaseModel):
    patch: str = Field(..., description="A unified diff patch (git apply compatible), no markdown.")


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    _m: object | None = None

    def _default_model_options(self) -> dict | None:
        raw = os.environ.get("MCODE_MAX_NEW_TOKENS")
        if not raw:
            return None
        try:
            max_new_tokens = int(raw)
        except ValueError:
            raise ValueError(f"MCODE_MAX_NEW_TOKENS must be an int (got {raw!r})")
        if max_new_tokens < 1:
            raise ValueError("MCODE_MAX_NEW_TOKENS must be >= 1")

        try:
            from mellea.backends.types import ModelOption
        except Exception:
            # If mellea isn't installed, we don't want to fail until check_available/open.
            return None

        return {ModelOption.MAX_NEW_TOKENS: max_new_tokens}

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
                model_options=self._default_model_options(),
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
            model_options=self._default_model_options(),
        ) as m:
            self._m = m
            try:
                yield self
            finally:
                self._m = None

    def generate_code(self, *, task: Task) -> str:
        if task.benchmark == "humaneval":
            prompt = (
                "You are an expert Python programmer.\n"
                "Complete the function defined in the prompt.\n"
                "Keep the function name and signature exactly the same.\n"
                "Return only Python code. Do not use markdown.\n\n"
                f"{task.prompt}\n"
            )
        else:
            prompt = (
                "You are an expert Python programmer.\n"
                "Return only Python code. Do not use markdown.\n\n"
                f"{task.prompt}\n"
            )
        return self._instruct_code(prompt)

    def debug_code(self, *, task: Task, code: str, error: str) -> str:
        prompt = (
            "You are an expert Python programmer.\n"
            "Fix the given Python code so that it passes the tests.\n"
            "Return only Python code. Do not use markdown.\n\n"
            f"Task:\n{task.prompt}\n\n"
            f"Current code:\n{code}\n\n"
            f"Test failure / stderr:\n{error}\n"
        )
        return self._instruct_code(prompt)

    def generate_patch(self, *, repo: str, problem_statement: str, hints_text: str = "") -> str:
        hints_block = f"\n\nHints:\n{hints_text.strip()}\n" if hints_text.strip() else ""
        prompt = (
            "You are an expert software engineer.\n"
            "Given a GitHub issue and a repository name, produce a single unified diff patch.\n"
            "The patch must fix the issue.\n"
            "Return ONLY the patch text. Do not use markdown. Do not add explanations.\n"
            "The patch must apply cleanly with `git apply` from the repository root.\n\n"
            f"Repository: {repo}\n\n"
            f"Issue:\n{problem_statement.strip()}\n"
            f"{hints_block}"
        )
        return self._instruct_patch(prompt)

    def debug_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        previous_patch: str,
        failure_output: str,
        hints_text: str = "",
    ) -> str:
        hints_block = f"\n\nHints:\n{hints_text.strip()}\n" if hints_text.strip() else ""
        prompt = (
            "You are an expert software engineer.\n"
            "Fix the patch so that the repository tests pass.\n"
            "Return ONLY a unified diff patch (git apply compatible), no markdown.\n"
            "The patch should be complete and apply cleanly to the original repository state.\n\n"
            f"Repository: {repo}\n\n"
            f"Issue:\n{problem_statement.strip()}\n"
            f"{hints_block}\n"
            f"Previous patch:\n{previous_patch}\n\n"
            f"Test output / failure:\n{failure_output}\n"
        )
        return self._instruct_patch(prompt)

    def _instruct_code(self, prompt: str) -> str:
        return self._instruct(prompt, format_model=CodeOutput, extractor=_extract_code)

    def _instruct_patch(self, prompt: str) -> str:
        return self._instruct(prompt, format_model=PatchOutput, extractor=_extract_patch)

    def _instruct(
        self,
        prompt: str,
        *,
        format_model: type[_T],
        extractor: Callable[[Any], str],
    ) -> str:
        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        m = self._m
        if m is not None:
            thunk = m.instruct(
                prompt,
                format=format_model,
                model_options=self._default_model_options(),
            )
            out = getattr(thunk, "value", thunk)
            return extractor(out)

        with mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
            model_options=self._default_model_options(),
        ) as m2:
            thunk = m2.instruct(
                prompt,
                format=format_model,
                model_options=self._default_model_options(),
            )
            out = getattr(thunk, "value", thunk)
            return extractor(out)


def _extract_code(out: object) -> str:
    if isinstance(out, CodeOutput):
        return out.code
    if isinstance(out, dict) and "code" in out:
        return str(out["code"])
    if isinstance(out, str):
        s = out.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = "\n".join(s.splitlines()[1:]).strip()
        if s.startswith("{") and '"code"' in s:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "code" in parsed:
                    return str(parsed["code"])
            except Exception:
                pass
        return s
    return str(out)


def _extract_patch(out: object) -> str:
    if isinstance(out, PatchOutput):
        return out.patch
    if isinstance(out, dict) and "patch" in out:
        return str(out["patch"])
    if isinstance(out, str):
        s = out.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = "\n".join(s.splitlines()[1:]).strip()
        if s.startswith("{") and '"patch"' in s:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "patch" in parsed:
                    return str(parsed["patch"])
            except Exception:
                pass
        return s
    return str(out)
