from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    _m: object | None = None

    def check_available(self) -> None:
        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        try:
            with mellea.start_session(backend_name=self.backend_name, model_id=self.model_id):
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

        with mellea.start_session(backend_name=self.backend_name, model_id=self.model_id) as m:
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

    def _instruct_code(self, prompt: str) -> str:
        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        m = self._m
        if m is not None:
            thunk = m.instruct(prompt, format=CodeOutput)
            out = getattr(thunk, "value", thunk)
            return _extract_code(out)

        with mellea.start_session(backend_name=self.backend_name, model_id=self.model_id) as m2:
            thunk = m2.instruct(prompt, format=CodeOutput)
            out = getattr(thunk, "value", thunk)
            return _extract_code(out)


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
