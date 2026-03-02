from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 3
    temperature: float | None = None
    seed: int | None = None
    strategy_name: str = "repair"
    s2_model_id: str | None = None
    s2_backend_name: str = "ollama"
    s2_solver_mode: str = "best_attempt"
    _m: object | None = field(default=None, repr=False)
    _s2_session: object | None = field(default=None, repr=False)

    def _backend_kwargs(self, *, backend_name: str | None = None) -> dict:
        name = backend_name or self.backend_name
        kwargs: dict = {}
        if name == "ollama":
            base_url = os.environ.get("OLLAMA_HOST")
            if base_url:
                kwargs["base_url"] = base_url
        elif name == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
        return kwargs

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
        ctx_raw = os.environ.get("MCODE_CONTEXT_WINDOW")
        if ctx_raw:
            opts[ModelOption.CONTEXT_WINDOW] = int(ctx_raw)
        elif self.backend_name == "ollama":
            opts[ModelOption.CONTEXT_WINDOW] = 16384
        return opts

    def _strategy(self):
        from mellea.stdlib.sampling import RepairTemplateStrategy

        budget = max(1, self.loop_budget)

        if self.strategy_name == "sofai":
            from mellea.stdlib.sampling import SOFAISamplingStrategy

            if self._s2_session is None:
                raise RuntimeError(
                    "SOFAI strategy requires an active S2 session. "
                    "Make sure s2_model_id is set and open() has been called."
                )
            return SOFAISamplingStrategy(
                s1_solver_backend=self._m.backend,
                s2_solver_backend=self._s2_session.backend,
                s2_solver_mode=self.s2_solver_mode,
                loop_budget=budget,
                feedback_strategy="first_error",
            )

        return RepairTemplateStrategy(loop_budget=budget)

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
                **self._backend_kwargs(),
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

        ctx = None
        if self.strategy_name == "sofai":
            from mellea.stdlib.context import ChatContext

            ctx = ChatContext()

        with mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
            ctx=ctx,
            **self._backend_kwargs(),
        ) as m:
            self._m = m
            try:
                if self.strategy_name == "sofai" and self.s2_model_id:
                    with mellea.start_session(
                        backend_name=self.s2_backend_name,
                        model_id=self.s2_model_id,
                        ctx=ChatContext(),
                        **self._backend_kwargs(backend_name=self.s2_backend_name),
                    ) as s2:
                        self._s2_session = s2
                        try:
                            yield self
                        finally:
                            self._s2_session = None
                else:
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
        file_paths: list[str] | None = None,
        repo_root: str,
    ) -> str:
        import asyncio

        from mellea.backends import ModelOption
        from mellea.backends.tools import tool
        from mellea.core.base import CBlock
        from mellea.stdlib import functional as mfuncs
        from mellea.stdlib.context import ChatContext

        from mcode.agent.tools import get_diff, make_tools

        tool_fns = make_tools(repo_root)
        tools = [tool(fn, name=name) for name, fn in tool_fns.items()]

        def _final_answer(answer: str) -> str:
            """Signal that you are done and provide your final answer."""
            return answer

        tools.append(tool(_final_answer, name="final_answer"))

        file_hint = ""
        if file_paths:
            file_hint = "\n\nFiles likely relevant (from BM25 ranking):\n" + "\n".join(
                f"  - {f}" for f in file_paths
            )
        hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""

        system_prompt = (
            "You are an expert software engineer fixing a bug in an "
            "open-source repository. You MUST edit existing source files "
            "to fix the bug. Do NOT create new files. Do NOT write test "
            "scripts. Only modify the existing code that contains the bug."
        )

        goal = (
            f"Fix this bug in {repo} by editing the existing source code.\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{file_hint}{hints_block}\n\n"
            "Only edit existing files. Do not create new files or test scripts.\n"
            "Call the final_answer tool when you are done."
        )

        budget = max(1, self.loop_budget) * 5
        model_opts = self._model_options(system_prompt=system_prompt)
        model_opts[ModelOption.TOOLS] = tools

        timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))

        async def _run():
            ctx = ChatContext()
            ctx = ctx.add(CBlock(goal))

            try:
                for turn in range(budget):
                    print(f"  [loop] turn {turn + 1}/{budget}", flush=True)
                    step, ctx = await asyncio.wait_for(
                        mfuncs.aact(
                            action=CBlock(""),
                            context=ctx,
                            backend=self._m.backend,
                            strategy=None,
                            model_options=model_opts,
                            tool_calls=True,
                        ),
                        timeout=timeout_s,
                    )

                    if step.tool_calls:
                        tool_msgs = mfuncs._call_tools(step, backend=self._m.backend)
                        for msg in tool_msgs:
                            ctx = ctx.add(msg)
                            if msg.name == "final_answer":
                                print(f"  [loop] final_answer: {msg.content[:120]}", flush=True)
                                return get_diff(repo_root)

                print("  [loop] budget exhausted without final_answer", flush=True)
            except TimeoutError:
                print(f"  [loop] timed out after {timeout_s}s", flush=True)
            return get_diff(repo_root)

        return asyncio.run(_run())


def _code_system_prompt(task: Task) -> str:
    if task.benchmark == "humaneval":
        return (
            "You are an expert Python programmer.\n"
            "Complete the function defined in the prompt.\n"
            "Keep the function name and signature exactly the same."
        )
    return "You are an expert Python programmer."
