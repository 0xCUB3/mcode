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
        import ollama as _ollama
        from mellea.backends.tools import convert_function_to_ollama_tool

        from mcode.agent.tools import get_diff, make_tools

        tool_fns = make_tools(repo_root)

        # Build ollama tool schemas from our functions
        tool_schemas = []
        for name, fn in tool_fns.items():
            schema = convert_function_to_ollama_tool(fn, name).model_dump()
            tool_schemas.append(schema)

        # Add final_answer tool
        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Call when you are done fixing the bug.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of changes made",
                            }
                        },
                        "required": ["summary"],
                    },
                },
            }
        )

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

        user_msg = (
            f"Fix this bug in {repo} by editing the existing source code.\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{file_hint}{hints_block}\n\n"
            "Steps:\n"
            "1. search_code to find relevant code\n"
            "2. read_file to understand the buggy code\n"
            "3. apply_edit to fix the bug IN THE EXISTING SOURCE FILE\n"
            "4. call final_answer when done\n\n"
            "IMPORTANT: Only edit existing files in the repository. "
            "Do not create new files or test scripts."
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        ctx_window = int(os.environ.get("MCODE_CONTEXT_WINDOW", "32768"))
        budget = max(1, self.loop_budget) * 5
        client = _ollama.Client(host=self._m.backend._base_url)

        for turn in range(1, budget + 1):
            print(f"  [react] turn {turn}/{budget}", flush=True)
            try:
                resp = client.chat(
                    model=self._m.backend._get_ollama_model_id(),
                    messages=messages,
                    tools=tool_schemas,
                    options={"num_ctx": ctx_window},
                )
            except Exception as e:
                print(f"  [react] ollama error: {e}", flush=True)
                break

            msg = resp.message
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                print(f"  [react] no tool call, model said: {str(msg.content)[:120]}", flush=True)
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments

                if name == "final_answer":
                    print(f"  [react] final_answer: {args}", flush=True)
                    return get_diff(repo_root)

                fn = tool_fns.get(name)
                if fn is None:
                    result = f"Error: unknown tool {name}"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"Error: {e}"

                messages.append({"role": "tool", "content": str(result)})

        return get_diff(repo_root)


def _code_system_prompt(task: Task) -> str:
    if task.benchmark == "humaneval":
        return (
            "You are an expert Python programmer.\n"
            "Complete the function defined in the prompt.\n"
            "Keep the function name and signature exactly the same."
        )
    return "You are an expert Python programmer."
