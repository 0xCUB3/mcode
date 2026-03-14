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
        if self.strategy_name == "raw":
            # Raw mode uses OpenAI API directly, skip mellea check
            from openai import OpenAI

            client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY", "unused"),
            )
            client.models.list()
            return

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
        if self.strategy_name == "raw":
            yield self
            return

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
        repo_root: str,
        n_samples: int = 1,
        test_cmds: list[str] | None = None,
        test_fn: object | None = None,
    ) -> str:
        if self.strategy_name == "raw":
            return self._generate_patch_raw(
                repo=repo,
                problem_statement=problem_statement,
                hints_text=hints_text,
                repo_root=repo_root,
            )

        import asyncio
        import subprocess

        from mellea.agent.tools import make_agent_tools
        from mellea.stdlib.context import ChatContext
        from mellea.stdlib.frameworks.react import react

        tools = make_agent_tools(repo_root, test_cmds=test_cmds, test_fn=test_fn)

        # Build repo map for initial context.
        repo_map_text = ""
        try:
            from mellea.agent.repomap import build_repo_map

            repo_map_text = build_repo_map(repo_root, problem_statement, max_tokens=2048)
        except Exception as e:
            print(f"  [repo_map] failed: {e}", flush=True)

        repo_map_block = f"\n\nRepository structure:\n{repo_map_text}" if repo_map_text else ""
        hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""

        if os.environ.get("MCODE_EXPLORE_PROMPT", "1") == "1":
            system_prompt = (
                "You are an expert software engineer fixing a bug in an "
                "open-source repository. You MUST edit existing source files "
                "to fix the bug. Do NOT create new files. Do NOT write test "
                "scripts. Only modify the existing code that contains the bug.\n\n"
                "Strategy:\n"
                "1. EXPLORE: Read the issue carefully. Search the codebase to "
                "find the relevant code. Read multiple files to understand "
                "the context. Do NOT edit anything yet.\n"
                "2. DIAGNOSE: Before making any edit, explain the root cause "
                "in your reasoning. If you cannot explain exactly why the "
                "current code is wrong, keep reading.\n"
                "3. EDIT: Make the minimal fix. Change the fewest lines "
                "possible. Prefer fixing the root cause over adding "
                "workarounds.\n"
                "4. VERIFY: Review your edit by reading the changed file. "
                "Make sure you didn't break anything.\n"
                "5. Call final_answer when done.\n\n"
                "Do NOT jump to editing after reading just one file. "
                "Understand the problem fully first."
            )
        else:
            system_prompt = (
                "You are an expert software engineer fixing a bug in an "
                "open-source repository. You MUST edit existing source files "
                "to fix the bug. Do NOT create new files. Do NOT write test "
                "scripts. Only modify the existing code that contains the bug.\n\n"
                "Strategy:\n"
                "1. Read the issue carefully\n"
                "2. Search the codebase to find the relevant code\n"
                "3. Identify the root cause\n"
                "4. Make the minimal edit to fix it\n"
                "5. Call final_answer when done"
            )

        test_block = ""
        if test_fn is not None or test_cmds:
            test_block = (
                "\n\nYou have a run_tests tool. Use it after editing to verify "
                "your fix. Pass 'default' to run the task's test suite."
            )

        goal = (
            f"Fix this bug in {repo} by editing the existing source code.\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{repo_map_block}{hints_block}{test_block}\n\n"
            "Only edit existing files. Do not create new files or test scripts."
        )

        budget = max(1, self.loop_budget)
        model_opts = self._model_options(system_prompt=system_prompt)

        timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))

        use_budget_warning = os.environ.get("MCODE_BUDGET_WARNING", "1") == "1"
        use_mid_nudge = os.environ.get("MCODE_MID_NUDGE", "0") == "1"

        def _on_turn(turn, total, ctx):
            from mellea.stdlib.components.chat import Message

            if use_mid_nudge and total > 5 and turn == total // 2:
                ctx = ctx.add(
                    Message(
                        role="user",
                        content=(
                            "You are halfway through your budget. If you have not "
                            "made any edits yet, you need to start editing NOW. "
                            "Pick the most likely fix based on what you've read "
                            "and make the edit. Do not spend more turns just reading."
                        ),
                    )
                )
            if use_budget_warning and total > 3 and turn == total - 2:
                ctx = ctx.add(
                    Message(
                        role="user",
                        content=(
                            "WARNING: You have 2 turns left. If you have already "
                            "made your edit, call final_answer now. If not, make "
                            "your best fix and call final_answer immediately."
                        ),
                    )
                )
            return ctx

        async def _one_attempt():
            try:
                result, _ = await asyncio.wait_for(
                    react(
                        goal=goal,
                        context=ChatContext(),
                        backend=self._m.backend,
                        tools=tools,
                        loop_budget=budget,
                        model_options=model_opts,
                        on_turn=_on_turn if (use_budget_warning or use_mid_nudge) else None,
                    ),
                    timeout=timeout_s,
                )
                print(f"  [react] final_answer: {result.value[:120]}", flush=True)
            except TimeoutError:
                print(f"  [react] timed out after {timeout_s}s", flush=True)
            except RuntimeError:
                print("  [react] budget exhausted without final_answer", flush=True)
            return _get_diff(repo_root)

        def _reset_repo():
            subprocess.run(["git", "checkout", "."], cwd=repo_root, capture_output=True)

        if n_samples <= 1:
            return asyncio.run(_one_attempt())

        # Multiple samples: run react() n_samples times, pick most common diff
        from collections import Counter

        diffs: list[str] = []
        for i in range(n_samples):
            print(f"\n  [sample {i + 1}/{n_samples}]", flush=True)
            diff = asyncio.run(_one_attempt())
            diffs.append(diff)
            if i < n_samples - 1:
                _reset_repo()

        non_empty = [d for d in diffs if d and d.strip()]
        if not non_empty:
            return ""

        # Vote: most common diff wins
        counts = Counter(non_empty)
        best_diff, best_count = counts.most_common(1)[0]
        print(
            f"  [voting] {len(non_empty)}/{n_samples} produced patches, "
            f"best has {best_count} votes",
            flush=True,
        )

        # Apply the winning diff
        _reset_repo()
        proc = subprocess.run(
            ["git", "apply", "--allow-empty"],
            input=best_diff,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  [voting] git apply failed: {proc.stderr[:200]}", flush=True)
            # Fall back to longest diff
            non_empty.sort(key=len, reverse=True)
            return non_empty[0]
        return _get_diff(repo_root)

    def _generate_patch_raw(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        repo_root: str,
    ) -> str:
        """Single-shot patch generation: no tools, no ReAct loop.

        Uses the OpenAI-compatible API directly instead of mellea's agent
        framework, to measure the raw model's patch generation ability.
        """
        import subprocess

        from openai import OpenAI

        repo_map_text = ""
        try:
            from mellea.agent.repomap import build_repo_map

            repo_map_text = build_repo_map(repo_root, problem_statement, max_tokens=4096)
        except Exception as e:
            print(f"  [repo_map] failed: {e}", flush=True)

        repo_map_block = f"\n\nRepository structure:\n{repo_map_text}" if repo_map_text else ""
        hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""

        system_prompt = (
            "You are an expert software engineer. Given a bug report and "
            "repository structure, output a unified diff (git diff format) "
            "that fixes the bug. Output ONLY the diff, nothing else. "
            "No explanation, no markdown fences, just the raw diff."
        )

        user_prompt = (
            f"Repository: {repo}\n\n"
            f"Bug report:\n{problem_statement.strip()}"
            f"{repo_map_block}{hints_block}\n\n"
            "Output the unified diff to fix this bug:"
        )

        timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", "120"))
        max_tokens = int(os.environ.get("MCODE_MAX_NEW_TOKENS", "4096"))

        try:
            client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY", "unused"),
            )
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=self.temperature or 0.0,
                timeout=timeout_s,
            )
            diff_text = response.choices[0].message.content or ""

            # Strip markdown fences if the model wrapped it
            if diff_text.startswith("```"):
                lines = diff_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                diff_text = "\n".join(lines)

            print(f"  [raw] got {len(diff_text)} char diff", flush=True)

            if not diff_text.strip():
                return ""

            # Apply the diff
            proc = subprocess.run(
                ["git", "apply", "--allow-empty"],
                input=diff_text,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"  [raw] git apply failed: {proc.stderr[:200]}", flush=True)
                # Try with --3way
                proc2 = subprocess.run(
                    ["git", "apply", "--3way", "--allow-empty"],
                    input=diff_text,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                )
                if proc2.returncode != 0:
                    print("  [raw] git apply --3way also failed", flush=True)
                    return ""

            return _get_diff(repo_root)

        except Exception as e:
            print(f"  [raw] error: {e}", flush=True)
            return ""


def _get_diff(repo_root: str) -> str:
    import subprocess
    from pathlib import Path

    git_dir = Path(repo_root) / ".git"
    if not git_dir.exists():
        print(f"  [diff] no .git in {repo_root}, cannot produce patch", flush=True)
        return ""

    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [diff] git diff failed: {result.stderr[:300]}", flush=True)
    return result.stdout


def _code_system_prompt(task: Task) -> str:
    if task.benchmark == "humaneval":
        return (
            "You are an expert Python programmer.\n"
            "Complete the function defined in the prompt.\n"
            "Keep the function name and signature exactly the same."
        )
    return "You are an expert Python programmer."
