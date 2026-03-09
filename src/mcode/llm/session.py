from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


class FileLocalization(BaseModel):
    files: list[str] = Field(
        ..., description="File paths most likely to need editing, from the candidate list"
    )


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

    def localize_files(
        self,
        *,
        candidates: list[str],
        problem_statement: str,
        max_files: int = 5,
    ) -> list[str]:
        system_prompt = (
            "You are an expert at localizing bugs in code repositories. "
            "Given a ranked list of candidate files and an issue description, "
            "pick 1 to 5 files most likely to need editing. "
            "Choose ONLY from the candidate list. "
            "Reply with ONLY the file paths, one per line, nothing else."
        )
        candidate_list = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(candidates))
        prompt = (
            f"Issue:\n{problem_statement.strip()}\n\n"
            f"Candidate files (BM25-ranked):\n{candidate_list}\n\n"
            f"Pick 1-{max_files} files from the list above that need editing to fix this issue.\n"
            "Reply with ONLY the file paths, one per line."
        )
        try:
            result = self._m.instruct(
                prompt,
                strategy=None,
                model_options=self._model_options(system_prompt=system_prompt),
            )
            # Parse plain text response: extract candidate paths from output
            candidate_set = set(candidates)
            raw_lines = (result.value or "").strip().splitlines()
            valid = []
            for line in raw_lines:
                line = line.strip().lstrip("0123456789.-) ")
                if line in candidate_set and line not in valid:
                    valid.append(line)
            if valid:
                print(f"  llm localized: {valid}", flush=True)
                return valid[:max_files]
        except Exception as e:
            print(f"  llm localization failed: {e}", flush=True)
        return candidates[:max_files]

    def generate_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        repo_root: str,
        n_samples: int = 1,
        test_cmds: list[str] | None = None,
    ) -> str:
        import asyncio
        import subprocess

        from mellea.agent.tools import make_agent_tools
        from mellea.stdlib.context import ChatContext
        from mellea.stdlib.frameworks.react import react

        tools = make_agent_tools(repo_root, test_cmds=test_cmds)

        # Build repo map for initial context.
        repo_map_text = ""
        localized_files: list[str] = []
        try:
            from pathlib import Path

            from mellea.agent.repomap import _SKIP_DIRS, _SOURCE_EXTS, build_repo_map
            from mellea.agent.repomap._bm25 import rank_bm25

            repo_map_text = build_repo_map(repo_root, problem_statement, max_tokens=2048)

            # Localization phase: BM25 rank -> LLM pick top files
            root = Path(repo_root)
            source_files: list[str] = []
            try:
                for p in root.rglob("*"):
                    if (
                        p.is_file()
                        and p.suffix.lower() in _SOURCE_EXTS
                        and not any(s in p.parts for s in _SKIP_DIRS)
                    ):
                        source_files.append(str(p))
            except OSError:
                pass
            if source_files:
                bm25_top = rank_bm25(source_files, problem_statement, repo_root, top_n=20)
                candidates = [str(Path(f).relative_to(repo_root)) for f in bm25_top]
                localized_files = self.localize_files(
                    candidates=candidates,
                    problem_statement=problem_statement,
                    max_files=5,
                )
        except Exception as e:
            print(f"  [repo_map/localize] failed: {e}", flush=True)

        repo_map_block = f"\n\nRepository structure:\n{repo_map_text}" if repo_map_text else ""
        hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""

        localized_block = ""
        if localized_files:
            file_list = ", ".join(localized_files)
            localized_block = (
                f"\n\nFiles most likely to need editing: {file_list}\n"
                "Use read_file to examine these files."
            )

        system_prompt = (
            "You are an expert software engineer fixing a bug in an "
            "open-source repository. You MUST edit existing source files "
            "to fix the bug. Do NOT create new files. Do NOT write test "
            "scripts. Only modify the existing code that contains the bug.\n\n"
            "Strategy:\n"
            "1. Read the issue and localized files carefully\n"
            "2. Identify the root cause\n"
            "3. Make the minimal edit to fix it\n"
            "4. Run tests to verify your fix works\n"
            "5. Call final_answer when done"
        )

        test_block = ""
        if test_cmds:
            test_block = (
                "\n\nYou have a run_tests tool. Use it after editing to verify "
                "your fix. Pass 'default' to run the task's test suite."
            )

        goal = (
            f"Fix this bug in {repo} by editing the existing source code.\n\n"
            f"Issue:\n{problem_statement.strip()}"
            f"{repo_map_block}{localized_block}{hints_block}{test_block}\n\n"
            "Only edit existing files. Do not create new files or test scripts."
        )

        budget = max(1, self.loop_budget)
        model_opts = self._model_options(system_prompt=system_prompt)

        timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))

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
