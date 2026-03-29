from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from mcode.agent.coding_agent import build_coding_agent
from mcode.agent.verification import (
    build_budget_warning,
    build_submit_block_message,
    verification_state_from_event_log,
)


def _seed_react_context_from_runtime(*, condensed_state):
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.context import ChatContext

    context = ChatContext()
    if condensed_state is None:
        return context

    show_reminder = bool(getattr(condensed_state, "show_reminder", False))
    omitted_messages = int(getattr(condensed_state, "omitted_messages", 0) or 0)
    working_memory = getattr(condensed_state, "working_memory", None)
    recent_messages = getattr(condensed_state, "recent_messages", ())

    if show_reminder and working_memory is not None:
        reminder = working_memory.as_message(omitted_messages=omitted_messages)
        context = context.add(Message(role=reminder["role"], content=str(reminder["content"])))

    for message in recent_messages:
        context = context.add(Message(role=message["role"], content=str(message["content"])))

    return context


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
        # Disable streaming to avoid tool call argument loss in vLLM.
        opts[ModelOption.STREAM] = False
        return opts

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

    def generate_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        repo_root: str,
        n_samples: int = 1,
        test_cmds: object | None = None,
        test_fn: object | None = None,
        command_fn: object | None = None,
        visible_repo_root: str | None = None,
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

        from mellea.stdlib.frameworks.react import react

        agent = build_coding_agent(
            session=self,
            repo=repo,
            problem_statement=problem_statement,
            hints_text=hints_text,
            repo_root=repo_root,
            visible_repo_root=visible_repo_root,
            test_cmds=test_cmds,
            test_fn=test_fn,
            command_fn=command_fn,
        )

        budget = agent.loop_budget
        timeout_s = agent.timeout_s
        use_text_tools = agent.use_text_tools
        tools = agent.tools
        system_prompt = agent.system_prompt
        goal = agent.goal
        model_opts = agent.model_options
        use_budget_warning = agent.use_budget_warning
        use_mid_nudge = agent.use_mid_nudge
        event_log = getattr(agent, "event_log", None)
        condensed_state = getattr(agent, "condensed_state", None)
        condensation = getattr(agent, "condensation", None)
        max_retries_per_turn = int(getattr(agent, "max_retries_per_turn", 0))
        verification_policy = getattr(agent, "verification_policy", None)
        verification_cmds = list(
            getattr(verification_policy, "test_cmds", getattr(agent, "verification_cmds", []))
        )
        has_run_tests_tool = any(getattr(tool, "name", None) == "run_tests" for tool in tools)
        require_default_verification = bool(verification_cmds)

        def _repo_has_changes() -> bool:
            result = subprocess.run(
                ["git", "diff", "--quiet", "--exit-code"],
                cwd=repo_root,
                capture_output=True,
            )
            return result.returncode == 1

        def _messages_include_tool(messages, tool_name: str) -> bool:
            marker = f"[{tool_name}]"
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str) and marker in content:
                    return True
            return False

        def _verification_state():
            return verification_state_from_event_log(event_log)

        def _submit_block_message() -> str | None:
            return build_submit_block_message(
                has_changes=_repo_has_changes(),
                has_run_tests_tool=has_run_tests_tool,
                verification_state=_verification_state(),
                require_default_verification=require_default_verification,
            )

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
                warning = build_budget_warning(
                    has_changes=_repo_has_changes(),
                    has_run_tests_tool=has_run_tests_tool,
                    used_run_tests=False,
                ).replace("`final_answer`", "final_answer")
                ctx = ctx.add(
                    Message(
                        role="user",
                        content=warning,
                    )
                )
            return ctx

        if use_text_tools:

            def _text_on_turn(turn, total, msgs):
                if (
                    total >= 12
                    and turn >= 8
                    and turn % 8 == 0
                    and not _repo_has_changes()
                    and not _messages_include_tool(msgs, "edit")
                    and (
                        _messages_include_tool(msgs, "read_file")
                        or _messages_include_tool(msgs, "search_code")
                    )
                ):
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "You have spent several turns reading/searching without "
                                "editing. Stop exploring. Pick the single most likely file, "
                                "make one concrete edit now, then verify it with run_tests."
                            ),
                        }
                    )
                if use_mid_nudge and total > 5 and turn == total // 2:
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "You are halfway through your budget. "
                                "If you have not made any edits yet, "
                                "start editing NOW."
                            ),
                        }
                    )
                if use_budget_warning and total > 3 and turn == total - 2:
                    warning = build_budget_warning(
                        has_changes=_repo_has_changes(),
                        has_run_tests_tool=has_run_tests_tool,
                        used_run_tests=_messages_include_tool(msgs, "run_tests"),
                    )
                    msgs.append(
                        {
                            "role": "user",
                            "content": warning,
                        }
                    )
                return msgs

            async def _one_attempt():
                from mellea.agent.text_react import text_react

                react_kwargs = {
                    "goal": goal,
                    "backend": self._m.backend if self._m else None,
                    "tools": tools,
                    "system_prompt": system_prompt,
                    "model_options": model_opts,
                    "loop_budget": budget,
                    "on_turn": _text_on_turn,
                    "final_answer_guard": (
                        lambda answer, *, messages, event_log: _submit_block_message()
                    ),
                }
                if condensed_state is not None:
                    react_kwargs["condensed_state"] = condensed_state
                if condensation is not None:
                    react_kwargs["condensation"] = condensation
                if event_log is not None:
                    react_kwargs["event_log"] = event_log
                if max_retries_per_turn > 0:
                    react_kwargs["max_retries_per_turn"] = max_retries_per_turn

                try:
                    answer, done = await asyncio.wait_for(
                        text_react(**react_kwargs),
                        timeout=timeout_s,
                    )
                    if done:
                        print(
                            f"  [react] final_answer: {answer[:120]}",
                            flush=True,
                        )
                    else:
                        print(
                            "  [react] budget exhausted without final_answer",
                            flush=True,
                        )
                except TimeoutError:
                    print(f"  [react] timed out after {timeout_s}s", flush=True)
                block_message = _submit_block_message()
                diff = _get_diff(repo_root)
                if diff and block_message is not None:
                    print(
                        f"  [react] discarding unverified diff: {block_message}",
                        flush=True,
                    )
                    return ""
                return diff

        else:

            async def _one_attempt():
                try:
                    result, _ = await asyncio.wait_for(
                        react(
                            goal=goal,
                            context=_seed_react_context_from_runtime(
                                condensed_state=condensed_state
                            ),
                            backend=self._m.backend,
                            tools=tools,
                            loop_budget=budget,
                            model_options=model_opts,
                            on_turn=_on_turn if (use_budget_warning or use_mid_nudge) else None,
                        ),
                        timeout=timeout_s,
                    )
                    print(
                        f"  [react] final_answer: {result.value[:120]}",
                        flush=True,
                    )
                except TimeoutError:
                    print(f"  [react] timed out after {timeout_s}s", flush=True)
                except RuntimeError:
                    print(
                        "  [react] budget exhausted without final_answer",
                        flush=True,
                    )
                block_message = _submit_block_message()
                diff = _get_diff(repo_root)
                if diff and block_message is not None:
                    print(
                        f"  [react] discarding unverified diff: {block_message}",
                        flush=True,
                    )
                    return ""
                return diff

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
