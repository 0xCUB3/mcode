from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field

from mcode.agent.coding_agent import build_coding_agent


@dataclass(frozen=True)
class PatchGenerationMetrics:
    turns_to_first_edit: int | None = None
    turns_to_first_verification: int | None = None
    zero_edit: bool = True
    zero_verification: bool = True
    verification_succeeded: bool = False
    terminal_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "turns_to_first_edit": self.turns_to_first_edit,
            "turns_to_first_verification": self.turns_to_first_verification,
            "zero_edit": self.zero_edit,
            "zero_verification": self.zero_verification,
            "verification_succeeded": self.verification_succeeded,
            "terminal_reason": self.terminal_reason,
        }


def _resolve_launch_endpoint(model_id: str) -> str | None:
    try:
        from mcode.launch import state as launch_state
    except Exception:
        return None
    try:
        snap = launch_state.load()
    except Exception:
        return None
    for s in snap.servers:
        if s.model == model_id and s.status == "healthy" and s.endpoint:
            return s.endpoint
    return None


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 15
    temperature: float | None = None
    seed: int | None = None
    _m: object | None = field(default=None, repr=False)
    _last_patch_metrics: PatchGenerationMetrics | None = field(default=None, repr=False)

    DEFAULT_MAX_NEW_TOKENS: int = 1024

    def _backend_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.backend_name == "ollama":
            base_url = os.environ.get("OLLAMA_HOST")
            if base_url:
                kwargs["base_url"] = base_url
        elif self.backend_name == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not base_url:
                base_url = _resolve_launch_endpoint(self.model_id)
                if base_url and not api_key:
                    api_key = "dummy"
                if base_url:
                    os.environ.setdefault("OPENAI_BASE_URL", base_url)
                    os.environ.setdefault("OPENAI_API_KEY", api_key or "dummy")
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
        else:
            opts[ModelOption.MAX_NEW_TOKENS] = self.DEFAULT_MAX_NEW_TOKENS
        ctx_raw = os.environ.get("MCODE_CONTEXT_WINDOW")
        if ctx_raw:
            opts[ModelOption.CONTEXT_WINDOW] = int(ctx_raw)
        elif self.backend_name == "ollama":
            opts[ModelOption.CONTEXT_WINDOW] = 16384
        opts[ModelOption.STREAM] = False
        return opts

    def check_available(self) -> None:
        try:
            import mellea
        except Exception as e:
            raise RuntimeError(
                "mellea is required for LLM interaction; install dependencies with `uv sync`"
            ) from e

        try:
            with mellea.start_session(
                backend_name=self.backend_name,
                model_id=self.model_id,
                **self._backend_kwargs(),
            ):
                return
        except Exception as e:
            raise RuntimeError(
                f"Could not start a Mellea session (backend={self.backend_name!r}, "
                f"model_id={self.model_id!r}). "
                "Ensure the backend is running and accessible and retry."
            ) from e

    @contextmanager
    def open(self):
        if self._m is not None:
            yield self
            return

        try:
            import mellea
        except Exception as e:
            raise RuntimeError(
                "mellea is required for LLM interaction; install dependencies with `uv sync`"
            ) from e

        with mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
            **self._backend_kwargs(),
        ) as m:
            self._m = m
            try:
                yield self
            finally:
                self._m = None

    @property
    def last_patch_metrics(self) -> dict[str, object] | None:
        if self._last_patch_metrics is None:
            return None
        return self._last_patch_metrics.as_dict()

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
        self._last_patch_metrics = None
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

        attempts: list[tuple[str, PatchGenerationMetrics]] = []
        for index in range(max(1, n_samples)):
            if index:
                _reset_repo(repo_root)
            diff, metrics = asyncio.run(
                _run_attempt(
                    session=self,
                    repo_root=repo_root,
                    goal=agent.goal,
                    tools=agent.tools,
                    model_options=agent.model_options,
                    loop_budget=agent.loop_budget,
                    timeout_s=agent.timeout_s,
                )
            )
            attempts.append((diff, metrics))
            if diff and metrics.verification_succeeded:
                self._last_patch_metrics = metrics
                return diff

        for diff, metrics in attempts:
            if diff:
                self._last_patch_metrics = metrics
                return diff

        if attempts:
            self._last_patch_metrics = attempts[-1][1]
        return ""


@dataclass
class _AttemptTracker:
    current_turn: int = 0
    turns_to_first_edit: int | None = None
    turns_to_first_verification: int | None = None
    verification_succeeded: bool = False

    def note_turn(self, turn: int) -> None:
        self.current_turn = turn

    def note_tool_call(self, tool_name: str) -> None:
        turn = max(1, self.current_turn or 1)
        if tool_name == "edit" and self.turns_to_first_edit is None:
            self.turns_to_first_edit = turn
        if tool_name == "run_tests" and self.turns_to_first_verification is None:
            self.turns_to_first_verification = turn

    def note_tool_result(self, tool_name: str, result: object) -> None:
        if tool_name == "run_tests" and _run_tests_succeeded(str(result)):
            self.verification_succeeded = True

    def metrics(self, *, terminal_reason: str) -> PatchGenerationMetrics:
        return PatchGenerationMetrics(
            turns_to_first_edit=self.turns_to_first_edit,
            turns_to_first_verification=self.turns_to_first_verification,
            zero_edit=self.turns_to_first_edit is None,
            zero_verification=self.turns_to_first_verification is None,
            verification_succeeded=self.verification_succeeded,
            terminal_reason=terminal_reason,
        )


async def _run_attempt(
    *,
    session: LLMSession,
    repo_root: str,
    goal: str,
    tools: list,
    model_options: dict,
    loop_budget: int,
    timeout_s: int,
) -> tuple[str, PatchGenerationMetrics]:
    import inspect

    from mellea.backends.tools import MelleaTool
    from mellea.stdlib.context import ChatContext
    from mellea.stdlib.frameworks.react import react

    tracker = _AttemptTracker()
    instrumented_tools: list[MelleaTool] = []
    for tool in tools:
        tool_name = getattr(tool, "name", "")

        def _make_call(current_tool):
            def _call(*args, **kwargs):
                tracker.note_tool_call(current_tool.name)
                result = current_tool.run(*args, **kwargs)
                tracker.note_tool_result(current_tool.name, result)
                return result

            return _call

        instrumented_tools.append(
            MelleaTool(
                name=tool_name,
                tool_call=_make_call(tool),
                as_json_tool=tool.as_json_tool,
            )
        )

    completed = False
    try:
        react_kwargs = {
            "goal": goal,
            "context": ChatContext(),
            "backend": session._m.backend,
            "tools": instrumented_tools,
            "loop_budget": loop_budget,
            "model_options": model_options,
        }
        if "on_turn" in inspect.signature(react).parameters:
            react_kwargs["on_turn"] = lambda turn, budget, ctx: _on_turn(tracker, turn, budget, ctx)
        await asyncio.wait_for(
            react(**react_kwargs),
            timeout=timeout_s,
        )
        completed = True
    except TimeoutError:
        pass
    except RuntimeError as e:
        if "could not complete react loop" not in str(e):
            raise

    diff = _get_diff(repo_root)
    if diff and any(getattr(tool, "name", "") == "run_tests" for tool in tools):
        if not tracker.verification_succeeded:
            return "", tracker.metrics(terminal_reason="unverified_diff_discarded")

    terminal_reason = "submitted"
    if not completed:
        terminal_reason = "budget_exhausted"
    return diff, tracker.metrics(terminal_reason=terminal_reason)


def _on_turn(tracker: _AttemptTracker, turn: int, budget: int, context):
    del budget
    tracker.note_turn(turn)
    return context


def _run_tests_succeeded(output: str) -> bool:
    return (
        "PASSED" in output
        and "FAILED" not in output
        and "TIMEOUT" not in output
        and "ERROR" not in output
        and "BLOCKED" not in output
    )


def _reset_repo(repo_root: str) -> None:
    subprocess.run(["git", "checkout", "."], cwd=repo_root, capture_output=True)


def _get_diff(repo_root: str) -> str:
    git_dir = os.path.join(repo_root, ".git")
    if not os.path.exists(git_dir):
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
