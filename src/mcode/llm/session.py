from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, Field

from mcode.agent.coding_agent import build_coding_agent
from mcode.agent.verification import (
    VerificationPolicy,
    build_submission_requirements,
    build_turn_requirements,
)
from mcode.llm.react_driver import SolveTraceCollector, SolveTracePlugin, run_react_loop
from mcode.llm.repo_state import get_git_diff, repo_snapshot, restore_repo_snapshot
from mcode.mellea_compat import apply_provider_compatibility_patches


class PatchSubmission(BaseModel):
    summary: str = Field(min_length=1)
    tests_ran: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SolveResult:
    patch: str = ""
    submission: PatchSubmission | None = None
    terminal_reason: str | None = None
    turns_to_first_edit: int | None = None
    turns_to_first_verification: int | None = None
    zero_edit: bool = True
    zero_verification: bool = True
    verification_succeeded: bool = False
    prompt_snapshot: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider: str | None = None
    response_model: str | None = None
    generation_latency_ms: int | None = None
    validation_passed_count: int | None = None
    validation_failed_count: int | None = None
    verification_evidence: list[dict[str, object]] | None = None
    diagnostic_events: list[dict[str, object]] | None = None
    last_model_output: dict[str, object] | None = None

    def as_metrics_dict(self) -> dict[str, object]:
        metrics: dict[str, object] = {
            "terminal_reason": self.terminal_reason,
            "turns_to_first_edit": self.turns_to_first_edit,
            "turns_to_first_verification": self.turns_to_first_verification,
            "zero_edit": self.zero_edit,
            "zero_verification": self.zero_verification,
            "verification_succeeded": self.verification_succeeded,
            "prompt_snapshot": self.prompt_snapshot,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "provider": self.provider,
            "response_model": self.response_model,
            "generation_latency_ms": self.generation_latency_ms,
            "validation_passed_count": self.validation_passed_count,
            "validation_failed_count": self.validation_failed_count,
        }
        if self.verification_evidence is not None:
            metrics["verification_evidence"] = self.verification_evidence
        if self.diagnostic_events is not None:
            metrics["diagnostic_events"] = self.diagnostic_events
        if self.last_model_output is not None:
            metrics["last_model_output"] = self.last_model_output
        return metrics


def hooks_available() -> bool:
    try:
        import cpex.framework.base  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_launch_endpoint(model_id: str) -> str | None:
    try:
        from mcode.launch import state as launch_state
    except Exception:
        return None
    try:
        snap = launch_state.load()
    except Exception:
        return None
    for server in snap.servers:
        if server.model == model_id and server.status == "healthy" and server.endpoint:
            return server.endpoint
    return None


class McodeSolverPowerup:
    async def solve_patch(
        self,
        *,
        repo_root: str,
        goal: str,
        tools: list,
        model_options: dict,
        loop_budget: int,
        timeout_s: int,
        verification_policy: VerificationPolicy,
        collector: SolveTraceCollector,
        sampling_strategy_name: str,
        sampling_budget: int,
        hooks_enabled: bool,
    ) -> SolveResult:
        use_requirement_sampling = sampling_strategy_name != "none"
        submission, terminal_reason = await run_react_loop(
            self,
            goal=goal,
            tools=tools,
            model_options=model_options,
            loop_budget=loop_budget,
            timeout_s=timeout_s,
            submission_format=PatchSubmission if use_requirement_sampling else None,
            collector=collector,
            turn_requirements=(
                lambda turn, budget, state: build_turn_requirements(
                    verification_policy=verification_policy,
                    enforce_run_tests=_should_enforce_run_tests(
                        turn=turn,
                        loop_budget=budget,
                        verification_policy=verification_policy,
                        verification_succeeded=state.verification_succeeded,
                    ),
                )
            )
            if use_requirement_sampling
            else (lambda *_args, **_kwargs: []),
            submission_requirements=(
                build_submission_requirements() if use_requirement_sampling else []
            ),
            strategy_for_requirements=lambda requirements: _strategy_for_requirements(
                backend=self.backend,
                requirements=requirements,
                strategy_name=sampling_strategy_name,
                sampling_budget=sampling_budget,
            ),
            hooks_enabled=hooks_enabled,
        )

        patch = get_git_diff(repo_root)
        raw_patch = patch
        collector.note_event("patch_stats", {"stage": "raw", **_patch_stats(raw_patch)})
        if (
            patch
            and _has_verification_tool(verification_policy)
            and not collector.verification_succeeded
        ):
            patch = ""
            terminal_reason = "unverified_diff_discarded"
        collector.note_event(
            "patch_stats",
            {
                "stage": "final",
                "discarded_unverified_diff": bool(raw_patch and not patch),
                **_patch_stats(patch),
            },
        )
        collector.note_event(
            "terminal",
            {
                "terminal_reason": terminal_reason,
                "verification_succeeded": collector.verification_succeeded,
                "turns_to_first_edit": collector.turns_to_first_edit,
                "turns_to_first_verification": collector.turns_to_first_verification,
            },
        )

        return SolveResult(
            patch=patch,
            submission=_coerce_submission(submission),
            terminal_reason=terminal_reason,
            turns_to_first_edit=collector.turns_to_first_edit,
            turns_to_first_verification=collector.turns_to_first_verification,
            zero_edit=collector.turns_to_first_edit is None,
            zero_verification=collector.turns_to_first_verification is None,
            verification_succeeded=collector.verification_succeeded,
            prompt_snapshot=collector.prompt_snapshot,
            prompt_tokens=_none_if_zero(collector.prompt_tokens),
            completion_tokens=_none_if_zero(collector.completion_tokens),
            total_tokens=_none_if_zero(collector.total_tokens),
            provider=collector.provider,
            response_model=collector.response_model,
            generation_latency_ms=_none_if_zero(collector.generation_latency_ms),
            validation_passed_count=collector.validation_passed_count,
            validation_failed_count=collector.validation_failed_count,
            verification_evidence=(
                list(collector.verification_evidence) if collector.verification_evidence else None
            ),
            diagnostic_events=(
                list(collector.diagnostic_events) if collector.diagnostic_enabled else None
            ),
            last_model_output=collector.last_model_output,
        )


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 15
    temperature: float | None = None
    seed: int | None = None
    sampling_strategy: str = "none"
    sampling_budget: int | None = None
    selection_attempts: int = 1
    diagnostic_traces: bool = False
    live_event_sink: Callable[[str, Mapping[str, object]], None] | None = field(
        default=None, repr=False
    )
    _m: object | None = field(default=None, repr=False)
    _last_result: SolveResult | None = field(default=None, repr=False)

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

    def _start_session(self, *, plugins: list[object] | None = None):
        import mellea
        from mellea.stdlib.context import ChatContext

        _ensure_powerup_registered()
        apply_provider_compatibility_patches()
        return mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
            ctx=ChatContext(),
            plugins=plugins,
            **self._backend_kwargs(),
        )

    def check_available(self) -> None:
        try:
            with self._start_session():
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
        with self._start_session() as session:
            self._m = session
            try:
                yield self
            finally:
                self._m = None

    @property
    def solve_result(self) -> SolveResult | None:
        return self._last_result

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
        editable_paths: Sequence[str] | None = None,
    ) -> str:
        return self.solve(
            repo=repo,
            problem_statement=problem_statement,
            hints_text=hints_text,
            repo_root=repo_root,
            n_samples=n_samples,
            test_cmds=test_cmds,
            test_fn=test_fn,
            command_fn=command_fn,
            visible_repo_root=visible_repo_root,
            editable_paths=editable_paths,
        ).patch

    def solve(
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
        editable_paths: Sequence[str] | None = None,
    ) -> SolveResult:
        self._last_result = None
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
            editable_paths=editable_paths,
        )

        attempts: list[SolveResult] = []
        if self.selection_attempts > 1:
            outer_attempts = self.selection_attempts
            select_best_attempt = True
        else:
            outer_attempts = max(1, n_samples) if self.sampling_strategy == "none" else 1
            select_best_attempt = False
        sampling_budget = self.sampling_budget or max(1, n_samples)
        enable_hooks = hooks_available()
        if self.diagnostic_traces and not enable_hooks:
            raise RuntimeError(
                "Diagnostic traces require Mellea hooks. Install the observability extra with "
                "`uv sync --extra observability` and retry."
            )
        with repo_snapshot(repo_root, enabled=outer_attempts > 1) as snapshot_dir:
            for index in range(outer_attempts):
                if index and snapshot_dir is not None:
                    restore_repo_snapshot(repo_root, snapshot_dir)
                collector = SolveTraceCollector(
                    diagnostic_enabled=self.diagnostic_traces,
                    live_event_sink=self.live_event_sink,
                )
                runtime_plugins = [SolveTracePlugin(collector)] if enable_hooks else None

                async def _solve_once() -> SolveResult:
                    loop = asyncio.get_running_loop()
                    loop.set_exception_handler(_ignore_async_client_close_noise)
                    try:
                        return await session.solve_patch(
                            repo_root=repo_root,
                            goal=agent.goal,
                            tools=agent.tools,
                            model_options=agent.model_options,
                            loop_budget=agent.loop_budget,
                            timeout_s=agent.timeout_s,
                            verification_policy=agent.verification_policy,
                            collector=collector,
                            sampling_strategy_name=self.sampling_strategy,
                            sampling_budget=sampling_budget,
                            hooks_enabled=enable_hooks,
                        )
                    finally:
                        await _close_backend_async_clients(getattr(session, "backend", None))

                with self._start_session(plugins=runtime_plugins) as session:
                    result = asyncio.run(_solve_once())
                attempts.append(result)
                if not select_best_attempt and result.patch and result.verification_succeeded:
                    self._last_result = result
                    return result

            if select_best_attempt and snapshot_dir is not None:
                restore_repo_snapshot(repo_root, snapshot_dir)

        if attempts:
            selected = _select_solve_result(attempts)
            self._last_result = selected
            return selected

        self._last_result = SolveResult()
        return self._last_result


def _select_solve_result(attempts: list[SolveResult]) -> SolveResult:
    return max(enumerate(attempts), key=lambda item: (_solve_result_score(item[1]), -item[0]))[1]


def _solve_result_score(result: SolveResult) -> tuple[int, int, int, int, int, int]:
    return (
        1 if result.patch else 0,
        1 if result.verification_succeeded else 0,
        1 if result.terminal_reason == "submitted" else 0,
        1 if result.submission is not None else 0,
        0 if result.zero_edit else 1,
        0 if result.zero_verification else 1,
    )


def _ensure_powerup_registered() -> None:
    from mellea.stdlib.session import MelleaSession

    if getattr(MelleaSession, "_mcode_solver_powerup", False):
        return
    MelleaSession.powerup(McodeSolverPowerup)
    setattr(MelleaSession, "_mcode_solver_powerup", True)


def _should_enforce_run_tests(
    *,
    turn: int,
    loop_budget: int,
    verification_policy: VerificationPolicy,
    verification_succeeded: bool,
) -> bool:
    return (
        _has_verification_tool(verification_policy)
        and not verification_succeeded
        and turn >= max(2, loop_budget - 1)
    )


def _has_verification_tool(verification_policy: VerificationPolicy) -> bool:
    return bool(
        verification_policy.test_cmds
        or verification_policy.test_fn is not None
        or verification_policy.command_fn is not None
    )


def _strategy_for_requirements(
    *,
    backend,
    requirements: list[object],
    strategy_name: str,
    sampling_budget: int,
):
    if not requirements:
        return None
    if strategy_name == "none":
        return None
    return _build_sampling_strategy(
        backend=backend,
        strategy_name=strategy_name,
        sampling_budget=sampling_budget,
    )


def _build_sampling_strategy(*, backend, strategy_name: str, sampling_budget: int):
    del backend
    if strategy_name == "multiturn":
        return MultiTurnStrategy(loop_budget=max(1, sampling_budget))
    raise ValueError(f"unknown sampling strategy: {strategy_name}")


def _coerce_submission(value: Any) -> PatchSubmission | None:
    if value is None:
        return None
    if isinstance(value, PatchSubmission):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"summary": text, "tests_ran": []}
    return PatchSubmission.model_validate(value)


def _patch_stats(patch: str) -> dict[str, object]:
    touched_files: list[str] = []
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                touched_files.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return {
        "touched_files": sorted(set(touched_files)),
        "added_lines": added,
        "deleted_lines": deleted,
        "byte_count": len(patch.encode("utf-8", errors="ignore")),
        "sha256": hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest(),
    }


def _ignore_async_client_close_noise(
    loop: asyncio.AbstractEventLoop, context: dict[str, object]
) -> None:
    exc = context.get("exception")
    future = context.get("future") or context.get("task")
    if (
        isinstance(exc, RuntimeError)
        and str(exc) == "Event loop is closed"
        and "AsyncClient.aclose" in repr(future)
    ):
        return
    loop.default_exception_handler(context)


async def _close_backend_async_clients(backend: object | None) -> None:
    cache = getattr(getattr(backend, "_client_cache", None), "cache", None)
    if not cache:
        return
    for client in list(cache.values()):
        aclose = getattr(client, "aclose", None)
        if not callable(aclose):
            continue
        try:
            await aclose()
        except Exception:
            pass


def _none_if_zero(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value
