from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from mcode.agent.coding_agent import build_coding_agent
from mcode.agent.verification import (
    build_phase_guidance,
    build_submit_block_message,
    build_tool_gate_message,
    tighten_available_tools,
    tool_phase_state_from_event_log,
    verification_state_from_event_log,
)


@dataclass(frozen=True)
class PatchGenerationMetrics:
    turns_to_first_edit: int | None = None
    turns_to_first_verification: int | None = None
    zero_edit: bool = True
    zero_verification: bool = True
    verification_succeeded: bool = False
    malformed_tool_call_recoveries: int = 0
    blocked_verification_commands: int = 0
    blocked_submissions: int = 0
    terminal_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "turns_to_first_edit": self.turns_to_first_edit,
            "turns_to_first_verification": self.turns_to_first_verification,
            "zero_edit": self.zero_edit,
            "zero_verification": self.zero_verification,
            "verification_succeeded": self.verification_succeeded,
            "malformed_tool_call_recoveries": self.malformed_tool_call_recoveries,
            "blocked_verification_commands": self.blocked_verification_commands,
            "blocked_submissions": self.blocked_submissions,
            "terminal_reason": self.terminal_reason,
        }


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 3
    temperature: float | None = None
    seed: int | None = None
    _m: object | None = field(default=None, repr=False)
    _last_patch_metrics: PatchGenerationMetrics | None = field(default=None, repr=False)

    def _backend_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.backend_name == "ollama":
            base_url = os.environ.get("OLLAMA_HOST")
            if base_url:
                kwargs["base_url"] = base_url
        elif self.backend_name == "openai":
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
        opts[ModelOption.STREAM] = False
        return opts

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
        import asyncio
        import subprocess
        from collections import Counter

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

        budget = agent.loop_budget
        timeout_s = agent.timeout_s
        tools = agent.tools
        tool_names = [getattr(tool, "name", "") for tool in tools]
        system_prompt = agent.system_prompt
        goal = agent.goal
        model_opts = agent.model_options
        event_log = getattr(agent, "event_log", None)
        condensed_state = getattr(agent, "condensed_state", None)
        condensation = getattr(agent, "condensation", None)
        max_retries_per_turn = int(getattr(agent, "max_retries_per_turn", 0))
        verification_policy = getattr(agent, "verification_policy", None)
        verification_cmds = list(
            getattr(verification_policy, "test_cmds", getattr(agent, "verification_cmds", []))
        )
        capability_contract = getattr(agent, "capability_contract", None)
        has_run_tests_tool = any(name == "run_tests" for name in tool_names)
        require_default_verification = bool(verification_cmds)

        def _repo_has_changes() -> bool:
            result = subprocess.run(
                ["git", "diff", "--quiet", "--exit-code"],
                cwd=repo_root,
                capture_output=True,
            )
            return result.returncode == 1

        def _verification_state():
            return verification_state_from_event_log(event_log)

        def _phase_state(turn: int):
            return tool_phase_state_from_event_log(
                event_log,
                turn=turn,
                budget=budget,
            )

        def _submit_block_message() -> str | None:
            return build_submit_block_message(
                has_changes=_repo_has_changes(),
                has_run_tests_tool=has_run_tests_tool,
                verification_state=_verification_state(),
                require_default_verification=require_default_verification,
            )

        def _metrics_from_attempt(
            *,
            turns_to_first_edit: int | None,
            turns_to_first_verification: int | None,
            malformed_tool_call_recoveries: int,
            blocked_verification_commands: int,
            blocked_submissions: int,
            verification_succeeded: bool,
            terminal_reason: str,
        ) -> PatchGenerationMetrics:
            return PatchGenerationMetrics(
                turns_to_first_edit=turns_to_first_edit,
                turns_to_first_verification=turns_to_first_verification,
                zero_edit=turns_to_first_edit is None,
                zero_verification=turns_to_first_verification is None,
                verification_succeeded=verification_succeeded,
                malformed_tool_call_recoveries=malformed_tool_call_recoveries,
                blocked_verification_commands=blocked_verification_commands,
                blocked_submissions=blocked_submissions,
                terminal_reason=terminal_reason,
            )

        async def _one_attempt() -> tuple[str, PatchGenerationMetrics]:
            from mellea.agent import text_react as text_react_module

            @dataclass
            class _AttemptState:
                current_turn: int = 0
                turns_to_first_edit: int | None = None
                turns_to_first_verification: int | None = None
                malformed_tool_call_recoveries: int = 0
                blocked_verification_commands: int = 0
                blocked_submissions: int = 0

            attempt_state = _AttemptState()

            def _emit_summary(kind: str, metadata: dict[str, object] | None = None) -> None:
                if event_log is None:
                    return
                try:
                    from mellea.agent.runtime.events import SummaryEvent
                except Exception:
                    return
                event_log.emit(
                    SummaryEvent(
                        message=kind.replace("_", " "),
                        metadata={"kind": kind, **(metadata or {})},
                    )
                )

            def _text_on_turn(turn: int, total: int, messages: list[dict]) -> list[dict]:
                attempt_state.current_turn = turn
                guidance = build_phase_guidance(
                    has_changes=_repo_has_changes(),
                    has_run_tests_tool=has_run_tests_tool,
                    verification_state=_verification_state(),
                    phase_state=_phase_state(turn),
                    require_default_verification=require_default_verification,
                )
                if guidance and (not messages or messages[-1].get("content") != guidance):
                    messages.append({"role": "user", "content": guidance})
                return messages

            def _tool_gate(name: str, args: dict, *, messages, event_log):
                del args, messages
                turn_index = max(1, attempt_state.current_turn or 1)
                if name == "edit" and attempt_state.turns_to_first_edit is None:
                    attempt_state.turns_to_first_edit = turn_index
                if name == "run_tests" and attempt_state.turns_to_first_verification is None:
                    attempt_state.turns_to_first_verification = turn_index

                capability_route = None
                if capability_contract is not None:
                    route_for_tool = getattr(capability_contract, "route_for_tool", None)
                    if callable(route_for_tool):
                        capability_route = route_for_tool(name)
                try:
                    from mellea.agent.strategy import get_available_tools
                except ImportError:
                    block_message = build_tool_gate_message(
                        name,
                        available_tools=tool_names,
                        has_changes=_repo_has_changes(),
                        has_run_tests_tool=has_run_tests_tool,
                        verification_state=_verification_state(),
                        require_default_verification=require_default_verification,
                        requested_family=(
                            None
                            if capability_route is None
                            else getattr(capability_route, "requested_family", None)
                        ),
                        route_mode=(
                            None
                            if capability_route is None
                            else getattr(capability_route, "mode", None)
                        ),
                    )
                    if block_message and name == "final_answer":
                        attempt_state.blocked_submissions += 1
                        _emit_summary("submission_blocked", {"turn": turn_index})
                    return block_message

                invocation_count = 0
                if event_log is not None:
                    to_dicts = getattr(event_log, "to_dicts", None)
                    if callable(to_dicts):
                        invocation_count = sum(
                            1
                            for event in to_dicts()
                            if isinstance(event, dict) and event.get("kind") == "tool_result"
                        )
                phase_state = tool_phase_state_from_event_log(
                    event_log,
                    turn=max(1, invocation_count + 1),
                    budget=budget,
                )
                available_tools = tighten_available_tools(
                    get_available_tools(
                        tool_names,
                        turn=phase_state.turn,
                        budget=budget,
                        state=phase_state,
                    ),
                    phase_state=phase_state,
                    has_changes=_repo_has_changes(),
                    verification_state=_verification_state(),
                    has_run_tests_tool=has_run_tests_tool,
                )
                block_message = build_tool_gate_message(
                    name,
                    available_tools=available_tools,
                    has_changes=_repo_has_changes(),
                    has_run_tests_tool=has_run_tests_tool,
                    verification_state=_verification_state(),
                    require_default_verification=require_default_verification,
                    requested_family=(
                        None
                        if capability_route is None
                        else getattr(capability_route, "requested_family", None)
                    ),
                    route_mode=(
                        None
                        if capability_route is None
                        else getattr(capability_route, "mode", None)
                    ),
                )
                if block_message and name == "final_answer":
                    attempt_state.blocked_submissions += 1
                    _emit_summary("submission_blocked", {"turn": turn_index})
                return block_message

            react_kwargs = {
                "goal": goal,
                "backend": self._m.backend if self._m else None,
                "tools": tools,
                "system_prompt": system_prompt,
                "model_options": model_opts,
                "loop_budget": budget,
                "on_turn": _text_on_turn,
                "tool_gate": _tool_gate,
            }
            if condensed_state is not None:
                react_kwargs["condensed_state"] = condensed_state
            if condensation is not None:
                react_kwargs["condensation"] = condensation
            if event_log is not None:
                react_kwargs["event_log"] = event_log
            if max_retries_per_turn > 0:
                react_kwargs["max_retries_per_turn"] = max_retries_per_turn

            timed_out = False
            parse_with_repair = getattr(text_react_module, "_parse_tool_calls_with_repair", None)
            restore_parse = callable(parse_with_repair)
            if restore_parse:

                def _counting_parse(text: str):
                    tool_calls, used_repair = parse_with_repair(text)
                    if used_repair or not tool_calls:
                        attempt_state.malformed_tool_call_recoveries += 1
                        _emit_summary(
                            "malformed_tool_call",
                            {
                                "repaired": used_repair,
                                "re_prompted": not tool_calls,
                            },
                        )
                    return tool_calls, used_repair

                text_react_module._parse_tool_calls_with_repair = _counting_parse

            try:
                try:
                    answer, done = await asyncio.wait_for(
                        text_react_module.text_react(**react_kwargs),
                        timeout=timeout_s,
                    )
                    if done:
                        print(f"  [react] final_answer: {answer[:120]}", flush=True)
                    else:
                        print(
                            "  [react] budget exhausted without final_answer",
                            flush=True,
                        )
                except TimeoutError:
                    timed_out = True
                    done = False
                    print(f"  [react] timed out after {timeout_s}s", flush=True)
            finally:
                if restore_parse:
                    text_react_module._parse_tool_calls_with_repair = parse_with_repair

            verification_state = _verification_state()
            attempt_state.blocked_verification_commands = (
                verification_state.blocked_verification_commands
            )
            block_message = _submit_block_message()
            diff = _get_diff(repo_root)
            terminal_reason = "submitted"
            if not done:
                terminal_reason = "infra_failure" if timed_out else "budget_exhausted"
            if diff and block_message is not None:
                print(
                    f"  [react] discarding unverified diff: {block_message}",
                    flush=True,
                )
                terminal_reason = "unverified_diff_discarded"
                return "", _metrics_from_attempt(
                    turns_to_first_edit=attempt_state.turns_to_first_edit,
                    turns_to_first_verification=attempt_state.turns_to_first_verification,
                    malformed_tool_call_recoveries=attempt_state.malformed_tool_call_recoveries,
                    blocked_verification_commands=attempt_state.blocked_verification_commands,
                    blocked_submissions=attempt_state.blocked_submissions,
                    verification_succeeded=verification_state.successful_run_tests,
                    terminal_reason=terminal_reason,
                )
            return diff, _metrics_from_attempt(
                turns_to_first_edit=attempt_state.turns_to_first_edit,
                turns_to_first_verification=attempt_state.turns_to_first_verification,
                malformed_tool_call_recoveries=attempt_state.malformed_tool_call_recoveries,
                blocked_verification_commands=attempt_state.blocked_verification_commands,
                blocked_submissions=attempt_state.blocked_submissions,
                verification_succeeded=verification_state.successful_run_tests,
                terminal_reason=terminal_reason,
            )

        def _reset_repo() -> None:
            subprocess.run(["git", "checkout", "."], cwd=repo_root, capture_output=True)

        if n_samples <= 1:
            diff, metrics = asyncio.run(_one_attempt())
            self._last_patch_metrics = metrics
            return diff

        attempts: list[tuple[str, PatchGenerationMetrics]] = []
        for index in range(n_samples):
            print(f"\n  [sample {index + 1}/{n_samples}]", flush=True)
            diff, metrics = asyncio.run(_one_attempt())
            attempts.append((diff, metrics))
            if index < n_samples - 1:
                _reset_repo()

        non_empty = [(diff, metrics) for diff, metrics in attempts if diff and diff.strip()]
        if not non_empty:
            if attempts:
                self._last_patch_metrics = attempts[-1][1]
            return ""

        counts = Counter(diff for diff, _ in non_empty)
        best_diff, best_count = counts.most_common(1)[0]
        best_metrics = next(metrics for diff, metrics in non_empty if diff == best_diff)
        print(
            f"  [voting] {len(non_empty)}/{n_samples} produced patches, "
            f"best has {best_count} votes",
            flush=True,
        )

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
            non_empty.sort(key=lambda item: len(item[0]), reverse=True)
            self._last_patch_metrics = non_empty[0][1]
            return non_empty[0][0]
        self._last_patch_metrics = best_metrics
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
