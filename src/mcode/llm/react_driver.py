from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

from mcode.mellea_compat import acall_tools


@dataclass
class SolveTraceCollector:
    current_turn: int = 0
    turns_to_first_edit: int | None = None
    turns_to_first_verification: int | None = None
    verification_succeeded: bool = False
    prompt_snapshot: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    provider: str | None = None
    response_model: str | None = None
    generation_latency_ms: int = 0
    validation_passed_count: int | None = None
    validation_failed_count: int | None = None

    def note_turn(self, turn: int) -> None:
        self.current_turn = turn

    def note_generation(
        self,
        *,
        prompt: object,
        usage: dict[str, Any] | None,
        provider: object | None,
        response_model: object | None,
        latency_ms: int,
    ) -> None:
        self.prompt_snapshot = _serialize_prompt(prompt)
        totals = _normalize_usage(usage)
        if totals is not None:
            self.prompt_tokens += totals["prompt_tokens"]
            self.completion_tokens += totals["completion_tokens"]
            self.total_tokens += totals["total_tokens"]
        normalized_provider = _normalize_optional_str(provider)
        normalized_model = _normalize_optional_str(response_model)
        if normalized_provider is not None:
            self.provider = normalized_provider
        if normalized_model is not None:
            self.response_model = normalized_model
        self.generation_latency_ms += latency_ms

    def note_tool(self, *, tool_name: str, output: object, success: bool) -> None:
        turn = max(1, self.current_turn or 1)
        if tool_name == "edit" and self.turns_to_first_edit is None:
            self.turns_to_first_edit = turn
        if tool_name == "run_tests" and self.turns_to_first_verification is None:
            self.turns_to_first_verification = turn
        if tool_name == "run_tests" and success and _run_tests_succeeded(str(output)):
            self.verification_succeeded = True

    def note_validation(self, *, passed_count: int, failed_count: int) -> None:
        self.validation_passed_count = passed_count
        self.validation_failed_count = failed_count


class SolveTracePlugin(Plugin, name="mcode-solve-trace", priority=20):
    def __init__(self, collector: SolveTraceCollector):
        self.collector = collector

    @hook("generation_post_call", mode=PluginMode.SEQUENTIAL)
    async def generation_post_call(self, payload, context) -> None:
        del context
        model_output = getattr(payload, "model_output", None)
        usage = getattr(model_output, "usage", None)
        provider = getattr(model_output, "provider", None)
        response_model = getattr(model_output, "model", None)
        self.collector.note_generation(
            prompt=getattr(payload, "prompt", None),
            usage=usage if isinstance(usage, dict) else None,
            provider=provider,
            response_model=response_model,
            latency_ms=int(getattr(payload, "latency_ms", 0) or 0),
        )

    @hook("tool_post_invoke", mode=PluginMode.SEQUENTIAL)
    async def tool_post_invoke(self, payload, context) -> None:
        del context
        tool_call = getattr(payload, "model_tool_call", None)
        tool_name = getattr(tool_call, "name", "")
        self.collector.note_tool(
            tool_name=tool_name,
            output=getattr(payload, "tool_output", None),
            success=bool(getattr(payload, "success", False)),
        )

    @hook("validation_post_check", mode=PluginMode.SEQUENTIAL)
    async def validation_post_check(self, payload, context) -> None:
        del context
        self.collector.note_validation(
            passed_count=int(getattr(payload, "passed_count", 0) or 0),
            failed_count=int(getattr(payload, "failed_count", 0) or 0),
        )


async def run_react_loop(
    session,
    *,
    goal: str,
    tools: list,
    model_options: dict,
    loop_budget: int,
    timeout_s: int,
    submission_format: type | None,
    collector: SolveTraceCollector,
    turn_requirements: Callable[[int, int, SolveTraceCollector], list[object]],
    submission_requirements: list[object],
    strategy_for_requirements: Callable[[list[object]], object | None],
    harness_experiments: tuple[str, ...] = (),
    hooks_enabled: bool = False,
 ) -> tuple[object | None, str]:
    from mellea.core.utils import FancyLogger
    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.react import (
        MELLEA_FINALIZER_TOOL,
        ReactInitiator,
        ReactThought,
    )
    from mellea.stdlib.context import ChatContext
    
    from mcode.llm.harness_experiments import (
        FINALIZER_SUCCESS_GUARD_V1,
        MELLEA_LOOP_DETECT_V1,
    )

    loop_detect_enabled = MELLEA_LOOP_DETECT_V1 in harness_experiments
    finalizer_success_guard_enabled = FINALIZER_SUCCESS_GUARD_V1 in harness_experiments
    force_message = "Switch approaches now. The recent tool-call pattern is stuck."
    nudge_message = (
        "You already tried this exact tool call. Use the result you have and make progress."
    )

    async def _run() -> tuple[object | None, str]:
        context = getattr(session, "ctx", None)
        if not isinstance(context, ChatContext):
            context = ChatContext()
        context = context.add(ReactInitiator(goal, tools))
        strict_tool_ordering = _requires_strict_tool_ordering(
            getattr(session.backend, "model_id", None)
        )

        compress_context = os.environ.get("MCODE_COMPRESS_CONTEXT", "0") == "1"
        loop_history: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        last_loop_signal: str | None = None
        has_run_tests_tool = any(getattr(tool, "name", "") == "run_tests" for tool in tools)
        edit_since_verification = False
        reminded_after_edit = False
        for turn in range(1, loop_budget + 1):
            collector.note_turn(turn)
            FancyLogger.get_logger().info(f"## ReACT TURN NUMBER {turn}")

            if compress_context and turn == max(3, loop_budget // 2):
                context = _compress_old_tool_outputs(context)

            if loop_detect_enabled:
                loop_signal = _detect_loop(loop_history)
                if loop_signal is None:
                    last_loop_signal = None
                elif loop_signal != last_loop_signal:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                force_message
                                if loop_signal == "force_switch"
                                else nudge_message
                            ),
                        )
                    )
                    last_loop_signal = loop_signal

            if (
                has_run_tests_tool
                and edit_since_verification
                and not reminded_after_edit
                and not collector.verification_succeeded
            ):
                context = context.add(
                    Message(
                        role="user",
                        content=(
                            "You changed files but have not verified the patch. "
                            "Call run_tests with test_cmd=\"default\" now, then fix failures "
                            "before final_answer."
                        ),
                    )
                )
                reminded_after_edit = True

            requirements = turn_requirements(turn, loop_budget, collector)
            result, next_context = await mfuncs.aact(
                ReactThought(),
                context=context,
                backend=session.backend,
                requirements=requirements or None,
                strategy=strategy_for_requirements(requirements),
                model_options=model_options,
                tool_calls=True,
                silence_context_type_warning=True,
                await_result=True,
            )
            assert isinstance(next_context, ChatContext)
            context = next_context
            if not hooks_enabled:
                collector.note_generation(
                    prompt=_last_prompt(session),
                    usage=getattr(result, "usage", None),
                    provider=getattr(result, "provider", None),
                    response_model=getattr(result, "model", None),
                    latency_ms=0,
                )

            tool_responses = []
            if result.tool_calls:
                invalid_calls = []
                blocked_finalizers = []
                valid_tool_calls = {}
                for key, tool_call in result.tool_calls.items():
                    tool_name = getattr(tool_call, "name", "") or str(key)
                    if not tool_name:
                        continue
                    if (
                        tool_name == MELLEA_FINALIZER_TOOL
                        and has_run_tests_tool
                        and not collector.verification_succeeded
                    ):
                        blocked_finalizers.append(
                            "final_answer requires successful verification first"
                        )
                        continue
                    missing_args = _missing_required_args(tool_call)
                    if _should_autofill_finalizer(
                        tool_name,
                        missing_args,
                        collector=collector,
                    ):
                        tool_call.args = {"answer": "Verified patch ready."}
                        missing_args = []
                    if missing_args and not _allow_default_missing_args(
                        tool_name,
                        missing_args,
                    ):
                        invalid_calls.append(
                            f"{tool_name} is missing required args: {', '.join(missing_args)}"
                        )
                        continue
                    valid_tool_calls[key] = tool_call
                    if tool_name != MELLEA_FINALIZER_TOOL:
                        loop_history.append(
                            (tool_name, _freeze_tool_args(getattr(tool_call, "args", None)))
                        )
                if blocked_finalizers:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                "Do not call final_answer yet. Run run_tests with "
                                "test_cmd=\"default\" or fix the failing tests first. "
                                + " ".join(blocked_finalizers)
                            ),
                        )
                    )
                if invalid_calls:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                "Some tool calls were malformed and were skipped. "
                                "Retry them with all required arguments. "
                                + " ".join(invalid_calls)
                            ),
                        )
                    )
                if not valid_tool_calls:
                    continue
                if len(valid_tool_calls) != len(result.tool_calls):
                    result.tool_calls = valid_tool_calls
                tool_responses = await acall_tools(result, backend=session.backend)
                for tool_result in tool_responses:
                    if strict_tool_ordering:
                        context = context.add(Message(role="assistant", content=""))
                    context = context.add(tool_result)
                    tool_output = getattr(
                        tool_result,
                        "tool_output",
                        getattr(tool_result, "content", None),
                    )
                    if not hooks_enabled:
                        collector.note_tool(
                            tool_name=tool_result.name,
                            output=tool_output,
                            success=not isinstance(tool_output, Exception),
                        )
                    if tool_result.name == "edit" and _edit_was_applied(tool_output):
                        edit_since_verification = True
                        reminded_after_edit = False
                    elif tool_result.name == "run_tests":
                        edit_since_verification = False
                        reminded_after_edit = False
            finalizer_response = next(
                (
                    tool_result
                    for tool_result in tool_responses
                    if tool_result.name == MELLEA_FINALIZER_TOOL
                ),
                None,
            )
            if finalizer_response is not None:
                finalizer_output = getattr(
                    finalizer_response,
                    "tool_output",
                    getattr(finalizer_response, "content", None),
                )
                if finalizer_success_guard_enabled and isinstance(finalizer_output, Exception):
                    continue
                if submission_format is None:
                    submission = str(finalizer_response.content)
                else:
                    submission, next_context = await mfuncs.aact(
                        ReactThought(),
                        context=context,
                        backend=session.backend,
                        requirements=submission_requirements or None,
                        strategy=strategy_for_requirements(submission_requirements),
                        format=submission_format,
                        model_options=model_options,
                        silence_context_type_warning=True,
                        await_result=True,
                    )
                    assert isinstance(next_context, ChatContext)
                    context = next_context
                    if not hooks_enabled:
                        collector.note_generation(
                            prompt=_last_prompt(session),
                            usage=getattr(submission, "usage", None),
                            provider=getattr(submission, "provider", None),
                            response_model=getattr(submission, "model", None),
                            latency_ms=0,
                        )
                session.ctx = context
                if submission_format is None:
                    return submission, "submitted"
                return getattr(submission, "parsed_repr", None), "submitted"

        session.ctx = context
        return None, "budget_exhausted"

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_s)
    except TimeoutError:
        return None, "budget_exhausted"


def _compress_old_tool_outputs(context):
    from mellea.stdlib.components.chat import ToolMessage
    from mellea.stdlib.context import ChatContext

    messages = context.view_for_generation()
    if messages is None or len(messages) < 6:
        return context

    new_ctx = ChatContext()
    keep_tail = 4
    for index, message in enumerate(messages):
        if index < 2 or index >= len(messages) - keep_tail:
            new_ctx = new_ctx.add(message)
            continue
        if isinstance(message, ToolMessage) and len(str(message.content)) > 200:
            summary = str(message.content)[:150] + "..."
            compressed = ToolMessage(
                role=message.role,
                name=message.name,
                content=f"[compressed] {summary}",
                tool_output=getattr(message, "_tool_output", message.content),
                args=message.arguments,
                tool=getattr(message, "_tool"),
            )
            new_ctx = new_ctx.add(compressed)
            continue
        new_ctx = new_ctx.add(message)
    return new_ctx


def _run_tests_succeeded(output: str) -> bool:
    return (
        "PASSED" in output
        and "FAILED" not in output
        and "TIMEOUT" not in output
        and "ERROR" not in output
        and "BLOCKED" not in output
    )


def _edit_was_applied(output: object) -> bool:
    if isinstance(output, Exception):
        return False
    return "APPLIED" in str(output)


def _serialize_prompt(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _normalize_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_usage(value: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None

    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = value.get(key)
        if raw is None:
            continue
        try:
            out[key] = int(raw)
        except (TypeError, ValueError):
            continue
    if not out:
        return None
    out.setdefault("prompt_tokens", 0)
    out.setdefault("completion_tokens", 0)
    out.setdefault("total_tokens", out["prompt_tokens"] + out["completion_tokens"])
    return out


def _last_prompt(session) -> object | None:
    last_prompt = getattr(session, "last_prompt", None)
    if last_prompt is None:
        return None
    try:
        return last_prompt()
    except Exception:
        return None


def _requires_strict_tool_ordering(model_id: object | None) -> bool:
    text = _normalize_optional_str(model_id)
    if text is None:
        return False
    lowered = text.lower()
    return "minimax" in lowered or "mistral" in lowered


def _freeze_tool_args(args: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(args, dict):
        return ()
    return tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, default=str))
            for key, value in args.items()
        )
    )


def _detect_loop(history: list[tuple[str, tuple[tuple[str, str], ...]]]) -> str | None:
    if len(history) < 2:
        return None
    if history[-1] != history[-2]:
        return None
    if len(history) >= 3 and history[-1] == history[-3]:
        return "force_switch"
    return "nudge"


def _missing_required_args(tool_call) -> list[str]:
    tool = getattr(tool_call, "func", None)
    schema = getattr(tool, "as_json_tool", None)
    if not isinstance(schema, dict):
        return []
    function_schema = schema.get("function")
    if not isinstance(function_schema, dict):
        return []
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return []
    required = parameters.get("required")
    if not isinstance(required, list):
        return []
    args = getattr(tool_call, "args", None)
    if not isinstance(args, dict):
        args = {}
    return [name for name in required if name not in args]


def _allow_default_missing_args(tool_name: str, missing_args: list[str]) -> bool:
    return tool_name == "run_tests" and missing_args == ["test_cmd"]


def _should_autofill_finalizer(
    tool_name: str,
    missing_args: list[str],
    *,
    collector: SolveTraceCollector,
 ) -> bool:
    return (
        tool_name == "final_answer"
        and missing_args == ["answer"]
        and collector.verification_succeeded
    )