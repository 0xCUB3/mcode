from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from mellea.plugins.base import Plugin
from mellea.plugins.decorators import hook
from mellea.plugins.types import PluginMode

from mcode.mellea_compat import acall_tools_with_arg_compat, inspect_tool_call_arg_compat


@dataclass
class SolveTraceCollector:
    current_turn: int = 0
    diagnostic_enabled: bool = False
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
    diagnostic_events: list[dict[str, object]] = field(default_factory=list)
    last_model_output: dict[str, object] | None = None

    verification_evidence: list[dict[str, object]] = field(default_factory=list)

    def note_event(
        self,
        event_type: str,
        payload: Mapping[str, object] | None = None,
        *,
        turn: int | None = None,
    ) -> None:
        if not self.diagnostic_enabled:
            return
        event_turn = self.current_turn if turn is None else turn
        self.diagnostic_events.append(
            {
                "turn": event_turn if event_turn > 0 else None,
                "event_type": event_type,
                "payload": _sanitize_diagnostic_payload(payload or {}),
            }
        )

    def note_turn(self, turn: int) -> None:
        self.current_turn = turn
        self.note_event("turn_start", {"turn": turn}, turn=turn)

    def note_generation(
        self,
        *,
        prompt: object,
        usage: dict[str, Any] | None,
        provider: object | None,
        response_model: object | None,
        latency_ms: int,
        tool_calls: object | None = None,
        model_output: object | None = None,
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
        if model_output is not None:
            model_text = _model_output_text(model_output)
            if model_text:
                self.last_model_output = _text_digest(model_text, max_preview=2000)
        payload: dict[str, object] = {
            "provider": normalized_provider,
            "response_model": normalized_model,
            "latency_ms": latency_ms,
            "tool_call_count": _tool_call_count(tool_calls),
        }
        if totals is not None:
            payload["usage"] = totals
        sanitized_calls = _sanitize_tool_calls(tool_calls)
        if sanitized_calls:
            payload["tool_calls"] = sanitized_calls
        self.note_event("generation", payload)

    def note_tool(
        self,
        *,
        tool_name: str,
        output: object,
        success: bool,
        tool_args: object | None = None,
        execution_time_ms: int | None = None,
        error: object | None = None,
    ) -> None:
        turn = max(1, self.current_turn or 1)
        args = _args_mapping(tool_args)
        output_text = _safe_text(output)
        if tool_name == "edit" and self.turns_to_first_edit is None:
            self.turns_to_first_edit = turn
        if tool_name == "run_tests" and self.turns_to_first_verification is None:
            self.turns_to_first_verification = turn
        if (
            tool_name == "run_tests"
            and success
            and _run_tests_succeeded(output_text)
            and _run_tests_counts_as_verification(tool_args)
            and self.turns_to_first_edit is not None
        ):
            self.verification_succeeded = True
        if tool_name == "run_tests":
            self.verification_evidence.append(
                _verification_evidence_payload(
                    args=args,
                    output_text=output_text,
                    execution_time_ms=execution_time_ms,
                )
            )
        self._note_tool_diagnostics(
            tool_name=tool_name,
            output=output_text,
            success=success,
            tool_args=args,
            execution_time_ms=execution_time_ms,
            error=error,
        )

    def _note_tool_diagnostics(
        self,
        *,
        tool_name: str,
        output: object,
        success: bool,
        tool_args: object | None,
        execution_time_ms: int | None,
        error: object | None,
    ) -> None:
        if not self.diagnostic_enabled:
            return
        output_text = _safe_text(output)
        payload: dict[str, object] = {
            "tool_name": tool_name,
            "success": success,
            "status": _parse_status(output_text),
            "output": _text_digest(output_text, max_preview=1000),
        }
        if execution_time_ms is not None:
            payload["execution_time_ms"] = execution_time_ms
        if error is not None:
            payload["error"] = _text_digest(_safe_text(error), max_preview=500)
        self.note_event("tool_result", payload)

        args = _args_mapping(tool_args)
        if target := _read_search_target(tool_name, args):
            self.note_event("read_search_target", target)
        if tool_name == "edit":
            self.note_event("edit_result", _edit_result_payload(args, output_text))
        if tool_name == "run_tests":
            self.note_event("run_tests", _run_tests_payload(args, output_text))

    def note_validation(self, *, passed_count: int, failed_count: int) -> None:
        self.validation_passed_count = passed_count
        self.validation_failed_count = failed_count
        self.note_event(
            "validation",
            {"passed_count": passed_count, "failed_count": failed_count},
        )


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
            tool_calls=getattr(model_output, "tool_calls", None),
            model_output=model_output,
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
            tool_args=getattr(tool_call, "args", None),
            execution_time_ms=int(getattr(payload, "execution_time_ms", 0) or 0),
            error=getattr(payload, "error", None),
        )

    @hook("validation_post_check", mode=PluginMode.SEQUENTIAL)
    async def validation_post_check(self, payload, context) -> None:
        del context
        self.collector.note_validation(
            passed_count=int(getattr(payload, "passed_count", 0) or 0),
            failed_count=int(getattr(payload, "failed_count", 0) or 0),
        )


# This stays as a benchmark adapter rather than delegating to
# mellea.stdlib.frameworks.react.react because the native loop currently lacks
# per-turn requirements, async tool dispatch, final-answer gating, timeout labels,
# and trace callbacks needed by result DB/report metrics.


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

    async def _run() -> tuple[object | None, str]:
        context = getattr(session, "ctx", None)
        if not isinstance(context, ChatContext):
            context = ChatContext()
        context = context.add(ReactInitiator(goal, tools))
        strict_tool_ordering = _requires_strict_tool_ordering(
            getattr(session.backend, "model_id", None)
        )

        has_run_tests_tool = any(getattr(tool, "name", "") == "run_tests" for tool in tools)
        edit_since_verification = False
        reminded_after_edit = False
        reminded_without_edit = False
        for turn in range(1, loop_budget + 1):
            collector.note_turn(turn)
            FancyLogger.get_logger().info(f"## ReACT TURN NUMBER {turn}")

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
                            'Call run_tests with test_cmd="default" now, then fix failures '
                            "before final_answer."
                        ),
                    )
                )
                reminded_after_edit = True

            if (
                has_run_tests_tool
                and collector.turns_to_first_edit is None
                and not reminded_without_edit
                and turn >= max(2, loop_budget // 3)
            ):
                context = context.add(
                    Message(
                        role="user",
                        content=(
                            "You have not edited yet. Stop browsing and call edit on "
                            "the best candidate source file with the smallest plausible fix now. "
                            "Do not call final_answer or another search/read before an edit "
                            "unless the last read failed."
                        ),
                    )
                )
                reminded_without_edit = True
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
                    tool_calls=getattr(result, "tool_calls", None),
                    model_output=result,
                )
            if not result.tool_calls:
                recovered_calls = _recover_text_tool_calls(
                    result,
                    tools=tools,
                    collector=collector,
                )
                if recovered_calls:
                    result.tool_calls = recovered_calls

            if not result.tool_calls:
                tool_names = _tool_names_for_feedback(tools)
                collector.note_event(
                    "no_tool_call",
                    {"available_tools": tool_names},
                )
                if tool_names:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                "Your last response did not call a tool, so no repository "
                                "action happened. On the next turn, respond with exactly one "
                                "tool call from: "
                                f"{', '.join(tool_names)}. "
                                "If you know the fix, call edit now. Do not explain in prose."
                            ),
                        )
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
                        collector.note_event(
                            "final_answer",
                            {"action": "autofilled", "reason": "verified_missing_answer"},
                        )
                    if missing_args and not _allow_default_missing_args(
                        tool_name,
                        missing_args,
                    ):
                        invalid_calls.append(
                            f"{tool_name} is missing required args: {', '.join(missing_args)}"
                        )
                        continue
                    valid_tool_calls[key] = tool_call
                collector.note_event(
                    "tool_call_filter",
                    {
                        "total_call_count": len(result.tool_calls),
                        "valid_call_count": len(valid_tool_calls),
                        "invalid_call_count": len(invalid_calls),
                        "blocked_finalizer_count": len(blocked_finalizers),
                        "valid_tool_names": [
                            getattr(call, "name", str(key))
                            for key, call in valid_tool_calls.items()
                        ],
                    },
                )
                if blocked_finalizers:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                "Do not call final_answer yet. Run run_tests with "
                                'test_cmd="default" or fix the failing tests first. '
                                + " ".join(blocked_finalizers)
                            ),
                        )
                    )
                    collector.note_event(
                        "final_answer",
                        {"action": "blocked", "reasons": blocked_finalizers},
                    )
                if invalid_calls:
                    context = context.add(
                        Message(
                            role="user",
                            content=(
                                "Some tool calls were malformed and were skipped. "
                                "Retry them with all required arguments. " + " ".join(invalid_calls)
                            ),
                        )
                    )
                if not valid_tool_calls:
                    continue

                if len(valid_tool_calls) != len(result.tool_calls):
                    result.tool_calls = valid_tool_calls
                compat_stats = inspect_tool_call_arg_compat(result.tool_calls)
                if compat_stats["raw_arg_call_count"]:
                    collector.note_event("tool_arg_compat", compat_stats)
                tool_responses = await acall_tools_with_arg_compat(result, backend=session.backend)
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
                            tool_args=getattr(
                                tool_result,
                                "args",
                                getattr(tool_result, "arguments", None),
                            ),
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
                            tool_calls=getattr(submission, "tool_calls", None),
                            model_output=submission,
                        )
                collector.note_event("final_answer", {"action": "accepted"})
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


def _tool_names_for_feedback(tools: list) -> list[str]:
    names = [str(getattr(tool, "name", "")).strip() for tool in tools]
    return [name for name in names if name]


def _recover_text_tool_calls(
    result: object,
    *,
    tools: list,
    collector: SolveTraceCollector,
) -> dict[str, object]:
    text = _model_output_text(result)
    if not text:
        return {}
    tool_by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    recovered: dict[str, object] = {}
    for payload in _iter_json_objects(text):
        if not isinstance(payload, dict):
            continue
        raw_name = str(
            payload.get("name") or payload.get("tool") or payload.get("action") or ""
        ).strip()
        inferred_from_args = False
        if not raw_name:
            raw_name = _infer_text_tool_name_from_args(payload)
            inferred_from_args = bool(raw_name)
        name = _normalize_text_tool_name(raw_name)
        tool = tool_by_name.get(name)
        if tool is None:
            continue
        if any(key in payload for key in ("arguments", "args", "action_input")):
            args = payload.get(
                "arguments",
                payload.get("args", payload.get("action_input", {})),
            )
        elif inferred_from_args:
            args = payload
        else:
            args = {}
        if not isinstance(args, dict):
            args = {}
        from mellea.core.base import ModelToolCall

        recovered[name] = ModelToolCall(
            name=name,
            func=tool,
            args=args,
        )
    if recovered:
        collector.note_event(
            "text_tool_call_recovery",
            {"tool_names": [getattr(call, "name", "") for call in recovered.values()]},
        )
    return recovered


def _model_output_text(result: object) -> str:
    for attr in ("value", "_underlying_value", "parsed_repr", "content", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    candidates = [
        match.group(1)
        for match in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )
    ]
    candidates.append(text)
    seen: set[str] = set()
    for candidate in candidates:
        yield from _decode_json_objects(candidate, decoder=decoder, seen=seen)
    yield from _iter_react_action_payloads(text, decoder=decoder, seen=seen)


def _decode_json_objects(text: str, *, decoder: json.JSONDecoder, seen: set[str]):
    for match in re.finditer(r"\{", text):
        snippet = text[match.start() :]
        if snippet in seen:
            continue
        seen.add(snippet)
        try:
            payload, _end = decoder.raw_decode(snippet)
        except json.JSONDecodeError:
            payload = _literal_eval_object(snippet)
            if payload is None:
                continue
        yield payload


def _iter_react_action_payloads(text: str, *, decoder: json.JSONDecoder, seen: set[str]):
    action_re = re.compile(r"\bAction\s*:\s*[`'\"]?([A-Za-z_][\w.-]*)", re.IGNORECASE)
    for match in action_re.finditer(text):
        name = match.group(1)
        tail = text[match.end() :]
        next_action = action_re.search(tail)
        if next_action is not None:
            tail = tail[: next_action.start()]
        for args in _decode_json_objects(tail, decoder=decoder, seen=seen):
            if isinstance(args, dict):
                yield {"name": name, "arguments": args}
                break


def _literal_eval_object(text: str) -> object | None:
    prefix = _balanced_brace_prefix(text)
    if prefix is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return ast.literal_eval(prefix)
    except (SyntaxError, ValueError):
        return None


def _balanced_brace_prefix(text: str) -> str | None:
    depth = 0
    in_string: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return None


def _infer_text_tool_name_from_args(payload: dict[str, object]) -> str:
    keys = set(payload)
    if {"old_str", "new_str", "path"} <= keys:
        return "edit"
    if "test_cmd" in keys:
        return "run_tests"
    if "query" in keys:
        return "search_code"
    if "path" in keys:
        return "read_file"
    return ""


def _normalize_text_tool_name(name: str) -> str:
    aliases = {
        "edit_file": "edit",
        "replace": "edit",
        "read": "read_file",
        "search": "search_code",
        "test": "run_tests",
    }
    return aliases.get(name, name)


_REDACTED_KEYS = {
    "api_key",
    "authorization",
    "new_str",
    "old_str",
    "password",
    "secret",
    "token",
}


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _text_digest(text: str, *, max_preview: int) -> dict[str, object]:
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "preview": text[:max_preview],
    }


def _sanitize_diagnostic_payload(value: object, *, max_preview: int = 500) -> object:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in _REDACTED_KEYS or any(part in lowered for part in _REDACTED_KEYS):
                out[key_text] = "[redacted]"
                continue
            out[key_text] = _sanitize_diagnostic_payload(item, max_preview=max_preview)
        return out
    if isinstance(value, list | tuple | set):
        return [_sanitize_diagnostic_payload(item, max_preview=max_preview) for item in value]
    if isinstance(value, bool | int | float) or value is None:
        return value
    text = _safe_text(value)
    if len(text) <= max_preview:
        return text
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "preview": text[:max_preview],
    }


def _args_mapping(value: object | None) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _tool_call_count(tool_calls: object | None) -> int:
    if isinstance(tool_calls, Mapping):
        return len(tool_calls)
    if isinstance(tool_calls, list | tuple):
        return len(tool_calls)
    return 0


def _sanitize_tool_calls(tool_calls: object | None) -> list[dict[str, object]]:
    items: list[object]
    if isinstance(tool_calls, Mapping):
        items = list(tool_calls.values())
    elif isinstance(tool_calls, list | tuple):
        items = list(tool_calls)
    else:
        return []
    out: list[dict[str, object]] = []
    for call in items:
        name = _normalize_optional_str(getattr(call, "name", None))
        args = _args_mapping(getattr(call, "args", None))
        out.append(
            {
                "name": name or "unknown",
                "args": _sanitize_diagnostic_payload(args),
            }
        )
    return out


def _read_search_target(tool_name: str, args: Mapping[str, object]) -> dict[str, object] | None:
    if tool_name == "search_code":
        return {"tool_name": tool_name, "query": args.get("query")}
    if tool_name == "read_file":
        return {
            "tool_name": tool_name,
            "path": args.get("path"),
            "start_line": args.get("start_line"),
            "end_line": args.get("end_line"),
        }
    if tool_name == "find_file":
        return {"tool_name": tool_name, "pattern": args.get("pattern")}
    if tool_name == "list_dir":
        return {"tool_name": tool_name, "path": args.get("path")}
    return None


def _parse_status(text: str) -> str | None:
    for status in ("APPLIED", "REJECTED", "PASSED", "FAILED", "TIMEOUT", "BLOCKED", "ERROR"):
        if status in text:
            return status
    return None


def _line_count(value: object | None) -> int | None:
    if not isinstance(value, str):
        return None
    if value == "":
        return 0
    return value.count("\n") + (0 if value.endswith("\n") else 1)


def _edit_result_payload(args: Mapping[str, object], output_text: str) -> dict[str, object]:
    return {
        "path": args.get("path"),
        "status": _parse_status(output_text),
        "old_line_count": _line_count(args.get("old_str")),
        "new_line_count": _line_count(args.get("new_str")),
        "output": _text_digest(output_text, max_preview=500),
    }


def _run_tests_payload(args: Mapping[str, object], output_text: str) -> dict[str, object]:
    command_match = re.search(r"^\$ (?P<command>.+)$", output_text, flags=re.MULTILINE)
    counts_as_verification = _run_tests_counts_as_verification(args)
    return {
        "test_cmd": args.get("test_cmd"),
        "timeout_s": args.get("timeout_s"),
        "max_output_chars": args.get("max_output_chars"),
        "expanded_command": command_match.group("command") if command_match else None,
        "counts_as_verification": counts_as_verification,
        "repeated_failed_run_suppressed": _is_repeated_failed_run_suppressed(output_text),
        "status": _parse_status(output_text),
        "output": _text_digest(output_text, max_preview=1000),
    }


def _verification_evidence_payload(
    *,
    args: Mapping[str, object],
    output_text: str,
    execution_time_ms: int | None,
) -> dict[str, object]:
    counts_as_verification = _run_tests_counts_as_verification(args)
    ended_at = datetime.now(UTC)
    started_at = (
        ended_at - timedelta(milliseconds=execution_time_ms)
        if execution_time_ms is not None
        else None
    )
    command_label = str(args.get("test_cmd") or "default")
    payload = _run_tests_payload(args, output_text)
    output_digest = _text_digest(output_text, max_preview=1000)
    expanded_command = payload.get("expanded_command")
    command_digest_source = expanded_command or command_label
    return {
        "verifier_name": "run_tests",
        "command_label": command_label,
        "command_digest": hashlib.sha256(
            str(command_digest_source).encode("utf-8", errors="ignore")
        ).hexdigest(),
        "status": payload.get("status") or "UNKNOWN",
        "counted_as_verification": counts_as_verification,
        "output_digest": output_digest["sha256"],
        "output_preview": output_digest["preview"],
        "execution_time_ms": execution_time_ms,
        "started_at": started_at.isoformat() if started_at is not None else None,
        "ended_at": ended_at.isoformat(),
        "timed_out": payload.get("status") == "TIMEOUT",
        "expanded_command": expanded_command,
        "repeated_failed_run_suppressed": payload.get("repeated_failed_run_suppressed", False),
    }


def _is_repeated_failed_run_suppressed(output_text: str) -> bool:
    return (
        "Previous run_tests already returned" in output_text
        and "with no edit since then" in output_text
    )


def _run_tests_counts_as_verification(tool_args: object | None) -> bool:
    args = _args_mapping(tool_args)
    test_cmd = args.get("test_cmd")
    if test_cmd in (None, "", "default"):
        return True
    if not isinstance(test_cmd, str):
        return False
    command = re.sub(r"/testbed\b", "", test_cmd.lower())
    marker_patterns = (
        r"\bpytest\b",
        r"\bunittest\b",
        r"\btox\b",
        r"\bnose\b",
        r"\bassert\b",
        r"\bmanage\.py\s+test\b",
        r"\bgo\s+test\b",
        r"\bcargo\s+test\b",
        r"\bnpm\s+test\b",
        r"\byarn\s+test\b",
        r"\bpnpm\s+test\b",
        r"\bmvn\s+test\b",
        r"\bgradle\s+test\b",
        r"\brspec\b",
        r"\bswift\s+test\b",
        r"(?:^|[\s/&|;])tests?(?:[\s/:]|$)",
        r"(?:^|[\s/&|;])test_[\w.-]+",
    )
    return any(re.search(pattern, command) for pattern in marker_patterns)


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
    callable_defaults = _callable_default_param_names(tool)
    return [name for name in required if name not in args and name not in callable_defaults]


def _callable_default_param_names(tool: object) -> set[str]:
    func = getattr(tool, "_call_tool", None)
    if not callable(func):
        return set()
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect._empty
    }


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
