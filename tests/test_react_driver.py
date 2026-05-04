from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mellea.backends.tools import MelleaTool
from mellea.core.base import ModelToolCall
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.context import ChatContext

from mcode.agent.verification import build_run_tests_tool, build_verification_policy
from mcode.llm.react_driver import (
    SolveTraceCollector,
    SolveTracePlugin,
    _model_output_text,
    _should_enforce_first_edit,
    _text_preview,
    run_react_loop,
)
from mcode.llm.session import PatchSubmission
from mcode.mellea_compat import (
    acall_tools_with_arg_compat,
    apply_provider_compatibility_patches,
    inspect_tool_call_arg_compat,
)


def test_model_output_text_extracts_nested_content():
    result = SimpleNamespace(
        _underlying_value={"content": [SimpleNamespace(text="nested response")]}
    )

    assert _model_output_text(result) == "nested response"


def test_should_enforce_first_edit_after_browsing_window():
    collector = SolveTraceCollector()

    assert not _should_enforce_first_edit(turn=9, loop_budget=15, collector=collector)
    assert _should_enforce_first_edit(turn=10, loop_budget=15, collector=collector)

    collector.turns_to_first_edit = 4
    assert not _should_enforce_first_edit(turn=10, loop_budget=15, collector=collector)


def test_text_preview_keeps_final_diagnostics():
    preview = _text_preview("start\n" + ("filler\n" * 100) + "final error", max_preview=80)

    assert preview.startswith("start")
    assert "[output preview truncated, keeping final diagnostics]" in preview
    assert preview.endswith("final error")


def test_solve_trace_collector_emits_live_events_without_diagnostics():
    events = []
    collector = SolveTraceCollector(
        live_event_sink=lambda event_type, payload: events.append((event_type, payload))
    )

    collector.note_turn(2)
    collector.note_tool(
        tool_name="run_tests", output="PASSED", success=True, tool_args={"test_cmd": "default"}
    )

    assert collector.diagnostic_events == []
    assert events[0] == ("turn_start", {"turn": 2, "payload": {"turn": 2}})
    assert any(event_type == "run_tests" for event_type, _payload in events)


def test_solve_trace_plugin_collects_generation_tool_and_validation_data():
    collector = SolveTraceCollector()
    plugin = SolveTracePlugin(collector)
    collector.note_turn(3)

    asyncio.run(
        plugin.generation_post_call(
            SimpleNamespace(
                prompt=[{"role": "user", "content": "hi"}],
                latency_ms=25,
                model_output=SimpleNamespace(
                    usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                    provider="openai",
                    model="test-model",
                ),
            ),
            {},
        )
    )
    asyncio.run(
        plugin.tool_post_invoke(
            SimpleNamespace(
                model_tool_call=SimpleNamespace(name="edit"),
                tool_output="ok",
                success=True,
            ),
            {},
        )
    )
    asyncio.run(
        plugin.tool_post_invoke(
            SimpleNamespace(
                model_tool_call=SimpleNamespace(name="run_tests"),
                tool_output="PASSED",
                success=True,
            ),
            {},
        )
    )
    asyncio.run(
        plugin.validation_post_check(
            SimpleNamespace(passed_count=2, failed_count=1),
            {},
        )
    )

    assert collector.prompt_snapshot == '[{"content": "hi", "role": "user"}]'
    assert collector.prompt_tokens == 10
    assert collector.completion_tokens == 4
    assert collector.total_tokens == 14
    assert collector.provider == "openai"
    assert collector.response_model == "test-model"
    assert collector.generation_latency_ms == 25
    assert collector.turns_to_first_edit == 3
    assert collector.turns_to_first_verification == 3
    assert collector.verification_succeeded is True
    assert collector.validation_passed_count == 2
    assert collector.validation_failed_count == 1


def test_run_tests_success_requires_test_like_command_for_verification():
    collector = SolveTraceCollector(diagnostic_enabled=True)
    collector.note_turn(1)

    collector.note_tool(
        tool_name="run_tests",
        output="$ cd /testbed && find . -type f | head\nPASSED\n",
        success=True,
        tool_args={"test_cmd": "cd /testbed && find . -type f | head"},
    )

    assert collector.turns_to_first_verification == 1
    assert collector.verification_succeeded is False
    event = collector.diagnostic_events[-1]
    assert event["event_type"] == "run_tests"
    assert event["payload"]["counts_as_verification"] is False

    collector.note_tool(
        tool_name="edit",
        output="$ edit foo.py\nAPPLIED\nok",
        success=True,
        tool_args={"path": "foo.py"},
    )

    collector.note_tool(
        tool_name="run_tests",
        output="$ cd /testbed && python -m pytest tests/test_foo.py\nPASSED\n",
        success=True,
        tool_args={"test_cmd": "cd /testbed && python -m pytest tests/test_foo.py"},
    )

    assert collector.verification_succeeded is True
    event = collector.diagnostic_events[-1]
    assert event["event_type"] == "run_tests"
    assert event["payload"]["counts_as_verification"] is True


def test_run_tests_event_marks_repeated_failure_suppression():
    collector = SolveTraceCollector(diagnostic_enabled=True)
    collector.note_turn(2)

    collector.note_tool(
        tool_name="run_tests",
        output=(
            "$ python -m pytest tests/test_foo.py\n"
            "SKIPPED\n"
            "Previous run_tests already returned FAILED with no edit since then. "
            "Edit the code before rerunning the same tests.\n"
        ),
        success=True,
        tool_args={"test_cmd": "python -m pytest tests/test_foo.py"},
    )

    event = collector.diagnostic_events[-1]
    assert event["event_type"] == "run_tests"
    assert event["payload"]["repeated_failed_run_suppressed"] is True


def test_inspect_tool_call_arg_compat_counts_recoverable_raw_args(tmp_path):
    tool = build_run_tests_tool(
        repo_root=str(tmp_path),
        verification_policy=build_verification_policy(test_cmds=["python -m pytest"]),
    )
    assert tool is not None

    stats = inspect_tool_call_arg_compat(
        {
            "run_tests": SimpleNamespace(args="default", func=tool),
            "edit": SimpleNamespace(args={"path": "foo.py"}, func=tool),
        }
    )

    assert stats == {"raw_arg_call_count": 1, "recoverable_call_count": 1}


def test_provider_patch_fills_missing_final_answer_text():
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_provider_compatibility_patches()
    tool = MelleaTool.from_callable(lambda answer: answer, name="final_answer")

    assert helpers.validate_tool_arguments(tool, {}, strict=False) == {"answer": "Done."}


def test_provider_patch_maps_run_tests_alias_args(tmp_path):
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_provider_compatibility_patches()
    tool = build_run_tests_tool(
        repo_root=str(tmp_path),
        verification_policy=build_verification_policy(test_cmds=["python -m pytest"]),
    )
    assert tool is not None

    assert helpers.validate_tool_arguments(
        tool,
        {"test_cmd": "default", "timeout": 45, "max_output": 1200},
        strict=False,
    ) == {"test_cmd": "default", "timeout_s": 45, "max_output_chars": 1200}


def test_solve_trace_plugin_records_sanitized_diagnostic_events():
    collector = SolveTraceCollector(diagnostic_enabled=True)
    plugin = SolveTracePlugin(collector)
    collector.note_turn(2)

    asyncio.run(
        plugin.generation_post_call(
            SimpleNamespace(
                prompt=[{"role": "user", "content": "secret prompt"}],
                latency_ms=10,
                model_output=SimpleNamespace(
                    usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    provider="openai",
                    model="test-model",
                    tool_calls={
                        "edit": SimpleNamespace(
                            name="edit",
                            args={"path": "foo.py", "old_str": "a", "new_str": "b"},
                        )
                    },
                ),
            ),
            {},
        )
    )
    asyncio.run(
        plugin.tool_post_invoke(
            SimpleNamespace(
                model_tool_call=SimpleNamespace(
                    name="edit",
                    args={"path": "foo.py", "old_str": "a", "new_str": "b"},
                ),
                tool_output="$ edit foo.py\nAPPLIED\nok",
                execution_time_ms=7,
                success=True,
                error=None,
            ),
            {},
        )
    )

    event_types = [event["event_type"] for event in collector.diagnostic_events]
    assert event_types == ["turn_start", "generation", "tool_result", "edit_result"]
    generation = collector.diagnostic_events[1]["payload"]
    assert generation["tool_calls"][0]["args"]["old_str"] == "[redacted]"
    edit_result = collector.diagnostic_events[-1]["payload"]
    assert edit_result["path"] == "foo.py"
    assert edit_result["status"] == "APPLIED"
    assert "old_str" not in edit_result


def test_run_react_loop_returns_structured_submission(monkeypatch):
    outputs = iter(
        [
            (
                SimpleNamespace(tool_calls={"final_answer": object()}),
                ChatContext(),
            ),
            (
                SimpleNamespace(parsed_repr=PatchSubmission(summary="done", tests_ran=["default"])),
                ChatContext(),
            ),
        ]
    )

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        return [SimpleNamespace(name="final_answer")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    collector = SolveTraceCollector()
    monkeypatch.setattr("mellea.stdlib.functional._acall_tools", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=3,
            timeout_s=5,
            submission_format=PatchSubmission,
            collector=collector,
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert terminal_reason == "submitted"
    assert submission == PatchSubmission(summary="done", tests_ran=["default"])


def test_run_react_loop_uses_plain_final_answer_without_format(monkeypatch):
    outputs = iter(
        [
            (
                SimpleNamespace(tool_calls={"final_answer": object()}),
                ChatContext(),
            ),
        ]
    )

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        return [SimpleNamespace(name="final_answer", content="done")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mellea.stdlib.functional._acall_tools", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=3,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert terminal_reason == "submitted"
    assert submission == "done"


def test_run_react_loop_uses_finalizer_content_when_other_tools_returned(monkeypatch):
    outputs = iter(
        [
            (
                SimpleNamespace(tool_calls={"edit": object(), "final_answer": object()}),
                ChatContext(),
            )
        ]
    )

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        return [
            SimpleNamespace(name="edit", content="patched"),
            SimpleNamespace(name="final_answer", content="done"),
        ]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=1,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert terminal_reason == "submitted"
    assert submission == "done"


def test_run_react_loop_retries_missing_required_args(monkeypatch):
    seen_user_messages: list[list[str]] = []
    outputs = iter(
        [
            (
                SimpleNamespace(
                    tool_calls={
                        "read_file": SimpleNamespace(
                            name="read_file",
                            args={},
                            func=MelleaTool.from_callable(
                                lambda path: path,
                                name="read_file",
                            ),
                        ),
                    }
                ),
                ChatContext(),
            ),
            (SimpleNamespace(tool_calls=None), ChatContext()),
        ]
    )
    acall_invoked = {"value": False}

    async def fake_aact(*args, **kwargs):
        del args
        context = kwargs["context"]
        seen_user_messages.append(
            [
                str(message.content)
                for message in context.as_list()
                if getattr(message, "role", None) == "user"
            ]
        )
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        acall_invoked["value"] = True
        return []

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=2,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert acall_invoked["value"] is False
    assert any(
        "read_file is missing required args: path" in message for message in seen_user_messages[-1]
    )


def test_run_react_loop_recovers_textual_tool_call(monkeypatch):
    edit_tool = MelleaTool.from_callable(
        lambda path, old_str, new_str: path,
        name="edit",
    )
    executed: list[dict[str, object]] = []

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return (
            SimpleNamespace(
                tool_calls=None,
                value=(
                    "Thought: fix it.\nAction:\n```json\n"
                    '{"name": "edit", "arguments": {"path": "foo.py", '
                    r'"old_str": "^[\w.@+-]+$", "new_str": "\\A[\\w.@+-]+\\Z"}}'
                    "\n```"
                ),
            ),
            ChatContext(),
        )

    async def fake_acall_tools(result, backend):
        del backend
        executed.append(result.tool_calls)
        return [SimpleNamespace(name="edit", content="$ edit foo.py\nAPPLIED\nok")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    collector = SolveTraceCollector(diagnostic_enabled=True)
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[edit_tool],
            model_options={},
            loop_budget=1,
            timeout_s=5,
            submission_format=None,
            collector=collector,
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert len(executed) == 1
    recovered = next(iter(executed[0].values()))
    assert set(executed[0]) == {"edit"}
    assert recovered.name == "edit"
    assert recovered.args == {
        "path": "foo.py",
        "old_str": "^[\\w.@+-]+$",
        "new_str": "\\A[\\w.@+-]+\\Z",
    }
    assert any(
        event["event_type"] == "text_tool_call_recovery" for event in collector.diagnostic_events
    )


def test_run_react_loop_nudges_when_budget_spent_without_edit(monkeypatch):
    seen_user_messages: list[list[str]] = []
    outputs = iter((SimpleNamespace(tool_calls=None), ChatContext()) for _ in range(6))

    async def fake_aact(*args, **kwargs):
        del args
        context = kwargs["context"]
        seen_user_messages.append(
            [
                str(message.content)
                for message in context.as_list()
                if getattr(message, "role", None) == "user"
            ]
        )
        return next(outputs)

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[SimpleNamespace(name="run_tests")],
            model_options={},
            loop_budget=6,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert any(
        "Stop browsing and call edit" in message
        for messages in seen_user_messages
        for message in messages
    )
    assert any("Stop browsing and call edit" in message for message in seen_user_messages[1])
    assert any(
        "did not call a tool" in message and "respond with exactly one tool call" in message
        for messages in seen_user_messages
        for message in messages
    )


def test_run_react_loop_executes_valid_calls_when_batch_has_malformed_finalizer(monkeypatch):
    edit_tool = MelleaTool.from_callable(
        lambda path, old_str, new_str: path,
        name="edit",
    )
    finalizer_tool = MelleaTool.from_callable(lambda answer: answer, name="final_answer")
    outputs = iter(
        [
            (
                SimpleNamespace(
                    tool_calls={
                        "edit": ModelToolCall(
                            name="edit",
                            func=edit_tool,
                            args={"path": "foo.py", "old_str": "a", "new_str": "b"},
                        ),
                        "final_answer": ModelToolCall(
                            name="final_answer",
                            func=finalizer_tool,
                            args={},
                        ),
                    }
                ),
                ChatContext(),
            ),
            (SimpleNamespace(tool_calls=None), ChatContext()),
        ]
    )
    executed: list[list[str]] = []

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del backend
        executed.append(list(result.tool_calls))
        return [SimpleNamespace(name="edit", content="$ edit foo.py\nAPPLIED\nok")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[SimpleNamespace(name="run_tests")],
            model_options={},
            loop_budget=2,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert executed == [["edit"]]


def test_run_react_loop_blocks_final_answer_until_verification_succeeds(monkeypatch):
    finalizer_tool = MelleaTool.from_callable(lambda answer: answer, name="final_answer")
    edit_tool = MelleaTool.from_callable(
        lambda path, old_str, new_str: path,
        name="edit",
    )
    run_tests_tool = MelleaTool.from_callable(
        lambda test_cmd="default": test_cmd,
        name="run_tests",
    )
    outputs = iter(
        [
            (
                SimpleNamespace(
                    tool_calls={
                        "final_answer": ModelToolCall(
                            name="final_answer",
                            func=finalizer_tool,
                            args={"answer": "done early"},
                        )
                    }
                ),
                ChatContext(),
            ),
            (
                SimpleNamespace(
                    tool_calls={
                        "edit": ModelToolCall(
                            name="edit",
                            func=edit_tool,
                            args={"path": "foo.py", "old_str": "x", "new_str": "y"},
                        )
                    }
                ),
                ChatContext(),
            ),
            (
                SimpleNamespace(
                    tool_calls={
                        "run_tests": ModelToolCall(
                            name="run_tests",
                            func=run_tests_tool,
                            args={"test_cmd": "default"},
                        )
                    }
                ),
                ChatContext(),
            ),
            (
                SimpleNamespace(
                    tool_calls={
                        "final_answer": ModelToolCall(
                            name="final_answer",
                            func=finalizer_tool,
                            args={"answer": "done"},
                        )
                    }
                ),
                ChatContext(),
            ),
        ]
    )
    executed: list[str] = []

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del backend
        name = next(iter(result.tool_calls))
        executed.append(name)
        if name == "edit":
            return [SimpleNamespace(name="edit", content="$ edit foo.py\nAPPLIED\nok")]
        if name == "run_tests":
            return [SimpleNamespace(name="run_tests", content="$ pytest\nPASSED\nok")]
        return [SimpleNamespace(name="final_answer", content="done")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[SimpleNamespace(name="run_tests"), SimpleNamespace(name="edit")],
            model_options={},
            loop_budget=4,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission == "done"
    assert terminal_reason == "submitted"
    assert executed == ["edit", "run_tests", "final_answer"]


def test_run_react_loop_reminds_to_verify_after_edit(monkeypatch):
    seen_user_messages: list[list[str]] = []
    edit_tool = MelleaTool.from_callable(
        lambda path, old_str, new_str: path,
        name="edit",
    )
    outputs = iter(
        [
            (
                SimpleNamespace(
                    tool_calls={
                        "edit": ModelToolCall(
                            name="edit",
                            func=edit_tool,
                            args={"path": "foo.py", "old_str": "a", "new_str": "b"},
                        )
                    }
                ),
                ChatContext(),
            ),
            (SimpleNamespace(tool_calls=None), ChatContext()),
        ]
    )

    async def fake_aact(*args, **kwargs):
        del args
        context = kwargs["context"]
        seen_user_messages.append(
            [
                str(message.content)
                for message in context.as_list()
                if getattr(message, "role", None) == "user"
            ]
        )
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        return [SimpleNamespace(name="edit", content="$ edit foo.py\nAPPLIED\nok")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[SimpleNamespace(name="run_tests")],
            model_options={},
            loop_budget=2,
            timeout_s=5,
            submission_format=None,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert any(
        'Call run_tests with test_cmd="default"' in message for message in seen_user_messages[-1]
    )


def test_run_react_loop_autofills_verified_finalizer(monkeypatch):
    finalizer_tool = MelleaTool.from_callable(
        lambda answer: answer,
        name="final_answer",
    )
    outputs = iter(
        [
            (
                SimpleNamespace(
                    tool_calls={
                        "final_answer": ModelToolCall(
                            name="final_answer",
                            func=finalizer_tool,
                            args={},
                        )
                    }
                ),
                ChatContext(),
            )
        ]
    )
    captured: dict[str, object] = {}

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del backend
        captured["args"] = result.tool_calls["final_answer"].args
        return [SimpleNamespace(name="final_answer", content="done")]

    session = SimpleNamespace(ctx=ChatContext(), backend=object())
    collector = SolveTraceCollector()
    collector.verification_succeeded = True
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=1,
            timeout_s=5,
            submission_format=None,
            collector=collector,
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    assert captured["args"] == {"answer": "Verified patch ready."}
    assert submission == "done"
    assert terminal_reason == "submitted"


def test_run_react_loop_times_out_as_budget_exhausted():
    async def fake_aact(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(0.05)
        return SimpleNamespace(tool_calls=None), ChatContext()

    session = SimpleNamespace(ctx=ChatContext(), backend=object())

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
        submission, terminal_reason = asyncio.run(
            run_react_loop(
                session,
                goal="Fix it",
                tools=[],
                model_options={},
                loop_budget=3,
                timeout_s=0,
                submission_format=PatchSubmission,
                collector=SolveTraceCollector(),
                turn_requirements=lambda turn, budget, state: [],
                submission_requirements=[],
                strategy_for_requirements=lambda requirements: None,
                hooks_enabled=False,
            )
        )

    assert submission is None
    assert terminal_reason == "budget_exhausted"


def test_acall_tools_normalizes_string_tool_args(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_acall_tools(result, backend):
        del backend
        captured["args"] = result.tool_calls["run_tests"].args
        return []

    monkeypatch.setattr("mellea.stdlib.functional._acall_tools", fake_acall_tools)

    tool = ModelToolCall(
        name="run_tests",
        func=MelleaTool.from_callable(lambda test_cmd="default": "ok", name="run_tests"),
        args="default",
    )
    asyncio.run(
        acall_tools_with_arg_compat(
            SimpleNamespace(tool_calls={"run_tests": tool}), backend=object()
        )
    )

    assert captured["args"] == {"test_cmd": "default"}


def test_acall_tools_normalizes_string_args_for_multi_param_tools(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_acall_tools(result, backend):
        del backend
        captured["args"] = result.tool_calls["run_tests"].args
        return []

    monkeypatch.setattr("mellea.stdlib.functional._acall_tools", fake_acall_tools)

    run_tests_tool = build_run_tests_tool(
        repo_root=".",
        verification_policy=build_verification_policy(test_cmds=["pytest -q"]),
    )
    assert run_tests_tool is not None
    tool = ModelToolCall(
        name="run_tests",
        func=run_tests_tool,
        args="default",
    )
    asyncio.run(
        acall_tools_with_arg_compat(
            SimpleNamespace(tool_calls={"run_tests": tool}), backend=object()
        )
    )

    assert captured["args"] == {"test_cmd": "default"}


def test_run_react_loop_inserts_assistant_bridge_for_strict_models(monkeypatch):
    tool = ModelToolCall(
        name="run_tests",
        func=MelleaTool.from_callable(lambda test_cmd="default": "ok", name="run_tests"),
        args={"test_cmd": "default"},
    )
    tool_message = ToolMessage(
        role="tool",
        content="ok",
        tool_output="ok",
        name="run_tests",
        args={"test_cmd": "default"},
        tool=tool,
    )
    outputs = iter([(SimpleNamespace(tool_calls={"run_tests": tool}), ChatContext())])

    async def fake_aact(*args, **kwargs):
        del args, kwargs
        return next(outputs)

    async def fake_acall_tools(result, backend):
        del result, backend
        return [tool_message]

    session = SimpleNamespace(
        ctx=ChatContext(),
        backend=SimpleNamespace(model_id="MiniMaxAI/MiniMax-M2.5"),
    )
    monkeypatch.setattr("mellea.stdlib.functional.aact", fake_aact)
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools_with_arg_compat", fake_acall_tools)

    submission, terminal_reason = asyncio.run(
        run_react_loop(
            session,
            goal="Fix it",
            tools=[],
            model_options={},
            loop_budget=1,
            timeout_s=5,
            submission_format=PatchSubmission,
            collector=SolveTraceCollector(),
            turn_requirements=lambda turn, budget, state: [],
            submission_requirements=[],
            strategy_for_requirements=lambda requirements: None,
            hooks_enabled=False,
        )
    )

    history = session.ctx.as_list()
    assert submission is None
    assert terminal_reason == "budget_exhausted"
    assert history[-2].role == "assistant"
    assert history[-2].content == ""
    assert history[-1].role == "tool"


def test_provider_patch_normalizes_raw_string_tool_args():
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_provider_compatibility_patches()
    tool = MelleaTool.from_callable(lambda test_cmd="default": "ok", name="run_tests")

    assert helpers.validate_tool_arguments(tool, "default", strict=False) == {"test_cmd": "default"}


def test_provider_patch_drops_unspecified_optional_none_values():
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_provider_compatibility_patches()
    run_tests_tool = build_run_tests_tool(
        repo_root=".",
        verification_policy=build_verification_policy(test_cmds=["pytest -q"]),
    )
    assert run_tests_tool is not None

    assert helpers.validate_tool_arguments(
        run_tests_tool, {"test_cmd": "default"}, strict=False
    ) == {"test_cmd": "default"}


def test_provider_patch_inserts_synthetic_assistant_before_tool_messages():
    import mellea.backends.openai as openai_backend

    apply_provider_compatibility_patches()
    fixed = openai_backend._fix_tool_call_ordering(
        [
            {"role": "system", "content": "hi"},
            {"role": "tool", "content": "ok", "name": "run_tests"},
        ]
    )

    assert fixed[1]["role"] == "assistant"
    assert fixed[1]["tool_calls"][0]["function"]["name"] == "run_tests"
    assert fixed[2]["role"] == "tool"
