from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mellea.backends.tools import MelleaTool
from mellea.core.base import ModelToolCall
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.context import ChatContext

from mcode.agent.verification import build_run_tests_tool, build_verification_policy
from mcode.llm.react_driver import SolveTraceCollector, SolveTracePlugin, run_react_loop
from mcode.llm.session import PatchSubmission
from mcode.mellea_compat import acall_tools, apply_runtime_patches, build_tool_from_callable


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
    outputs = iter([
        (
            SimpleNamespace(tool_calls={"edit": object(), "final_answer": object()}),
            ChatContext(),
        )
    ])

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
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools", fake_acall_tools)

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
    asyncio.run(acall_tools(SimpleNamespace(tool_calls={"run_tests": tool}), backend=object()))

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
    asyncio.run(acall_tools(SimpleNamespace(tool_calls={"run_tests": tool}), backend=object()))

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
    monkeypatch.setattr("mcode.llm.react_driver.acall_tools", fake_acall_tools)

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


def test_runtime_patch_normalizes_openai_helper_tool_args():
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_runtime_patches()
    tool = MelleaTool.from_callable(lambda test_cmd="default": "ok", name="run_tests")

    assert helpers.validate_tool_arguments(tool, "default", strict=False) == {"test_cmd": "default"}


def test_runtime_patch_preserves_optional_defaults_in_validated_tool_args():
    import mellea.helpers.openai_compatible_helpers as helpers

    apply_runtime_patches()
    read_tool = build_tool_from_callable(
        lambda path, start_line=1, end_line=None: "ok",
        name="read_file",
    )
    run_tests_tool = build_run_tests_tool(
        repo_root=".",
        verification_policy=build_verification_policy(test_cmds=["pytest -q"]),
    )
    assert run_tests_tool is not None

    assert helpers.validate_tool_arguments(read_tool, {"path": "README.md"}, strict=False) == {
        "path": "README.md"
    }
    assert helpers.validate_tool_arguments(run_tests_tool, "default", strict=False) == {
        "test_cmd": "default"
    }


def test_runtime_patch_inserts_synthetic_assistant_before_tool_messages():
    import mellea.backends.openai as openai_backend

    apply_runtime_patches()
    fixed = openai_backend._fix_tool_call_ordering(
        [
            {"role": "system", "content": "hi"},
            {"role": "tool", "content": "ok", "name": "run_tests"},
        ]
    )

    assert fixed[1]["role"] == "assistant"
    assert fixed[1]["tool_calls"][0]["function"]["name"] == "run_tests"
    assert fixed[2]["role"] == "tool"
