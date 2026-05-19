from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from mellea.backends.tools import MelleaTool

from mcode.agent.tool_policy import blocked_shell_command_reason
from mcode.llm.react_driver import SolveTraceCollector, SolveTracePlugin, run_react_loop
from mcode.llm.session import LLMSession, hooks_available

_SYSTEM_PROMPT = (
    "You are an expert terminal operator solving a realistic Terminal-Bench task. "
    "You are working inside an isolated benchmark container. Complete the user's "
    "instruction by inspecting files, running commands, writing scripts or outputs, "
    "and leaving the container in the correct final state. You may create and modify "
    "files under the task workspace. Do not read or modify /solution, /tests, or "
    "/logs/verifier. When the task is complete, call final_answer with a brief summary."
)

_PROTECTED_PATHS = ("/solution", "/tests", "/logs/verifier")
_DEFAULT_CWD = "/app"
_MAX_TOOL_OUTPUT = 20_000


@dataclass(frozen=True)
class TerminalSolveResult:
    summary: str | None
    terminal_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider: str | None = None
    response_model: str | None = None
    generation_latency_ms: int | None = None
    diagnostic_events: list[dict[str, object]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentCommandBridge:
    """Synchronous tool bridge for Harbor's async BaseEnvironment.exec API."""

    def __init__(self, environment: Any, loop: asyncio.AbstractEventLoop):
        self.environment = environment
        self.loop = loop

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = _DEFAULT_CWD,
        timeout_sec: int = 120,
        user: str | int | None = None,
    ) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self.environment.exec(
                command,
                cwd=cwd,
                timeout_sec=timeout_sec,
                user=user,
            ),
            self.loop,
        )
        return future.result(timeout=timeout_sec + 30)


def solve_terminal_task(
    *,
    instruction: str,
    command_bridge: EnvironmentCommandBridge,
    model_id: str,
    backend_name: str,
    loop_budget: int = 25,
    temperature: float | None = None,
    seed: int | None = None,
    diagnostic_traces: bool = True,
    live_event_sink: Callable[[str, Mapping[str, object]], None] | None = None,
) -> TerminalSolveResult:
    session = LLMSession(
        model_id=model_id,
        backend_name=backend_name,
        loop_budget=loop_budget,
        temperature=temperature,
        seed=seed,
        diagnostic_traces=diagnostic_traces,
        live_event_sink=live_event_sink,
    )
    tools = make_terminal_tools(command_bridge)
    collector = SolveTraceCollector(
        diagnostic_enabled=diagnostic_traces,
        live_event_sink=live_event_sink,
    )
    enable_hooks = hooks_available()
    runtime_plugins = [SolveTracePlugin(collector)] if enable_hooks else None
    goal = _build_goal(instruction)

    async def _run_once(mellea_session) -> tuple[object | None, str]:
        return await run_react_loop(
            mellea_session,
            goal=goal,
            tools=tools,
            model_options=session._model_options(system_prompt=_SYSTEM_PROMPT),
            loop_budget=max(1, loop_budget),
            timeout_s=int(os.environ.get("MCODE_REACT_TIMEOUT", str(max(1, loop_budget) * 45))),
            submission_format=None,
            collector=collector,
            turn_requirements=lambda *_args, **_kwargs: [],
            submission_requirements=[],
            strategy_for_requirements=lambda _requirements: None,
            hooks_enabled=enable_hooks,
        )

    with session._start_session(plugins=runtime_plugins) as mellea_session:
        submission, terminal_reason = asyncio.run(_run_once(mellea_session))

    return TerminalSolveResult(
        summary=_stringify_submission(submission),
        terminal_reason=terminal_reason,
        prompt_tokens=_none_if_zero(collector.prompt_tokens),
        completion_tokens=_none_if_zero(collector.completion_tokens),
        total_tokens=_none_if_zero(collector.total_tokens),
        provider=collector.provider,
        response_model=collector.response_model,
        generation_latency_ms=_none_if_zero(collector.generation_latency_ms),
        diagnostic_events=(
            list(collector.diagnostic_events) if collector.diagnostic_enabled else None
        ),
        metadata={
            "turns": collector.current_turn,
            "validation_passed_count": collector.validation_passed_count,
            "validation_failed_count": collector.validation_failed_count,
            "last_model_output": collector.last_model_output,
        },
    )


def make_terminal_tools(command_bridge: EnvironmentCommandBridge) -> list[MelleaTool]:
    def shell(command: str, cwd: str | None = None, timeout_sec: int | None = None) -> str:
        if reason := _blocked_command_reason(command):
            return f"BLOCKED: {reason}"
        timeout = _clamp_timeout(timeout_sec or 120)
        try:
            result = command_bridge.exec(command, cwd=cwd or _DEFAULT_CWD, timeout_sec=timeout)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = _truncate("\n".join(part for part in (stdout, stderr) if part))
        return f"exit={result.return_code}\n{output}".rstrip()

    def list_dir(path: str = _DEFAULT_CWD) -> str:
        resolved_path = _workspace_path(path)
        if reason := _blocked_path_reason(resolved_path):
            return f"BLOCKED: {reason}"
        return shell(f"ls -la {shlex.quote(resolved_path)}", cwd="/", timeout_sec=30)

    def read_file(path: str, max_chars: int | None = None) -> str:
        resolved_path = _workspace_path(path)
        if reason := _blocked_path_reason(resolved_path):
            return f"BLOCKED: {reason}"
        max_chars = max(1, min(int(max_chars or _MAX_TOOL_OUTPUT), _MAX_TOOL_OUTPUT))
        command = f"head -c {max_chars} {shlex.quote(resolved_path)}"
        return shell(command, cwd="/", timeout_sec=30)

    def write_file(path: str, content: str) -> str:
        resolved_path = _workspace_path(path)
        if reason := _blocked_path_reason(resolved_path):
            return f"BLOCKED: {reason}"
        payload = base64.b64encode(content.encode("utf-8", errors="replace")).decode("ascii")
        script = (
            "import base64, pathlib, sys; "
            "path = pathlib.Path(sys.argv[1]); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.write_bytes(base64.b64decode(sys.argv[2]))"
        )
        command = (
            f"python3 -c {shlex.quote(script)} {shlex.quote(resolved_path)} {shlex.quote(payload)}"
        )
        return shell(command, cwd="/", timeout_sec=60)

    def replace_in_file(path: str, old_str: str, new_str: str) -> str:
        resolved_path = _workspace_path(path)
        if reason := _blocked_path_reason(resolved_path):
            return f"BLOCKED: {reason}"
        script = """
import pathlib, sys
path = pathlib.Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
text = path.read_text(errors='replace')
count = text.count(old)
if count != 1:
    print(f'ERROR: expected exactly one match, found {count}')
    raise SystemExit(1)
path.write_text(text.replace(old, new, 1))
print('APPLIED')
""".strip()
        command = (
            f"python3 - {shlex.quote(resolved_path)} {shlex.quote(old_str)} "
            f"{shlex.quote(new_str)} <<'PY'\n{script}\nPY"
        )
        return shell(command, cwd="/", timeout_sec=60)

    return [
        MelleaTool.from_callable(shell, name="shell"),
        MelleaTool.from_callable(list_dir, name="list_dir"),
        MelleaTool.from_callable(read_file, name="read_file"),
        MelleaTool.from_callable(write_file, name="write_file"),
        MelleaTool.from_callable(replace_in_file, name="replace_in_file"),
    ]


def _build_goal(instruction: str) -> str:
    return (
        "Complete this Terminal-Bench task in the container. Work from /app unless the "
        "instruction says otherwise. You may create scripts, run commands, and write output "
        "files.\n\n"
        f"Instruction:\n{instruction.strip()}\n\n"
        "Important constraints: do not inspect /solution or /tests, and do not write to "
        "/logs/verifier or reward files. Leave the final answer concise."
    )


def _workspace_path(path: str) -> str:
    text = (path or _DEFAULT_CWD).strip()
    if not text or text == ".":
        return _DEFAULT_CWD
    if text.startswith("/"):
        return text
    return f"{_DEFAULT_CWD.rstrip('/')}/{text}"


def _blocked_command_reason(command: str) -> str | None:
    if reason := blocked_shell_command_reason(command):
        return reason
    for token in _command_path_tokens(command):
        if reason := _blocked_path_reason(token):
            return reason
    return None


def _command_path_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths = []
    for token in tokens:
        candidate = token.lstrip("<>").rstrip(";|&")
        if candidate.startswith("/"):
            paths.append(candidate)
    return paths


def _blocked_path_reason(path: str) -> str | None:
    normalized = path.strip().lower()
    for protected in _PROTECTED_PATHS:
        if normalized == protected or normalized.startswith(protected + "/"):
            return f"protected benchmark path is not available to the agent: {protected}"
    return None


def _clamp_timeout(timeout_sec: int) -> int:
    try:
        value = int(timeout_sec)
    except (TypeError, ValueError):
        value = 120
    return max(1, min(value, int(os.environ.get("MCODE_TERMINAL_TOOL_TIMEOUT", "300"))))


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated to {limit} characters ..."


def _stringify_submission(submission: object | None) -> str | None:
    if submission is None:
        return None
    if isinstance(submission, str):
        return submission
    try:
        return json.dumps(submission, default=str, ensure_ascii=False)
    except TypeError:
        return str(submission)


def _none_if_zero(value: int | None) -> int | None:
    if not value:
        return None
    return int(value)
