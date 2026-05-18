from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from mcode.agent.terminal_agent import EnvironmentCommandBridge, solve_terminal_task

try:  # pragma: no cover - exercised when Harbor imports the agent.
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except Exception:  # pragma: no cover - keeps normal mCode imports independent of Harbor.
    BaseAgent = object  # type: ignore[assignment]
    BaseEnvironment = Any  # type: ignore[assignment]
    AgentContext = Any  # type: ignore[assignment]


class MCodeTerminalBenchAgent(BaseAgent):  # type: ignore[misc, valid-type]
    """Harbor external agent that runs mCode's terminal-mode ReACT harness."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "mcode"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        backend_name: str | None = None,
        loop_budget: int | str | None = None,
        temperature: float | str | None = None,
        seed: int | str | None = None,
        diagnostic_traces: bool | str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if BaseAgent is object:
            self.logs_dir = logs_dir
            self.model_name = model_name
        else:
            super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self.backend_name = backend_name or os.environ.get("MCODE_BACKEND", "openai")
        self.loop_budget = _coerce_int(
            loop_budget if loop_budget is not None else os.environ.get("MCODE_LOOP_BUDGET"),
            default=25,
        )
        self.temperature = _coerce_optional_float(
            temperature if temperature is not None else os.environ.get("MCODE_TEMPERATURE")
        )
        self.seed = _coerce_optional_int(seed if seed is not None else os.environ.get("MCODE_SEED"))
        self.diagnostic_traces = _coerce_bool(
            diagnostic_traces
            if diagnostic_traces is not None
            else os.environ.get("MCODE_DIAGNOSTIC_TRACES"),
            default=True,
        )

    def version(self) -> str | None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            return version("mcode")
        except (ImportError, PackageNotFoundError):
            return "unknown"

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model_id = self.model_name or os.environ.get("MCODE_MODEL")
        if not model_id:
            raise RuntimeError("MCodeTerminalBenchAgent requires --model or MCODE_MODEL")
        loop = asyncio.get_running_loop()
        bridge = EnvironmentCommandBridge(environment=environment, loop=loop)
        result = await asyncio.to_thread(
            solve_terminal_task,
            instruction=instruction,
            command_bridge=bridge,
            model_id=model_id,
            backend_name=self.backend_name,
            loop_budget=self.loop_budget,
            temperature=self.temperature,
            seed=self.seed,
            diagnostic_traces=self.diagnostic_traces,
        )
        context.n_input_tokens = result.prompt_tokens
        context.n_output_tokens = result.completion_tokens
        context.metadata = {
            "terminal_reason": result.terminal_reason,
            "summary": result.summary,
            "provider": result.provider,
            "response_model": result.response_model,
            "generation_latency_ms": result.generation_latency_ms,
            **result.metadata,
        }
        if result.diagnostic_events is not None:
            trace_path = self.logs_dir / "mcode-diagnostic-events.json"
            import json

            trace_path.write_text(json.dumps(result.diagnostic_events, indent=2), encoding="utf-8")
            context.metadata["diagnostic_events_path"] = str(trace_path)


def _coerce_int(value: object, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default
