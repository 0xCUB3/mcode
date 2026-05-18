from __future__ import annotations

from dataclasses import dataclass

from mcode.agent.terminal_agent import make_terminal_tools
from mcode.bench.terminalbench_agent import MCodeTerminalBenchAgent


@dataclass
class _ExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class _Bridge:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str, **_kwargs) -> _ExecResult:
        self.commands.append(command)
        return _ExecResult(stdout="hello\n", return_code=0)


def _tool(tools, name: str):
    return next(tool for tool in tools if tool.name == name)


def test_terminal_tools_block_protected_paths() -> None:
    bridge = _Bridge()
    tools = make_terminal_tools(bridge)  # type: ignore[arg-type]

    output = _tool(tools, "read_file").run(path="/solution/solve.sh")

    assert output.startswith("BLOCKED:")
    assert bridge.commands == []


def test_terminal_tools_execute_shell_through_bridge() -> None:
    bridge = _Bridge()
    tools = make_terminal_tools(bridge)  # type: ignore[arg-type]

    output = _tool(tools, "shell").run(command="pwd", cwd="/app")

    assert "exit=0" in output
    assert "hello" in output
    assert bridge.commands == ["pwd"]


def test_terminal_tools_write_file_uses_base64_payload() -> None:
    bridge = _Bridge()
    tools = make_terminal_tools(bridge)  # type: ignore[arg-type]

    output = _tool(tools, "write_file").run(path="/app/out.txt", content="hello")

    assert "exit=0" in output
    assert "base64 -d > /app/out.txt" in bridge.commands[0]


def test_mcode_terminal_bench_agent_can_initialize_without_harbor(tmp_path) -> None:
    agent = MCodeTerminalBenchAgent(
        logs_dir=tmp_path,
        model_name="model",
        backend_name="openai",
        loop_budget="7",
    )

    assert agent.name() == "mcode"
    assert agent.model_name == "model"
    assert agent.backend_name == "openai"
    assert agent.loop_budget == 7
