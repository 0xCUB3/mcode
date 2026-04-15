from __future__ import annotations

import subprocess
import sys

from mellea.agent.capabilities import OrchestratorContract
from mellea.agent.localization import format_candidate_files
from mellea.agent.strategy import ToolInvocation, ToolPhaseState, get_available_tools


def test_installed_mellea_exposes_required_agent_contract(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "parser.py").write_text("def parse_error_report():\n    return None\n")

    contract = OrchestratorContract.from_tool_names(
        ["search_code", "edit", "run_tests", "final_answer"],
        default_verification_commands=["pytest -q"],
    )
    phase_state = ToolPhaseState(
        turn=2,
        budget=10,
        invocations=(ToolInvocation("search_code"), ToolInvocation("read_file")),
    )

    assert contract.route_for_tool("run_tests").requested_family == "verification"
    assert contract.snapshot()["verification_required"] is True
    assert "search_code" in get_available_tools(
        ["search_code", "read_file", "edit", "run_tests", "final_answer"],
        turn=phase_state.turn,
        budget=phase_state.budget,
        state=phase_state,
    )
    assert "pkg/parser.py" in format_candidate_files(str(tmp_path), "parse error report")


def test_installed_mellea_text_react_accepts_mcode_kwargs():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inspect; "
                "from mellea.agent.text_react import text_react; "
                "params = inspect.signature(text_react).parameters; "
                "assert 'tool_gate' in params; "
                "assert 'condensation' in params"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
