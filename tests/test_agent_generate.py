from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch


def test_generate_patch_uses_mellea_aact(tmp_path):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        env=env,
    )

    from mcode.llm.session import LLMSession

    session = LLMSession(model_id="test", backend_name="ollama")

    mock_mellea = MagicMock()
    session._m = mock_mellea

    # aact returns (ModelOutputThunk, Context).
    # Simulate final_answer tool call on first turn.
    mock_tool_result = MagicMock()
    mock_tool_result.name = "final_answer"
    mock_tool_result.content = "done"

    mock_step = MagicMock()
    mock_step.tool_calls = {"final_answer": MagicMock()}
    mock_step.value = ""

    mock_ctx = MagicMock()

    call_count = 0

    async def mock_aact(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return (mock_step, mock_ctx)

    def mock_call_tools(result, backend):
        return [mock_tool_result]

    with (
        patch("mellea.stdlib.functional.aact", mock_aact),
        patch("mellea.stdlib.functional._call_tools", mock_call_tools),
    ):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    assert call_count == 1
    assert isinstance(result, str)
