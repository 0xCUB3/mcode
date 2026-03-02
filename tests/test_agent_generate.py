from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch


def test_generate_patch_uses_lean_loop(tmp_path):
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

    # Mock aact to return a step with a final_answer tool call
    mock_step = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.call_func.return_value = "done"
    mock_step.tool_calls = {"final_answer": mock_tool_call}

    mock_ctx = MagicMock()

    async def mock_aact(*args, **kwargs):
        return (mock_step, mock_ctx)

    with patch("mellea.stdlib.functional.aact", mock_aact):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    assert isinstance(result, str)
