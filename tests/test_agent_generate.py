from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch


def test_generate_patch_calls_ollama(tmp_path):
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
    mock_mellea.backend._base_url = None
    mock_mellea.backend._get_ollama_model_id.return_value = "test"
    session._m = mock_mellea

    # Model calls final_answer on first turn
    mock_tc = MagicMock()
    mock_tc.function.name = "final_answer"
    mock_tc.function.arguments = {"summary": "done"}

    mock_resp = MagicMock()
    mock_resp.message.tool_calls = [mock_tc]
    mock_resp.message.content = ""
    mock_resp.message.model_dump.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
    }

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_resp

    with patch("ollama.Client", return_value=mock_client):
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    mock_client.chat.assert_called_once()
    assert isinstance(result, str)
