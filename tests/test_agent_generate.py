from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch


def test_generate_patch_calls_react(tmp_path):
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

    mock_thunk = MagicMock()
    mock_thunk.value = "done"

    with patch("mcode.llm.session.asyncio") as mock_asyncio:
        mock_asyncio.run.return_value = (mock_thunk, MagicMock())
        result = session.generate_patch(
            repo="test/repo",
            problem_statement="Fix the bug",
            repo_root=str(tmp_path),
        )
    mock_asyncio.run.assert_called_once()
    assert isinstance(result, str)
