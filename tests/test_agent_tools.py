from __future__ import annotations

import os
import subprocess

from mcode.llm.repo_state import get_git_diff


def test_diff_after_edits(tmp_path):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "foo.py").write_text("a = 1\nb = 2\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)

    (tmp_path / "foo.py").write_text("a = 1\nb = 42\n")

    patch = get_git_diff(str(tmp_path))
    assert "-b = 2" in patch
    assert "+b = 42" in patch
