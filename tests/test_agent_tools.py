from __future__ import annotations

import os
import subprocess

from mcode.agent.tooling import format_tool_result
from mcode.agent.verification import (
    VerificationProgress,
    build_run_tests_tool,
    build_verification_policy,
)
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


def test_run_tests_suppresses_repeat_failed_run_without_edit(tmp_path):
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return "$ pytest\nFAILED\nfailed"

    progress = VerificationProgress()
    policy = build_verification_policy(
        test_cmds={"verification_cmds": ["pytest"]},
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(
        repo_root=str(tmp_path),
        verification_policy=policy,
        progress=progress,
    )
    assert tool is not None

    first = tool.run("default")
    second = tool.run("default")
    progress.note_edit_applied()
    third = tool.run("default")

    assert "FAILED" in first
    assert "SKIPPED" in second
    assert "Previous run_tests already returned FAILED" in second
    assert "Edit the code before rerunning" in second
    assert "FAILED" in third
    assert calls == ["pytest", "pytest"]


def test_format_tool_result_keeps_status_visible_for_multiline_commands():
    command = "python - <<'PY'\n" + "print('x')\n" * 80 + "PY"
    result = format_tool_result(command, "FAILED", "actual failure")
    lines = result.splitlines()

    assert lines[0].startswith("$ python - <<'PY'")
    assert len(lines[0]) <= 302
    assert lines[1] == "FAILED"
    assert lines[2] == "actual failure"



def test_str_replace_edit_allows_small_multi_replace(tmp_path):
    from mcode.agent.tooling import str_replace_edit

    path = tmp_path / "sample.py"
    path.write_text('a = "old"\nb = "old"\n')

    result = str_replace_edit(
        "sample.py",
        "old",
        "new",
        repo_root=str(tmp_path),
    )

    assert "APPLIED" in result
    assert "across 2 occurrences" in result
    assert path.read_text() == 'a = "new"\nb = "new"\n'