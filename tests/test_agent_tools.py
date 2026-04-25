from __future__ import annotations

import os
import subprocess

from mcode.agent.tooling import build_candidate_files, suggest_verification_commands
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


def test_command_fn_only_rejects_default_and_allows_concrete_command(tmp_path):
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return "$ python -m pytest\nPASSED\nok"

    policy = build_verification_policy(
        command_fn=command_fn,
        suggested_test_cmds=["python -m pytest"],
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)
    assert tool is not None

    default_result = tool.run("default")
    concrete_result = tool.run("python -m pytest")

    assert policy.allow_default_test_cmd is False
    assert "no default verifier" in policy.prompt_block.lower()
    assert "python -m pytest" in policy.prompt_block
    assert "REJECTED" in default_result
    assert "No default verifier is available" in default_result
    assert "PASSED" in concrete_result
    assert calls == ["python -m pytest"]


def test_declared_commands_keep_default_verifier(tmp_path):
    calls: list[str] = []

    def command_fn(command: str) -> str:
        calls.append(command)
        return "$ pytest\nPASSED\nok"

    policy = build_verification_policy(
        test_cmds=["pytest"],
        command_fn=command_fn,
    )
    tool = build_run_tests_tool(repo_root=str(tmp_path), verification_policy=policy)
    assert tool is not None

    result = tool.run("default")

    assert policy.allow_default_test_cmd is True
    assert "test_cmd=\"default\"" in policy.prompt_block
    assert "PASSED" in result
    assert calls == ["pytest"]


def test_suggest_verification_commands_from_visible_repo_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tox.ini").write_text("[tox]\nenvlist = py\n[testenv:py]\n")
    (tmp_path / "runtests.py").write_text("print('run')\n")

    suggestions = suggest_verification_commands(str(tmp_path))

    assert suggestions[:3] == ["tox -e py", "python runtests.py", "python -m pytest"]


def test_candidate_files_rank_symbol_matches_above_path_only_matches(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "payment_timeout_notes.py").write_text("TIMEOUT = 30\n")
    (tmp_path / "src" / "processor.py").write_text(
        "class PaymentProcessor:\n"
        "    def calculate_refund_total(self, items):\n"
        "        return sum(items)\n"
    )

    result = build_candidate_files(
        str(tmp_path),
        "Refund bug in PaymentProcessor.calculate_refund_total during checkout timeout",
        top_n=2,
    )

    lines = result.splitlines()
    assert "src/processor.py" in lines[1]
    assert "symbol match" in lines[1]
    assert any("calculate_refund_total" in line for line in lines)