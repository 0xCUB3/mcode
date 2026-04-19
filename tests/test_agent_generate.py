from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcode.agent.verification import build_run_tests_tool, build_verification_policy
from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.llm.session import LLMSession, PatchSubmission, SolveResult, _coerce_submission


def _init_repo(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        env=env,
    )


@contextmanager
def _session_context(fake_session):
    yield fake_session


def test_generate_patch_returns_verified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)

    async def fake_solve_patch(**kwargs):
        del kwargs
        return SolveResult(
            patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
            submission=PatchSubmission(summary="Updated foo.py", tests_ran=["default"]),
            terminal_reason="submitted",
            turns_to_first_edit=1,
            turns_to_first_verification=2,
            zero_edit=False,
            zero_verification=False,
            verification_succeeded=True,
            prompt_snapshot="final prompt",
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            provider="openai",
            response_model="test-model",
        )

    monkeypatch.setattr(
        session,
        "_start_session",
        lambda **kwargs: _session_context(
            SimpleNamespace(backend=object(), solve_patch=fake_solve_patch)
        ),
    )
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert "+x = 2" in patch_text
    assert session.last_patch_metrics == {
        "turns_to_first_edit": 1,
        "turns_to_first_verification": 2,
        "zero_edit": False,
        "zero_verification": False,
        "verification_succeeded": True,
        "terminal_reason": "submitted",
    }
    assert session.last_submission == {
        "summary": "Updated foo.py",
        "tests_ran": ["default"],
    }
    assert session.last_generation_trace == {
        "prompt_snapshot": "final prompt",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "provider": "openai",
        "response_model": "test-model",
    }


def test_generate_patch_discards_unverified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)

    async def fake_solve_patch(**kwargs):
        del kwargs
        return SolveResult(
            patch="",
            submission=PatchSubmission(summary="Changed foo.py", tests_ran=[]),
            terminal_reason="unverified_diff_discarded",
            verification_succeeded=False,
            prompt_snapshot='[{"content": "final prompt", "role": "assistant"}]',
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            provider="openai",
            response_model="test-model",
        )

    monkeypatch.setattr(
        session,
        "_start_session",
        lambda **kwargs: _session_context(
            SimpleNamespace(backend=object(), solve_patch=fake_solve_patch)
        ),
    )
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert patch_text == ""
    assert session.last_patch_metrics is not None
    assert session.last_patch_metrics["verification_succeeded"] is False
    assert session.last_patch_metrics["terminal_reason"] == "unverified_diff_discarded"
    assert session.last_submission == {
        "summary": "Changed foo.py",
        "tests_ran": [],
    }


def test_coerce_submission_accepts_json_string():
    submission = _coerce_submission('{"summary":"done","tests_ran":["default"]}')

    assert submission == PatchSubmission(summary="done", tests_ran=["default"])


def test_coerce_submission_falls_back_to_plain_summary():
    submission = _coerce_submission("done")

    assert submission == PatchSubmission(summary="done", tests_ran=[])


def test_generate_patch_retries_outer_samples_until_verified(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    attempts = {"count": 0}

    async def fake_solve_patch(**kwargs):
        del kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            return SolveResult(
                patch="",
                submission=PatchSubmission(summary="Attempt one", tests_ran=[]),
                terminal_reason="unverified_diff_discarded",
                verification_succeeded=False,
            )
        return SolveResult(
            patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
            submission=PatchSubmission(summary="Attempt two", tests_ran=["default"]),
            terminal_reason="submitted",
            turns_to_first_edit=1,
            turns_to_first_verification=2,
            zero_edit=False,
            zero_verification=False,
            verification_succeeded=True,
            prompt_snapshot="retry prompt",
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
            provider="openai",
            response_model="test-model",
        )

    monkeypatch.setattr(
        session,
        "_start_session",
        lambda **kwargs: _session_context(
            SimpleNamespace(backend=object(), solve_patch=fake_solve_patch)
        ),
    )
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        n_samples=2,
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert attempts["count"] == 2
    assert "+x = 2" in patch_text
    assert session.last_submission == {
        "summary": "Attempt two",
        "tests_ran": ["default"],
    }


def test_generate_patch_disables_outer_retry_when_sampling_enabled(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(
        model_id="test",
        backend_name="openai",
        loop_budget=4,
        sampling_strategy="rejection",
    )
    attempts = {"count": 0}
    captured: dict[str, object] = {}

    async def fake_solve_patch(**kwargs):
        captured.update(kwargs)
        attempts["count"] += 1
        return SolveResult(
            patch="",
            submission=PatchSubmission(summary="Attempt one", tests_ran=[]),
            terminal_reason="budget_exhausted",
            verification_succeeded=False,
        )

    monkeypatch.setattr(
        session,
        "_start_session",
        lambda **kwargs: _session_context(
            SimpleNamespace(backend=object(), solve_patch=fake_solve_patch)
        ),
    )
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        n_samples=4,
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert attempts["count"] == 1
    assert captured["sampling_strategy_name"] == "rejection"
    assert captured["sampling_budget"] == 4


def test_generate_patch_retries_from_clean_repo_snapshot(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    seen_states: list[tuple[str, bool]] = []
    attempts = {"count": 0}

    async def fake_solve_patch(**kwargs):
        repo_root = Path(kwargs["repo_root"])
        del kwargs
        attempts["count"] += 1
        seen_states.append(((repo_root / "foo.py").read_text(), (repo_root / "extra.txt").exists()))
        if attempts["count"] == 1:
            (repo_root / "foo.py").write_text("x = 99\n")
            (repo_root / "extra.txt").write_text("leftover\n")
            return SolveResult(
                patch="",
                submission=PatchSubmission(summary="Attempt one", tests_ran=[]),
                terminal_reason="unverified_diff_discarded",
                verification_succeeded=False,
            )
        return SolveResult(
            patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
            submission=PatchSubmission(summary="Attempt two", tests_ran=["default"]),
            terminal_reason="submitted",
            verification_succeeded=True,
        )

    monkeypatch.setattr(
        session,
        "_start_session",
        lambda **kwargs: _session_context(
            SimpleNamespace(backend=object(), solve_patch=fake_solve_patch)
        ),
    )
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        n_samples=2,
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert attempts["count"] == 2
    assert seen_states == [("x = 1\n", False), ("x = 1\n", False)]
    assert "+x = 2" in patch_text


def test_run_tests_tool_schema_only_requires_test_cmd():
    tool = build_run_tests_tool(
        repo_root=".",
        verification_policy=build_verification_policy(test_cmds=["pytest -q"]),
    )

    assert tool is not None
    params = tool.as_json_tool["function"]["parameters"]
    assert params["required"] == ["test_cmd"]
    assert params["properties"]["timeout_s"]["type"] == "integer"
    assert params["properties"]["max_output_chars"]["type"] == "integer"


def test_swebench_runner_passes_task_metadata_to_solver(tmp_path):
    results_db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama"),
        results_db=results_db,
    )

    task = MagicMock()
    task.repo = "test/repo"
    task.problem_statement = "Fix the bug"
    task.hints_text = "Hint"
    task.raw_instance = {"test_cmds": ["pytest -q tests/test_bug.py"]}

    captured: dict[str, object] = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        return SolveResult()

    runner.llm.solve = fake_solve  # type: ignore[method-assign]

    patch, metrics = runner._generate_task_patch(
        task,
        repo_root=str(tmp_path),
        command_fn=lambda command: command,
        visible_repo_root="/testbed",
    )

    assert patch == ""
    assert metrics is None
    assert captured["test_cmds"] == {"test_cmds": ["pytest -q tests/test_bug.py"]}
    assert captured["command_fn"] is not None
    assert captured["visible_repo_root"] == "/testbed"
