from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.llm.session import LLMSession


def _init_repo(tmp_path):
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


def test_generate_patch_returns_verified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = SimpleNamespace(backend=object())

    async def fake_react(goal, context, backend, *, tools, loop_budget, model_options, on_turn):
        del goal, backend, model_options
        on_turn(1, loop_budget, context)
        next(tool for tool in tools if tool.name == "edit").run("foo.py", "x = 1", "x = 2")
        on_turn(2, loop_budget, context)
        next(tool for tool in tools if tool.name == "run_tests").run("default")
        return ("done", context)

    monkeypatch.setattr("mellea.stdlib.frameworks.react.react", fake_react)
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        test_cmds={"test_cmds": [f"{sys.executable} -c \"print('ok')\""]},
    )

    metrics = session.last_patch_metrics
    assert "+x = 2" in patch_text
    assert metrics is not None
    assert metrics["turns_to_first_edit"] == 1
    assert metrics["turns_to_first_verification"] == 2
    assert metrics["verification_succeeded"] is True
    assert metrics["terminal_reason"] == "submitted"


def test_generate_patch_discards_unverified_diff(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = SimpleNamespace(backend=object())

    async def fake_react(goal, context, backend, *, tools, loop_budget, model_options, on_turn):
        del goal, backend, model_options
        on_turn(1, loop_budget, context)
        next(tool for tool in tools if tool.name == "edit").run("foo.py", "x = 1", "x = 2")
        return ("done", context)

    monkeypatch.setattr("mellea.stdlib.frameworks.react.react", fake_react)
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        test_cmds={"test_cmds": [f"{sys.executable} -c \"print('ok')\""]},
    )

    metrics = session.last_patch_metrics
    assert patch_text == ""
    assert metrics is not None
    assert metrics["verification_succeeded"] is False
    assert metrics["terminal_reason"] == "unverified_diff_discarded"


def test_generate_patch_timeout_counts_as_budget_exhausted(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = SimpleNamespace(backend=object())
    monkeypatch.setenv("MCODE_REACT_TIMEOUT", "1")

    async def fake_react(goal, context, backend, *, tools, loop_budget, model_options, on_turn):
        del goal, backend, tools, loop_budget, model_options, on_turn
        await asyncio.sleep(2)
        return ("done", context)

    monkeypatch.setattr("mellea.stdlib.frameworks.react.react", fake_react)
    monkeypatch.setattr("mcode.agent.coding_agent.build_repo_map", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "mcode.agent.coding_agent.build_candidate_files",
        lambda *args, **kwargs: "",
    )

    patch_text = session.generate_patch(
        repo="test/repo",
        problem_statement="Fix the bug",
        repo_root=str(tmp_path),
        test_cmds={"test_cmds": [f"{sys.executable} -c \"print('ok')\""]},
    )

    metrics = session.last_patch_metrics
    assert patch_text == ""
    assert metrics is not None
    assert metrics["terminal_reason"] == "budget_exhausted"


def test_generate_patch_retries_samples_until_verified(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(model_id="test", backend_name="openai", loop_budget=4)
    session._m = SimpleNamespace(backend=object())
    attempts = {"count": 0}

    async def fake_react(goal, context, backend, *, tools, loop_budget, model_options, on_turn):
        del goal, backend, model_options
        attempts["count"] += 1
        on_turn(1, loop_budget, context)
        next(tool for tool in tools if tool.name == "edit").run("foo.py", "x = 1", "x = 2")
        if attempts["count"] == 1:
            return ("done", context)
        on_turn(2, loop_budget, context)
        next(tool for tool in tools if tool.name == "run_tests").run("default")
        return ("done", context)

    monkeypatch.setattr("mellea.stdlib.frameworks.react.react", fake_react)
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
        test_cmds={"test_cmds": [f"{sys.executable} -c \"print('ok')\""]},
    )

    assert attempts["count"] == 2
    assert "+x = 2" in patch_text
    assert session.last_patch_metrics is not None
    assert session.last_patch_metrics["verification_succeeded"] is True


def test_swebench_runner_passes_task_metadata_to_generate_patch(tmp_path):
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

    def fake_generate_patch(**kwargs):
        captured.update(kwargs)
        return ""

    runner.llm.open = lambda: nullcontext()  # type: ignore[method-assign]
    runner.llm.generate_patch = fake_generate_patch  # type: ignore[method-assign]

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
