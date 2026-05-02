from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcode.agent.verification import build_run_tests_tool, build_verification_policy
from mcode.bench import runner as runner_module
from mcode.bench.results import ResultsDB
from mcode.bench.runner import BenchConfig, BenchmarkRunner
from mcode.execution.sandbox import DockerUnavailableError
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
    assert session.solve_result is not None
    assert session.solve_result.as_metrics_dict() == {
        "terminal_reason": "submitted",
        "turns_to_first_edit": 1,
        "turns_to_first_verification": 2,
        "zero_edit": False,
        "zero_verification": False,
        "verification_succeeded": True,
        "prompt_snapshot": "final prompt",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "provider": "openai",
        "response_model": "test-model",
        "generation_latency_ms": None,
        "validation_passed_count": None,
        "validation_failed_count": None,
    }
    assert session.solve_result.submission == PatchSubmission(
        summary="Updated foo.py",
        tests_ran=["default"],
    )


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
    assert session.solve_result is not None
    assert session.solve_result.verification_succeeded is False
    assert session.solve_result.terminal_reason == "unverified_diff_discarded"
    assert session.solve_result.submission == PatchSubmission(
        summary="Changed foo.py",
        tests_ran=[],
    )


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
    assert session.solve_result is not None
    assert session.solve_result.submission == PatchSubmission(
        summary="Attempt two",
        tests_ran=["default"],
    )


def test_generate_patch_disables_outer_retry_when_sampling_enabled(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(
        model_id="test",
        backend_name="openai",
        loop_budget=4,
        sampling_strategy="multiturn",
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
    assert captured["sampling_strategy_name"] == "multiturn"
    assert captured["sampling_budget"] == 4


def test_selection_attempts_runs_sampling_trajectories_and_selects_best(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(
        model_id="test",
        backend_name="openai",
        loop_budget=4,
        sampling_strategy="multiturn",
        sampling_budget=2,
        selection_attempts=3,
    )
    attempts = {"count": 0}
    captured: list[object] = []

    async def fake_solve_patch(**kwargs):
        captured.append(kwargs["sampling_strategy_name"])
        attempts["count"] += 1
        if attempts["count"] == 1:
            return SolveResult(
                patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
                terminal_reason="budget_exhausted",
                verification_succeeded=True,
                zero_edit=False,
                zero_verification=False,
            )
        if attempts["count"] == 2:
            return SolveResult(
                patch="",
                terminal_reason="budget_exhausted",
                verification_succeeded=False,
            )
        return SolveResult(
            patch="diff --git a/foo.py b/foo.py\n+x = 3\n",
            submission=PatchSubmission(summary="Attempt three", tests_ran=["default"]),
            terminal_reason="submitted",
            verification_succeeded=True,
            zero_edit=False,
            zero_verification=False,
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
        n_samples=1,
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert attempts["count"] == 3
    assert captured == ["multiturn", "multiturn", "multiturn"]
    assert "+x = 3" in patch_text
    assert session.solve_result is not None
    assert session.solve_result.submission == PatchSubmission(
        summary="Attempt three",
        tests_ran=["default"],
    )


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


def test_run_task_retries_docker_unavailable_once(tmp_path):
    results_db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama"),
        results_db=results_db,
    )
    task = SimpleNamespace(repo="test/repo", problem_statement="Fix the bug", hints_text="")
    eval_attempts = {"count": 0}

    @contextmanager
    def repo_context():
        yield SimpleNamespace(
            repo_root=str(tmp_path), command_fn=None, visible_repo_root="/testbed"
        )

    runner._generate_task_patch = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "diff --git a/foo.py b/foo.py\n+x = 2\n",
        None,
    )

    def evaluate_patch(_patch: str) -> runner_module._TaskEvaluation:
        eval_attempts["count"] += 1
        if eval_attempts["count"] == 1:
            raise DockerUnavailableError("socket timed out")
        return runner_module._TaskEvaluation(
            passed=True,
            timed_out=False,
            stdout="ok",
            stderr="",
            error=None,
        )

    result = runner._run_task(
        task_id="task-1",
        generation_task=task,
        repo_context_factory=repo_context,
        evaluate_patch=evaluate_patch,
    )

    assert eval_attempts["count"] == 2
    assert result["passed"] is True
    assert result["attempts_used"] == 2
    assert result["terminal_reason"] == "submitted"


def test_run_task_records_infra_failure_after_docker_retry(tmp_path):
    results_db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama"),
        results_db=results_db,
    )
    task = SimpleNamespace(repo="test/repo", problem_statement="Fix the bug", hints_text="")
    eval_attempts = {"count": 0}

    @contextmanager
    def repo_context():
        yield SimpleNamespace(
            repo_root=str(tmp_path), command_fn=None, visible_repo_root="/testbed"
        )

    runner._generate_task_patch = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "diff --git a/foo.py b/foo.py\n+x = 2\n",
        None,
    )

    def evaluate_patch(_patch: str) -> runner_module._TaskEvaluation:
        eval_attempts["count"] += 1
        raise DockerUnavailableError("socket timed out")

    result = runner._run_task(
        task_id="task-1",
        generation_task=task,
        repo_context_factory=repo_context,
        evaluate_patch=evaluate_patch,
    )

    assert eval_attempts["count"] == 2
    assert result["passed"] is False
    assert result["attempts_used"] == 2
    assert result["terminal_reason"] == "infra_failure"
    assert result["error"] == "DockerUnavailableError: socket timed out"


def test_swebench_runner_skips_completed_tasks_without_touching_sandbox(
    tmp_path, monkeypatch
) -> None:
    task = SimpleNamespace(
        raw_instance={"instance_id": "task-complete"},
        instance_id="task-complete",
        repo="test/repo",
        problem_statement="Fix the bug",
        hints_text="",
    )

    @contextmanager
    def repo_context(_instance):
        yield SimpleNamespace(repo_root=str(tmp_path), command_fn=None, visible_repo_root=None)

    class FakeRun:
        resolved = True
        timed_out = False
        test_output = "ok"
        report = {"task-complete": {"resolved": True}}

    class FirstSandbox:
        def __init__(self, **kwargs) -> None:
            pass

        def prepare_images(self, instances):
            assert instances == [task.raw_instance]

        def repo_context(self, instance):
            return repo_context(instance)

        def evaluate_patch(self, **kwargs):
            return FakeRun()

    class ForbiddenSandbox:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("completed rerun should not touch sandbox")

    monkeypatch.setattr(runner_module, "_runtime_metadata", lambda: {})
    monkeypatch.setattr(
        "mcode.bench.swebench_lite.load_swebench_lite",
        lambda *args, **kwargs: [task],
    )
    monkeypatch.setattr("mcode.execution.swebench.SWEbenchSandbox", FirstSandbox)

    results_db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama"),
        results_db=results_db,
    )
    runner.llm.check_available = lambda: None  # type: ignore[method-assign]
    runner._generate_task_patch = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "diff --git a/foo.py b/foo.py\n+x = 2\n",
        None,
    )

    first = runner.run_benchmark("swebench-lite")
    assert first.total == 1
    assert first.passed == 1

    monkeypatch.setattr("mcode.execution.swebench.SWEbenchSandbox", ForbiddenSandbox)
    second = runner.run_benchmark("swebench-lite")

    assert second.run_id == first.run_id
    assert second.total == 1
    assert second.passed == 1


def test_swebench_runner_retries_prior_infra_failure(tmp_path, monkeypatch) -> None:
    task = SimpleNamespace(
        raw_instance={"instance_id": "task-retry"},
        instance_id="task-retry",
        repo="test/repo",
        problem_statement="Fix the bug",
        hints_text="",
    )
    sandbox_calls = {"prepare": 0, "evaluate": 0, "generate": 0}

    @contextmanager
    def repo_context(_instance):
        yield SimpleNamespace(repo_root=str(tmp_path), command_fn=None, visible_repo_root=None)

    class FakeRun:
        resolved = True
        timed_out = False
        test_output = "ok"
        report = {"task-retry": {"resolved": True}}

    class FirstSandbox:
        def __init__(self, **kwargs) -> None:
            pass

        def prepare_images(self, instances):
            assert instances == [task.raw_instance]
            raise RuntimeError("unpacking failed: Chown error detected")

    class SecondSandbox:
        def __init__(self, **kwargs) -> None:
            pass

        def prepare_images(self, instances):
            sandbox_calls["prepare"] += 1
            assert instances == [task.raw_instance]

        def repo_context(self, instance):
            return repo_context(instance)

        def evaluate_patch(self, **kwargs):
            sandbox_calls["evaluate"] += 1
            return FakeRun()

    monkeypatch.setattr(runner_module, "_runtime_metadata", lambda: {})
    monkeypatch.setattr(
        "mcode.bench.swebench_lite.load_swebench_lite",
        lambda *args, **kwargs: [task],
    )

    results_db = ResultsDB(tmp_path / "results.db")
    runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama"),
        results_db=results_db,
    )
    runner.llm.check_available = lambda: None  # type: ignore[method-assign]

    def generate_patch(*args, **kwargs):
        del args, kwargs
        sandbox_calls["generate"] += 1
        return "diff --git a/foo.py b/foo.py\n+x = 2\n", None

    runner._generate_task_patch = generate_patch  # type: ignore[method-assign]

    monkeypatch.setattr("mcode.execution.swebench.SWEbenchSandbox", FirstSandbox)
    first = runner.run_benchmark("swebench-lite")
    assert first.total == 1
    assert first.passed == 0
    first_rows = results_db.task_terminal_rows(first.run_id)
    assert first_rows[task.instance_id]["terminal_reason"] == "infra_failure"
    assert sandbox_calls["generate"] == 0

    monkeypatch.setattr("mcode.execution.swebench.SWEbenchSandbox", SecondSandbox)
    second = runner.run_benchmark("swebench-lite")

    assert second.run_id == first.run_id
    assert second.total == 1
    assert second.passed == 1
    assert sandbox_calls == {"prepare": 1, "evaluate": 1, "generate": 1}
    second_rows = results_db.task_terminal_rows(second.run_id)
    assert second_rows[task.instance_id]["terminal_reason"] == "submitted"


def test_solve_result_carries_diagnostic_events_only_when_enabled(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    session = LLMSession(
        model_id="test",
        backend_name="openai",
        loop_budget=4,
        diagnostic_traces=True,
    )
    captured: dict[str, object] = {}

    async def fake_solve_patch(**kwargs):
        collector = kwargs["collector"]
        captured["diagnostic_enabled"] = collector.diagnostic_enabled
        collector.note_event("turn_start", {"turn": 1}, turn=1)
        return SolveResult(
            patch="",
            terminal_reason="budget_exhausted",
            diagnostic_events=list(collector.diagnostic_events),
        )

    monkeypatch.setattr("mcode.llm.session.hooks_available", lambda: True)
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
        test_cmds={"test_cmds": ["python -c pass"]},
    )

    assert captured["diagnostic_enabled"] is True
    assert session.solve_result is not None
    assert session.solve_result.diagnostic_events == [
        {"turn": 1, "event_type": "turn_start", "payload": {"turn": 1}}
    ]

    disabled_result = SolveResult().as_metrics_dict()
    assert "diagnostic_events" not in disabled_result


def test_run_task_generate_then_evaluate_uses_artifacts(tmp_path):
    results_db = ResultsDB(tmp_path / "results.db")
    task = SimpleNamespace(
        benchmark="swebench-lite",
        instance_id="task-artifact",
        repo="test/repo",
        problem_statement="Fix the bug",
        hints_text="",
        raw_instance={"instance_id": "task-artifact", "test_cmds": ["pytest -q"]},
    )

    @contextmanager
    def repo_context():
        yield SimpleNamespace(
            repo_root=str(tmp_path),
            command_fn=None,
            visible_repo_root="/testbed",
        )

    generate_runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama", phase="generate"),
        results_db=results_db,
    )
    generate_runner._generate_task_patch = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "diff --git a/foo.py b/foo.py\n+x = 2\n",
        {
            "terminal_reason": "submitted",
            "turns_to_first_edit": 1,
            "turns_to_first_verification": 2,
            "zero_edit": False,
            "zero_verification": False,
            "verification_succeeded": True,
            "submission_json": '{"summary":"done"}',
        },
    )
    generate_run_id = results_db.start_run(
        "swebench-lite",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "loop_budget": 15,
            "timeout_s": 60,
            "phase": "generate",
        },
    )
    generated = generate_runner._run_task(
        run_id=generate_run_id,
        task_id=task.instance_id,
        generation_task=task,
        repo_context_factory=repo_context,
        evaluate_patch=lambda _patch: (_ for _ in ()).throw(AssertionError("no eval")),
    )
    assert generated is None
    artifact_rows = results_db.task_artifact_rows(generate_run_id)
    assert artifact_rows[task.instance_id]["candidate_count"] == 1

    evaluate_runner = BenchmarkRunner(
        config=BenchConfig(model_id="test-model", backend_name="ollama", phase="evaluate"),
        results_db=results_db,
    )
    evaluate_runner._generate_task_patch = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("solver should not run in evaluate phase")
    )
    evaluate_run_id = results_db.start_run(
        "swebench-lite",
        {
            "backend_name": "ollama",
            "model_id": "test-model",
            "loop_budget": 15,
            "timeout_s": 60,
            "phase": "evaluate",
        },
    )
    seen: dict[str, str] = {}

    def evaluate_patch(patch: str) -> runner_module._TaskEvaluation:
        seen["patch"] = patch
        return runner_module._TaskEvaluation(
            passed=True,
            timed_out=False,
            stdout="ok",
            stderr="",
            error=None,
            evaluator_name="swebench-lite",
            report={"resolved": True},
        )

    result = evaluate_runner._run_task(
        run_id=evaluate_run_id,
        task_id=task.instance_id,
        generation_task=task,
        repo_context_factory=repo_context,
        evaluate_patch=evaluate_patch,
    )
    assert result is not None
    assert result["passed"] is True
    assert "+x = 2" in seen["patch"]
    eval_count = results_db.conn.execute(
        "SELECT COUNT(*) FROM artifact_evaluations WHERE run_id = ?",
        (evaluate_run_id,),
    ).fetchone()[0]
    assert eval_count == 1