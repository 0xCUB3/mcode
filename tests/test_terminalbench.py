from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcode.bench.results import ResultsDB
from mcode.bench.terminalbench import (
    MCODE_AGENT_IMPORT_PATH,
    TerminalBenchConfig,
    build_harbor_command,
    import_harbor_job,
    normalize_task_id,
    trial_to_task_result,
)


def test_build_harbor_command_for_mcode_agent(tmp_path: Path) -> None:
    config = TerminalBenchConfig(
        model_id="Qwen/Qwen3",
        backend_name="openai",
        agent="mcode",
        jobs_dir=tmp_path / "jobs",
        job_name="tb-smoke",
        n_concurrent=2,
        timeout_multiplier=1.5,
    )

    cmd = build_harbor_command(config, limit=3, task_ids=["log-summary-date-ranges"])

    assert cmd[:4] == ["harbor", "run", "-d", "terminal-bench/terminal-bench-2"]
    assert _option(cmd, "--job-name") == "tb-smoke"
    assert _option(cmd, "--n-concurrent") == "2"
    assert _option(cmd, "--timeout-multiplier") == "1.5"
    assert _option(cmd, "--n-tasks") == "3"
    assert _option(cmd, "--include-task-name") == "terminal-bench/log-summary-date-ranges"
    assert _option(cmd, "--agent-import-path") == MCODE_AGENT_IMPORT_PATH
    assert _option(cmd, "--model") == "Qwen/Qwen3"
    assert "--agent" not in cmd


def test_build_harbor_command_for_builtin_oracle(tmp_path: Path) -> None:
    config = TerminalBenchConfig(
        model_id="ignored-for-oracle",
        agent="oracle",
        jobs_dir=tmp_path / "jobs",
    )

    cmd = build_harbor_command(config, limit=1)

    assert _option(cmd, "--agent") == "oracle"
    assert "--model" not in cmd
    assert "--agent-import-path" not in cmd


def test_trial_to_task_result_reads_reward_and_logs(tmp_path: Path) -> None:
    trial_dir = _write_trial(
        tmp_path,
        task_name="terminal-bench/log-summary-date-ranges",
        reward=1,
        exception=None,
    )
    trial = json.loads((trial_dir / "result.json").read_text())

    result = trial_to_task_result(trial, trial_dir=trial_dir)

    assert result["task_id"] == "log-summary-date-ranges"
    assert result["passed"] is True
    assert result["verification_succeeded"] is True
    assert result["terminal_reason"] == "submitted"
    assert result["time_ms"] == 2000
    assert result["stdout"] == "ok\n"
    assert result["stderr"] == ""


def test_trial_to_task_result_marks_timeout(tmp_path: Path) -> None:
    trial_dir = _write_trial(
        tmp_path,
        task_name="terminal-bench/slow-task",
        reward=0,
        exception={
            "exception_type": "AgentTimeoutError",
            "exception_message": "Agent execution timed out",
        },
    )
    trial = json.loads((trial_dir / "result.json").read_text())

    result = trial_to_task_result(trial, trial_dir=trial_dir)

    assert result["task_id"] == "slow-task"
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["terminal_reason"] == "timeout"
    assert result["error"] == "Agent execution timed out"


def test_import_harbor_job_saves_results_and_artifacts(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "tb-job"
    trial_dir = _write_trial(job_dir, task_name="terminal-bench/example", reward=1)
    db = ResultsDB(tmp_path / "results.db")
    config = TerminalBenchConfig(
        model_id="model",
        backend_name="openai",
        jobs_dir=tmp_path / "jobs",
        job_name="tb-job",
        artifact_dir=tmp_path / "artifacts",
    )
    run_id = db.start_run("terminal-bench", config.run_config(limit=1, task_ids=None))

    imported = import_harbor_job(job_dir=job_dir, results_db=db, run_id=run_id, config=config)

    assert [item.task_id for item in imported] == ["example"]
    summary = db.run_summary(run_id)
    assert summary.total == 1
    assert summary.passed == 1
    assert imported[0].manifest_path.is_file()
    rows = db.task_artifact_rows(run_id)
    assert rows["example"]["manifest_path"] == str(imported[0].manifest_path)
    assert trial_dir.is_dir()


def test_normalize_task_id_strips_terminal_bench_prefix() -> None:
    assert normalize_task_id("terminal-bench/foo") == "foo"
    assert normalize_task_id("foo") == "foo"


def _option(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    index = cmd.index(flag)
    return cmd[index + 1]


def _write_trial(
    root: Path,
    *,
    task_name: str,
    reward: int,
    exception: dict[str, str] | None = None,
) -> Path:
    trial_dir = root / task_name.rsplit("/", 1)[-1]
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "reward.txt").write_text(str(reward), encoding="utf-8")
    (verifier_dir / "test-stdout.txt").write_text("ok\n", encoding="utf-8")
    (verifier_dir / "test-stderr.txt").write_text("", encoding="utf-8")
    started = datetime(2026, 1, 1, tzinfo=UTC)
    finished = started + timedelta(seconds=2)
    trial = {
        "task_name": task_name,
        "trial_name": task_name.rsplit("/", 1)[-1],
        "trial_uri": str(trial_dir),
        "task_checksum": "abc123",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "agent_info": {
            "name": "mcode",
            "version": "unknown",
            "model_info": {"provider": "openai", "name": "model"},
        },
        "agent_result": {
            "n_input_tokens": 10,
            "n_output_tokens": 5,
        },
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": exception,
    }
    (trial_dir / "result.json").write_text(json.dumps(trial), encoding="utf-8")
    return trial_dir
