from __future__ import annotations

import io
import re
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.results import ResultsDB
from mcode.cli import (
    SHARDED_INFRA_EXIT_CODE,
    _latest_run_summary,
    _run_sharded_benchmark,
    app,
)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class _FakePopen:
    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_count = int(args[args.index("--shard-count") + 1])
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        db.parent.mkdir(parents=True, exist_ok=True)

        with ResultsDB(db) as rdb:
            run_id = rdb.start_run(
                "swebench-lite",
                {
                    "backend_name": "ollama",
                    "model_id": "test-model",
                    "loop_budget": 15,
                    "timeout_s": 300,
                    "task_shard_count": shard_count,
                    "task_shard_index": shard_index,
                    "cache_dir": str(db.parent / "cache"),
                },
            )
            rdb.save_task_result(
                run_id,
                {
                    "task_id": f"task-{shard_index}",
                    "passed": shard_index % 2 == 0,
                    "attempts_used": 1,
                    "time_ms": 1000 + shard_index,
                    "exit_code": 0,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "code_sha256": f"sha-{shard_index}",
                },
            )

        self.stdout = io.StringIO(f"shard {shard_index}/{shard_count}\n")
        self.returncode = 0
        self.pid = 100000 + shard_index

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 130


def test_run_sharded_benchmark_merges_into_existing_db(tmp_path: Path, monkeypatch) -> None:
    out_db = tmp_path / "results.db"
    with ResultsDB(out_db) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "ollama",
                "model_id": "existing-model",
                "loop_budget": 3,
                "timeout_s": 60,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": "existing-task",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 1,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "existing",
            },
        )

    monkeypatch.setattr("mcode.cli.subprocess.Popen", _FakePopen)

    _run_sharded_benchmark(
        command="swebench-lite",
        base_argv=["--model", "test-model"],
        shards=2,
        db=out_db,
        benchmark="swebench-lite",
        backend="ollama",
        model="test-model",
        loop_budget=15,
        timeout_s=300,
    )

    summary = _latest_run_summary(out_db)
    assert summary.total == 2
    assert summary.passed == 1

    with ResultsDB(out_db) as rdb:
        runs = rdb.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert runs == 2

    shard_dirs = list((tmp_path / "results-shards").iterdir())
    assert len(shard_dirs) == 1
    assert sorted(path.name for path in shard_dirs[0].iterdir()) == [
        "results-shard-0.db",
        "results-shard-0.log",
        "results-shard-1.db",
        "results-shard-1.log",
    ]


def _write_shard_result(
    db: Path,
    *,
    task_id: str,
    terminal_reason: str | None,
    error: str | None,
    passed: bool = False,
) -> None:
    with ResultsDB(db) as rdb:
        run_id = rdb.start_run(
            "swebench-lite",
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 15,
                "timeout_s": 300,
                "cache_dir": str(db.parent / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": task_id,
                "passed": passed,
                "attempts_used": 1,
                "time_ms": 1,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": error,
                "code_sha256": None,
                "terminal_reason": terminal_reason,
            },
        )


class _PartialFakePopen:
    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        db.parent.mkdir(parents=True, exist_ok=True)
        _write_shard_result(
            db,
            task_id=f"task-{shard_index}",
            terminal_reason="submitted" if shard_index == 1 else "infra_failure",
            error=None if shard_index == 1 else "podman socket did not come up",
            passed=shard_index == 1,
        )
        self.stdout = io.StringIO("")
        self.returncode = 7 if shard_index == 0 else 0
        self.pid = 210000 + shard_index

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 130


def test_run_sharded_benchmark_merges_nonzero_shard_with_rows(tmp_path: Path, monkeypatch) -> None:
    out_db = tmp_path / "results.db"
    monkeypatch.setattr("mcode.cli.subprocess.Popen", _PartialFakePopen)

    _run_sharded_benchmark(
        command="swebench-lite",
        base_argv=["--model", "test-model"],
        shards=2,
        db=out_db,
        benchmark="swebench-lite",
        backend="ollama",
        model="test-model",
        loop_budget=15,
        timeout_s=300,
    )

    summary = _latest_run_summary(out_db)
    assert summary.total == 2
    assert summary.passed == 1


class _RelaunchFakePopen:
    attempts: dict[int, int] = {}

    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        db.parent.mkdir(parents=True, exist_ok=True)
        attempt = self.attempts.get(shard_index, 0) + 1
        self.attempts[shard_index] = attempt
        if shard_index != 0 or attempt > 1:
            _write_shard_result(
                db,
                task_id=f"task-{shard_index}",
                terminal_reason="submitted",
                error=None,
                passed=True,
            )
        self.stdout = io.StringIO("")
        self.returncode = SHARDED_INFRA_EXIT_CODE if shard_index == 0 and attempt == 1 else 0
        self.pid = 220000 + shard_index + (attempt * 10)

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 130


def test_run_sharded_benchmark_relaunches_empty_infra_shard_once(
    tmp_path: Path, monkeypatch
) -> None:
    _RelaunchFakePopen.attempts = {}
    out_db = tmp_path / "results.db"
    monkeypatch.setattr("mcode.cli.subprocess.Popen", _RelaunchFakePopen)

    _run_sharded_benchmark(
        command="swebench-lite",
        base_argv=["--model", "test-model"],
        shards=2,
        db=out_db,
        benchmark="swebench-lite",
        backend="ollama",
        model="test-model",
        loop_budget=15,
        timeout_s=300,
    )

    assert _RelaunchFakePopen.attempts == {0: 2, 1: 1}
    summary = _latest_run_summary(out_db)
    assert summary.total == 2
    assert summary.passed == 2


def test_cli_rejects_auto_and_manual_shards_together() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bench",
            "swebench-lite",
            "--model",
            "test-model",
            "--shards",
            "2",
            "--shard-count",
            "2",
        ],
        color=False,
    )
    assert res.exit_code != 0
    output = _strip_ansi(res.output)
    assert "--shards cannot be combined with --shard-count/--shard-index" in output


def test_swebench_lite_shards_forwards_phase_artifact_dir(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_sharded_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_sharded_benchmark", fake_run_sharded_benchmark)

    res = runner.invoke(
        app,
        [
            "bench",
            "swebench-lite",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "lite.db"),
            "--shards",
            "2",
            "--phase",
            "generate",
            "--artifact-dir",
            str(tmp_path / "stable-artifacts"),
        ],
    )

    assert res.exit_code == 0
    assert "--phase" in captured["base_argv"]
    assert captured["base_argv"][captured["base_argv"].index("--phase") + 1] == "generate"
    assert "--artifact-dir" in captured["base_argv"]
    assert (
        captured["base_argv"][captured["base_argv"].index("--artifact-dir") + 1]
        == str(tmp_path / "stable-artifacts")
    )


def test_smoke_bluevela_forwards_shard_args(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_bench_on_bluevela(*, bench_argv, model, local_db, fetch_db, fetch_artifacts):
        captured["bench_argv"] = bench_argv
        captured["model"] = model
        captured["local_db"] = local_db
        captured["fetch_db"] = fetch_db
        captured["fetch_artifacts"] = fetch_artifacts
        return 0

    monkeypatch.setattr("mcode.bench.remote.run_bench_on_bluevela", fake_run_bench_on_bluevela)

    res = runner.invoke(
        app,
        [
            "bench",
            "smoke",
            "--model",
            "test-model",
            "--on",
            "bluevela",
            "--db",
            str(tmp_path / "smoke.db"),
            "--shards",
            "4",
        ],
    )

    assert res.exit_code == 0
    assert captured["bench_argv"] == [
        "smoke",
        "--model",
        "test-model",
        "--backend",
        "openai",
        "--mem-limit",
        "8g",
        "--phase",
        "run",
        "--artifact-dir",
        str(tmp_path / "smoke" / "artifacts"),
        "--no-check-image-digests",
        "--shards",
        "4",
    ]


def test_swebench_lite_bluevela_forwards_args(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_bench_on_bluevela(*, bench_argv, model, local_db, fetch_db, fetch_artifacts):
        captured["bench_argv"] = bench_argv
        captured["model"] = model
        captured["local_db"] = local_db
        captured["fetch_db"] = fetch_db
        captured["fetch_artifacts"] = fetch_artifacts
        return 0

    monkeypatch.setattr("mcode.bench.remote.run_bench_on_bluevela", fake_run_bench_on_bluevela)

    res = runner.invoke(
        app,
        [
            "bench",
            "swebench-lite",
            "--model",
            "test-model",
            "--backend",
            "openai",
            "--on",
            "bluevela",
            "--db",
            str(tmp_path / "lite.db"),
            "--task-ids",
            "task-1,task-2",
            "--selection-attempts",
            "3",
            "--phase",
            "evaluate",
            "--artifact-dir",
            str(tmp_path / "lite-artifacts"),
            "--shards",
            "2",
        ],
    )

    assert res.exit_code == 0
    assert captured["bench_argv"] == [
        "swebench-lite",
        "--model",
        "test-model",
        "--backend",
        "openai",
        "--loop-budget",
        "15",
        "--timeout",
        "1800",
        "--split",
        "test",
        "--arch",
        "auto",
        "--namespace",
        "swebench",
        "--max-workers",
        "4",
        "--mem-limit",
        "4g",
        "--pids-limit",
        "512",
        "--n-samples",
        "1",
        "--sampling",
        "none",
        "--dataset",
        "SWE-bench/SWE-bench_Lite",
        "--phase",
        "evaluate",
        "--artifact-dir",
        str(tmp_path / "lite-artifacts"),
        "--selection-attempts",
        "3",
        "--task-ids",
        "task-1,task-2",
        "--no-check-image-digests",
        "--shards",
        "2",
    ]


def test_swebench_live_bluevela_forwards_args(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_bench_on_bluevela(*, bench_argv, model, local_db, fetch_db, fetch_artifacts):
        captured["bench_argv"] = bench_argv
        captured["model"] = model
        captured["local_db"] = local_db
        captured["fetch_db"] = fetch_db
        captured["fetch_artifacts"] = fetch_artifacts
        return 0

    monkeypatch.setattr("mcode.bench.remote.run_bench_on_bluevela", fake_run_bench_on_bluevela)

    res = runner.invoke(
        app,
        [
            "bench",
            "swebench-live",
            "--model",
            "test-model",
            "--on",
            "bluevela",
            "--db",
            str(tmp_path / "live.db"),
            "--limit",
            "3",
            "--phase",
            "run",
            "--fetch-db",
        ],
    )

    assert res.exit_code == 0
    assert captured["fetch_db"] is True
    assert captured["bench_argv"] == [
        "swebench-live",
        "--model",
        "test-model",
        "--backend",
        "ollama",
        "--loop-budget",
        "15",
        "--timeout",
        "1800",
        "--split",
        "verified",
        "--mem-limit",
        "4g",
        "--pids-limit",
        "512",
        "--n-samples",
        "1",
        "--sampling",
        "none",
        "--phase",
        "run",
        "--artifact-dir",
        str(tmp_path / "live" / "artifacts"),
        "--limit",
        "3",
        "--no-check-image-digests",
    ]


def test_smoke_local_forwards_manual_shards(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_bench_swebench_lite(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli.bench_swebench_lite", fake_bench_swebench_lite)

    res = runner.invoke(
        app,
        [
            "bench",
            "smoke",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "smoke.db"),
            "--shard-count",
            "4",
            "--shard-index",
            "2",
        ],
    )

    assert res.exit_code == 0
    assert captured["shards"] is None
    assert captured["shard_count"] == 4
    assert captured["shard_index"] == 2
    assert captured["task_ids"].endswith("smoke-16.txt")
    assert captured["phase"] == "run"
    assert captured["artifact_dir"] == tmp_path / "smoke" / "artifacts"


def test_aider_polyglot_cli_forwards_config(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_single_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_single_benchmark", fake_run_single_benchmark)

    res = runner.invoke(
        app,
        [
            "bench",
            "aider-polyglot",
            "--model",
            "test-model",
            "--language",
            "python",
            "--exercise",
            "hello-world",
            "--benchmark-root",
            str(tmp_path / "polyglot"),
            "--phase",
            "evaluate",
            "--artifact-dir",
            str(tmp_path / "polyglot-artifacts"),
        ],
    )

    assert res.exit_code == 0
    assert captured["benchmark"] == "aider-polyglot"
    assert captured["task_ids"] == "python/hello-world"
    config = captured["config"]
    assert config.model_id == "test-model"
    assert config.backend_name == "openai"
    assert config.loop_budget == 12
    assert config.aider_polyglot_retry_loop_budget == 8
    assert config.aider_polyglot_language == "python"
    assert config.aider_polyglot_root == tmp_path / "polyglot"
    assert config.phase == "evaluate"
    assert config.artifact_dir == tmp_path / "polyglot-artifacts"
    assert captured["loop_budget"] == 20

    assert captured["json_mode"] is False


def test_aider_polyglot_local_forwards_json(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_single_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_single_benchmark", fake_run_single_benchmark)

    res = runner.invoke(
        app,
        [
            "bench",
            "aider-polyglot",
            "--model",
            "test-model",
            "--language",
            "python",
            "--exercise",
            "hello-world",
            "--benchmark-root",
            str(tmp_path / "polyglot"),
            "--json",
        ],
    )

    assert res.exit_code == 0
    assert captured["json_mode"] is True

def test_aider_polyglot_bluevela_forwards_args(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_bench_on_bluevela(*, bench_argv, model, local_db, fetch_db, fetch_artifacts):
        captured["bench_argv"] = bench_argv
        captured["model"] = model
        captured["local_db"] = local_db
        captured["fetch_db"] = fetch_db
        captured["fetch_artifacts"] = fetch_artifacts
        return 0

    monkeypatch.setattr("mcode.bench.remote.run_bench_on_bluevela", fake_run_bench_on_bluevela)

    res = runner.invoke(
        app,
        [
            "bench",
            "aider-polyglot",
            "--model",
            "test-model",
            "--backend",
            "openai",
            "--on",
            "bluevela",
            "--db",
            str(tmp_path / "aider.db"),
            "--loop-budget",
            "20",
            "--retry-loop-budget",
            "8",
            "--temperature",
            "0.3",
            "--seed",
            "123",
            "--language",
            "python",
            "--task-ids",
            "python/connect,go/react",
            "--limit",
            "2",
            "--benchmark-root",
            str(tmp_path / "polyglot"),
            "--shards",
            "4",
            "--phase",
            "generate",
            "--artifact-dir",
            str(tmp_path / "aider-artifacts"),
            "--no-retry",
            "--no-fetch-db",
        ],
    )

    assert res.exit_code == 0
    assert captured["fetch_db"] is False
    assert captured["local_db"] == tmp_path / "aider.db"
    assert captured["model"] == "test-model"
    assert captured["bench_argv"] == [
        "aider-polyglot",
        "--model",
        "test-model",
        "--backend",
        "openai",
        "--loop-budget",
        "20",
        "--retry-loop-budget",
        "8",
        "--benchmark-root",
        str(tmp_path / "polyglot"),
        "--language",
        "python",
        "--phase",
        "generate",
        "--artifact-dir",
        str(tmp_path / "aider-artifacts"),
        "--temperature",
        "0.3",
        "--seed",
        "123",
        "--limit",
        "2",
        "--task-ids",
        "python/connect,go/react",
        "--no-retry",
        "--shards",
        "4",
    ]


def test_aider_polyglot_local_forwards_manual_shards(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_single_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_single_benchmark", fake_run_single_benchmark)

    res = runner.invoke(
        app,
        [
            "bench",
            "aider-polyglot",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "aider.db"),
            "--task-ids",
            "python/connect,go/react",
            "--shard-count",
            "4",
            "--shard-index",
            "2",
        ],
    )

    assert res.exit_code == 0
    config = captured["config"]
    assert config.task_shard_count == 4
    assert config.task_shard_index == 2
    assert captured["task_ids"] == "python/connect,go/react"


def test_aider_polyglot_cli_rejects_exercise_without_language() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bench",
            "aider-polyglot",
            "--model",
            "test-model",
            "--exercise",
            "hello-world",
        ],
    )

    assert res.exit_code != 0
    assert "Invalid value" in _strip_ansi(res.output)



class _SuiteFakePopen:
    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        db.parent.mkdir(parents=True, exist_ok=True)
        with ResultsDB(db) as rdb:
            first_run = rdb.start_run(
                "swebench-lite",
                {
                    "backend_name": "openai",
                    "model_id": "test-model",
                    "loop_budget": 15,
                    "timeout_s": 300,
                    "task_shard_count": 2,
                    "task_shard_index": shard_index,
                    "phase": "evaluate",
                },
            )
            rdb.save_task_result(
                first_run,
                {
                    "task_id": f"lite-{shard_index}",
                    "passed": shard_index == 0,
                    "attempts_used": 1,
                    "time_ms": 10,
                    "exit_code": 0 if shard_index == 0 else 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None if shard_index == 0 else "failed",
                    "code_sha256": f"sha-lite-{shard_index}",
                },
            )
            second_run = rdb.start_run(
                "aider-polyglot",
                {
                    "backend_name": "openai",
                    "model_id": "test-model",
                    "loop_budget": 23,
                    "timeout_s": 300,
                    "task_shard_count": 2,
                    "task_shard_index": shard_index,
                    "phase": "evaluate",
                },
            )
            rdb.save_task_result(
                second_run,
                {
                    "task_id": f"poly-{shard_index}",
                    "passed": True,
                    "attempts_used": 1,
                    "time_ms": 11,
                    "exit_code": 0,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "code_sha256": f"sha-poly-{shard_index}",
                },
            )
        self.stdout = io.StringIO("")
        self.returncode = 0
        self.pid = 300000 + shard_index

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 130


def test_run_sharded_benchmark_merges_full_db_for_suite(tmp_path: Path, monkeypatch) -> None:
    out_db = tmp_path / "suite.db"
    monkeypatch.setattr("mcode.cli.subprocess.Popen", _SuiteFakePopen)

    _run_sharded_benchmark(
        command="suite",
        base_argv=["--model", "test-model"],
        shards=2,
        db=out_db,
        benchmark="suite",
        backend="openai",
        model="test-model",
        loop_budget=23,
        timeout_s=300,
        merge_mode="full_db",
    )

    with ResultsDB(out_db) as rdb:
        run_counts = rdb.conn.execute(
            "SELECT benchmark, COUNT(*) AS runs FROM runs GROUP BY benchmark ORDER BY benchmark"
        ).fetchall()
        task_rows = rdb.conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
    assert [(row["benchmark"], row["runs"]) for row in run_counts] == [
        ("aider-polyglot", 2),
        ("swebench-lite", 2),
    ]
    assert task_rows == 4


def test_suite_shards_forward_to_runner(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_sharded_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_sharded_benchmark", fake_run_sharded_benchmark)

    res = runner.invoke(
        app,
        [
            "bench",
            "suite",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "suite.db"),
            "--shards",
            "2",
            "--phase",
            "generate",
            "--artifact-dir",
            str(tmp_path / "suite-artifacts"),
        ],
        color=False,
    )

    assert res.exit_code == 0
    assert captured["command"] == "suite"
    assert captured["merge_mode"] == "full_db"
    assert "--phase" in captured["base_argv"]
    assert captured["base_argv"][captured["base_argv"].index("--phase") + 1] == "generate"
    assert "--artifact-dir" in captured["base_argv"]
    assert captured["base_argv"][captured["base_argv"].index("--artifact-dir") + 1] == str(
        tmp_path / "suite-artifacts"
    )