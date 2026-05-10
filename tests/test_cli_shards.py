from __future__ import annotations

import io
import re
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.results import ResultsDB
from mcode.bench.shards import _latest_run_summary, _run_sharded_benchmark
from mcode.cli import app


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _save_result(db: Path, *, benchmark: str, task_id: str, shard_index: int, passed: bool) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with ResultsDB(db) as rdb:
        run_id = rdb.start_run(
            benchmark,
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 15,
                "timeout_s": 300,
                "task_shard_count": 2,
                "task_shard_index": shard_index,
                "cache_dir": str(db.parent / "cache"),
            },
        )
        rdb.save_task_result(
            run_id,
            {
                "task_id": task_id,
                "passed": passed,
                "attempts_used": 1,
                "time_ms": 1000 + shard_index,
                "exit_code": 0 if passed else 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None if passed else "failed",
                "code_sha256": f"sha-{task_id}",
            },
        )


class _FakePopen:
    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        _save_result(
            db,
            benchmark="swebench-lite",
            task_id=f"task-{shard_index}",
            shard_index=shard_index,
            passed=shard_index == 0,
        )
        self.stdout = io.StringIO(f"shard {shard_index}\n")
        self.returncode = 0
        self.pid = 100000 + shard_index

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 130


def test_run_sharded_benchmark_merges_shards_into_output_db(tmp_path: Path, monkeypatch) -> None:
    out_db = tmp_path / "results.db"
    monkeypatch.setattr("mcode.bench.shards.subprocess.Popen", _FakePopen)

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
    assert sorted(
        path.name for path in next((tmp_path / "results-shards").iterdir()).iterdir()
    ) == [
        "results-shard-0.db",
        "results-shard-0.log",
        "results-shard-1.db",
        "results-shard-1.log",
    ]


def test_cli_rejects_auto_and_manual_shards_together(tmp_path: Path) -> None:
    res = CliRunner().invoke(
        app,
        [
            "bench",
            "swebench-lite",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "results.db"),
            "--shards",
            "2",
            "--shard-count",
            "2",
            "--shard-index",
            "0",
        ],
    )

    assert res.exit_code != 0
    assert "--shards cannot be combined" in _strip_ansi(res.output)


def test_smoke_bluevela_forwards_minimal_remote_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_bench_on_bluevela(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("mcode.bench.remote.run_bench_on_bluevela", fake_run_bench_on_bluevela)

    res = CliRunner().invoke(
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
    assert captured["model"] == "test-model"
    assert captured["local_db"] == tmp_path / "smoke.db"
    assert captured["fetch_artifacts"] is False
    output = _strip_ansi(res.output)
    assert "smoke on bluevela" in output
    assert "db=" in output
    assert "smoke.db" in output
    argv = captured["bench_argv"]
    assert argv[:3] == ["smoke", "--model", "test-model"]
    assert argv[argv.index("--shards") + 1] == "4"
    assert "--artifact-dir" in argv


def test_smoke_local_forwards_manual_shards(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "mcode.bench.cli.bench_swebench_lite", lambda **kwargs: captured.update(kwargs)
    )

    res = CliRunner().invoke(
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
    assert captured["shard_count"] == 4
    assert captured["shard_index"] == 2
    assert captured["artifact_dir"] == tmp_path / "smoke" / "artifacts"


def test_aider_polyglot_cli_forwards_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "mcode.bench.cli._run_single_benchmark", lambda **kwargs: captured.update(kwargs)
    )

    res = CliRunner().invoke(
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
        ],
    )

    assert res.exit_code == 0
    config = captured["config"]
    assert captured["benchmark"] == "aider-polyglot"
    assert captured["task_ids"] == "python/hello-world"
    assert config.model_id == "test-model"
    assert config.aider_polyglot_root == tmp_path / "polyglot"
    assert config.phase == "evaluate"


def test_aider_polyglot_rejects_exercise_without_language() -> None:
    res = CliRunner().invoke(
        app,
        ["bench", "aider-polyglot", "--model", "test-model", "--exercise", "hello-world"],
    )

    assert res.exit_code != 0
    assert "Invalid value" in _strip_ansi(res.output)


class _SuiteFakePopen(_FakePopen):
    def __init__(self, argv, **kwargs) -> None:
        del kwargs
        args = list(argv)
        shard_index = int(args[args.index("--shard-index") + 1])
        db = Path(args[args.index("--db") + 1])
        _save_result(
            db,
            benchmark="swebench-lite",
            task_id=f"lite-{shard_index}",
            shard_index=shard_index,
            passed=shard_index == 0,
        )
        _save_result(
            db,
            benchmark="aider-polyglot",
            task_id=f"poly-{shard_index}",
            shard_index=shard_index,
            passed=True,
        )
        self.stdout = io.StringIO("")
        self.returncode = 0
        self.pid = 300000 + shard_index


def test_suite_shards_use_full_db_merge(tmp_path: Path, monkeypatch) -> None:
    out_db = tmp_path / "suite.db"
    monkeypatch.setattr("mcode.bench.shards.subprocess.Popen", _SuiteFakePopen)

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
