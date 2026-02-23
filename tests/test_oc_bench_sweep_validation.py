from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mcode.bench.results import ResultsDB


def _load_oc_bench_sweep_module():
    script = Path(__file__).resolve().parents[1] / "deploy" / "k8s" / "oc_bench_sweep.py"
    spec = importlib.util.spec_from_file_location("oc_bench_sweep_script", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_db(
    path: Path,
    *,
    benchmark: str,
    shard_index: int,
    planned_count: int,
    rows: int,
) -> None:
    with ResultsDB(path) as rdb:
        run_id = rdb.start_run(
            benchmark,
            {
                "backend_name": "ollama",
                "model_id": "test-model",
                "loop_budget": 3,
                "retrieval": False,
                "timeout_s": 60,
                "task_shard_count": 20,
                "task_shard_index": shard_index,
                "planned_task_count": planned_count,
            },
        )
        for i in range(rows):
            rdb.save_task_result(
                run_id,
                {
                    "task_id": f"BigCodeBench/{i}",
                    "passed": False,
                    "attempts_used": 1,
                    "time_ms": 10,
                    "exit_code": 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "fail",
                    "error": "Execution failed",
                    "code_sha256": "abc",
                },
            )


def test_validate_shard_db_accepts_matching_counts(tmp_path: Path) -> None:
    mod = _load_oc_bench_sweep_module()
    db = tmp_path / "bigcodebench-complete-shard-9.db"
    _make_db(
        db,
        benchmark="bigcodebench-complete",
        shard_index=9,
        planned_count=2,
        rows=2,
    )
    ok, detail = mod._validate_shard_db(  # noqa: SLF001
        db,
        benchmark="bigcodebench-complete",
        shard_index=9,
    )
    assert ok is True
    assert detail == "rows=2"


def test_validate_shard_db_rejects_incomplete_rows(tmp_path: Path) -> None:
    mod = _load_oc_bench_sweep_module()
    db = tmp_path / "bigcodebench-complete-shard-9.db"
    _make_db(
        db,
        benchmark="bigcodebench-complete",
        shard_index=9,
        planned_count=5,
        rows=2,
    )
    ok, detail = mod._validate_shard_db(  # noqa: SLF001
        db,
        benchmark="bigcodebench-complete",
        shard_index=9,
    )
    assert ok is False
    assert "incomplete task_results" in detail


def test_validate_shard_db_rejects_wrong_shard_index(tmp_path: Path) -> None:
    mod = _load_oc_bench_sweep_module()
    db = tmp_path / "bigcodebench-complete-shard-9.db"
    _make_db(
        db,
        benchmark="bigcodebench-complete",
        shard_index=8,
        planned_count=2,
        rows=2,
    )
    ok, detail = mod._validate_shard_db(  # noqa: SLF001
        db,
        benchmark="bigcodebench-complete",
        shard_index=9,
    )
    assert ok is False
    assert "shard index mismatch" in detail


def test_should_recycle_stuck_shard_when_idle_too_long() -> None:
    mod = _load_oc_bench_sweep_module()
    should = mod._should_recycle_stuck_shard(  # noqa: SLF001
        auto_recycle_stuck_shards=True,
        recycle_stuck_seconds=1200,
        max_stuck_recycles_per_shard=3,
        recycle_count=0,
        mcode_running=True,
        mcode_terminated=False,
        idle_seconds=1300.0,
    )
    assert should is True


def test_should_not_recycle_when_under_limit_or_disabled() -> None:
    mod = _load_oc_bench_sweep_module()
    assert (
        mod._should_recycle_stuck_shard(  # noqa: SLF001
            auto_recycle_stuck_shards=False,
            recycle_stuck_seconds=1200,
            max_stuck_recycles_per_shard=3,
            recycle_count=0,
            mcode_running=True,
            mcode_terminated=False,
            idle_seconds=5000.0,
        )
        is False
    )
    assert (
        mod._should_recycle_stuck_shard(  # noqa: SLF001
            auto_recycle_stuck_shards=True,
            recycle_stuck_seconds=1200,
            max_stuck_recycles_per_shard=3,
            recycle_count=0,
            mcode_running=True,
            mcode_terminated=False,
            idle_seconds=100.0,
        )
        is False
    )


def test_should_not_recycle_after_max_attempts_or_when_not_running() -> None:
    mod = _load_oc_bench_sweep_module()
    assert (
        mod._should_recycle_stuck_shard(  # noqa: SLF001
            auto_recycle_stuck_shards=True,
            recycle_stuck_seconds=1200,
            max_stuck_recycles_per_shard=2,
            recycle_count=2,
            mcode_running=True,
            mcode_terminated=False,
            idle_seconds=5000.0,
        )
        is False
    )
    assert (
        mod._should_recycle_stuck_shard(  # noqa: SLF001
            auto_recycle_stuck_shards=True,
            recycle_stuck_seconds=1200,
            max_stuck_recycles_per_shard=2,
            recycle_count=0,
            mcode_running=False,
            mcode_terminated=False,
            idle_seconds=5000.0,
        )
        is False
    )
