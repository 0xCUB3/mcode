from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.compare import compare_runs
from mcode.bench.results import ResultsDB
from mcode.cli import app


def _write_compare_db(
    db_path: Path,
    *,
    suite_entry_name: str,
    results: dict[str, bool],
) -> None:
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": suite_entry_name,
            },
        )
        for task_id, passed in results.items():
            rdb.save_task_result(
                run_id,
                {
                    "task_id": task_id,
                    "passed": passed,
                    "attempts_used": 1,
                    "time_ms": 10,
                    "exit_code": 0 if passed else 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "error": None if passed else "failed",
                    "code_sha256": task_id,
                },
            )


def test_compare_runs_filters_suite_entry(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    _write_compare_db(
        baseline_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": False},
    )
    _write_compare_db(
        baseline_dir / "b.db",
        suite_entry_name="polyglot-go",
        results={"go/alphametics": True},
    )
    _write_compare_db(
        candidate_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": True},
    )
    _write_compare_db(
        candidate_dir / "b.db",
        suite_entry_name="polyglot-go",
        results={"go/alphametics": False},
    )

    report = compare_runs(
        str(baseline_dir),
        str(candidate_dir),
        suite_name="tiny-polyglot-suite",
        suite_entry_name="polyglot-python",
    )

    assert report["gained"] == ["python/affine-cipher"]
    assert report["lost"] == []
    assert report["baseline_total"] == 1
    assert report["candidate_total"] == 1



def test_compare_cli_json_filters_suite_entry(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline-json"
    candidate_dir = tmp_path / "candidate-json"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    _write_compare_db(
        baseline_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": False},
    )
    _write_compare_db(
        candidate_dir / "a.db",
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": True},
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compare",
            "--baseline-dir",
            str(baseline_dir),
            "--candidate-dir",
            str(candidate_dir),
            "--suite",
            "tiny-polyglot-suite",
            "--suite-entry",
            "polyglot-python",
            "--json",
        ],
        color=False,
    )

    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["gained"] == ["python/affine-cipher"]


def test_results_cli_json_includes_generated_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as rdb:
        run_id = rdb.start_run(
            "aider-polyglot",
            {
                "backend_name": "openai",
                "model_id": "test-model",
                "loop_budget": 23,
                "timeout_s": 300,
                "suite_name": "tiny-polyglot-suite",
                "suite_entry_name": "polyglot-python",
            },
        )
        rdb.conn.execute(
            """
            INSERT INTO artifact_tasks
            (run_id, task_id, benchmark, phase, artifact_root, manifest_path, schema_version,
             repo_id, task_digest, candidate_count, evaluation_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                "python/affine-cipher",
                "aider-polyglot",
                "generate",
                "aider-polyglot/python/affine-cipher",
                str(tmp_path / "manifest.json"),
                1,
                "repo",
                "digest",
                1,
                0,
                "{}",
            ),
        )
        rdb.conn.commit()

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "results",
            "--db",
            str(db_path),
            "--suite",
            "tiny-polyglot-suite",
            "--suite-entry",
            "polyglot-python",
            "--json",
        ],
        color=False,
    )

    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload[0]["artifact_generated_tasks"] == 1
    assert payload[0]["artifact_evaluated_tasks"] == 0