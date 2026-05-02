from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.artifacts import (
    TaskArtifactManifest,
    TaskArtifactStore,
    digest_json,
    iso_utc_now,
    make_task_digest,
)
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


def _write_compare_artifact_db(
    db_path: Path,
    *,
    suite_entry_name: str,
    task_id: str,
    patch: str,
    verified: bool,
) -> None:
    artifact_dir = db_path.parent / f"{db_path.stem}-artifacts"
    store = TaskArtifactStore.from_task(
        artifact_dir=artifact_dir,
        benchmark="aider-polyglot",
        task_id=task_id,
    )
    task_ref = store.build_task_ref(
        repo_id=f"aider-polyglot/{task_id}",
        task_digest=make_task_digest(
            benchmark="aider-polyglot",
            task_id=task_id,
            repo_id=f"aider-polyglot/{task_id}",
            metadata={"suite_entry_name": suite_entry_name},
        ),
        metadata={"suite_entry_name": suite_entry_name},
    )
    candidate = store.write_candidate(
        candidate_index=0,
        patch=patch,
        terminal_reason="submitted",
        selected=True,
        submission_json='{"summary":"done"}',
        generation_time_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        provider="openai",
        response_model="test-model",
        validation_passed_count=1 if verified else 0,
        validation_failed_count=0 if verified else 1,
        zero_edit=False,
        zero_verification=not verified,
        verification_succeeded=verified,
        trace_events=None,
        verification_evidence=None,
        failure_counters=None,
    )
    manifest = TaskArtifactManifest(
        schema_version=1,
        phase="generate",
        generated_at=iso_utc_now(),
        run_config_digest=digest_json({"suite_entry_name": suite_entry_name}),
        code_sha=None,
        model_id="test-model",
        backend_name="openai",
        task=task_ref,
        candidates=(candidate,),
        evaluations=(),
    )
    manifest_path = store.write_manifest(manifest)
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
        rdb.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)


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


def test_compare_runs_accepts_db_file_inputs(tmp_path: Path) -> None:
    baseline_db = tmp_path / "baseline.db"
    candidate_db = tmp_path / "candidate.db"

    _write_compare_db(
        baseline_db,
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": False},
    )
    _write_compare_db(
        candidate_db,
        suite_entry_name="polyglot-python",
        results={"python/affine-cipher": True},
    )

    report = compare_runs(
        str(baseline_db),
        str(candidate_db),
        suite_name="tiny-polyglot-suite",
        suite_entry_name="polyglot-python",
    )

    assert report["gained"] == ["python/affine-cipher"]
    assert report["candidate_passed"] == 1


def test_compare_runs_includes_artifact_only_summary(tmp_path: Path) -> None:
    baseline_db = tmp_path / "baseline-artifacts.db"
    candidate_db = tmp_path / "candidate-artifacts.db"

    _write_compare_artifact_db(
        baseline_db,
        suite_entry_name="polyglot-python",
        task_id="python/affine-cipher",
        patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
        verified=False,
    )
    _write_compare_artifact_db(
        candidate_db,
        suite_entry_name="polyglot-python",
        task_id="python/affine-cipher",
        patch="diff --git a/foo.py b/foo.py\n+x = 2\n+y = 3\n",
        verified=True,
    )

    report = compare_runs(
        str(baseline_db),
        str(candidate_db),
        suite_name="tiny-polyglot-suite",
        suite_entry_name="polyglot-python",
    )

    assert report["baseline_total"] == 0
    assert report["candidate_total"] == 0
    assert report["baseline_artifacts"]["generated_tasks"] == 1
    assert report["candidate_artifacts"]["selected_verified_count"] == 1
    assert (
        report["candidate_artifacts"]["selected_patch_byte_count_total"]
        > (report["baseline_artifacts"]["selected_patch_byte_count_total"])
    )
