from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench.artifacts import (
    SCHEMA_VERSION,
    TaskArtifactManifest,
    TaskArtifactStore,
    digest_json,
    iso_utc_now,
    make_task_digest,
)
from mcode.bench.results import ResultsDB
from mcode.cli import app


def _seed_artifact_run(tmp_path: Path) -> tuple[Path, int, str]:
    db_path = tmp_path / "results.db"
    artifact_dir = tmp_path / "artifacts"
    store = TaskArtifactStore.from_task(
        artifact_dir=artifact_dir,
        benchmark="aider-polyglot",
        task_id="python/affine-cipher",
    )
    task_ref = store.build_task_ref(
        repo_id="aider-polyglot/python/affine-cipher",
        task_digest=make_task_digest(
            benchmark="aider-polyglot",
            task_id="python/affine-cipher",
            repo_id="aider-polyglot/python/affine-cipher",
            metadata={"language": "python"},
        ),
        metadata={"language": "python"},
    )
    candidate = store.write_candidate(
        candidate_index=0,
        patch="diff --git a/foo.py b/foo.py\n+x = 2\n",
        terminal_reason="submitted",
        selected=True,
        submission_json='{"summary":"done"}',
        generation_time_ms=123,
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        provider="openai",
        response_model="test-model",
        validation_passed_count=1,
        validation_failed_count=0,
        zero_edit=False,
        zero_verification=False,
        verification_succeeded=True,
        trace_events=None,
        verification_evidence=None,
        failure_counters=None,
    )
    manifest = TaskArtifactManifest(
        schema_version=SCHEMA_VERSION,
        phase="evaluate",
        generated_at=iso_utc_now(),
        run_config_digest=digest_json({"suite_entry_name": "polyglot-python"}),
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
                "suite_entry_name": "polyglot-python",
            },
        )
        rdb.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)
    return db_path, run_id, task_ref.task_id


def test_bench_artifacts_list_and_show(tmp_path: Path) -> None:
    db_path, run_id, task_id = _seed_artifact_run(tmp_path)
    runner = CliRunner()

    list_res = runner.invoke(
        app,
        ["bench", "artifacts-list", "--db", str(db_path), "--run-id", str(run_id)],
        color=False,
    )
    assert list_res.exit_code == 0
    assert task_id in list_res.stdout
    assert "evaluate" in list_res.stdout

    filtered_res = runner.invoke(
        app,
        [
            "bench",
            "artifacts-list",
            "--db",
            str(db_path),
            "--run-id",
            str(run_id),
            "--task-id",
            task_id,
            "--phase",
            "evaluate",
        ],
        color=False,
    )
    assert filtered_res.exit_code == 0
    assert task_id in filtered_res.stdout

    list_json_res = runner.invoke(
        app,
        [
            "bench",
            "artifacts-list",
            "--db",
            str(db_path),
            "--run-id",
            str(run_id),
            "--json",
        ],
        color=False,
    )
    assert list_json_res.exit_code == 0
    payload = json.loads(list_json_res.stdout)
    assert payload[0]["task_id"] == task_id
    assert payload[0]["phase"] == "evaluate"
    patch_res = runner.invoke(
        app,
        ["bench", "artifacts-patch", task_id, "--db", str(db_path), "--run-id", str(run_id)],
        color=False,
    )
    assert patch_res.exit_code == 0
    assert "+x = 2" in patch_res.stdout

    show_res = runner.invoke(
        app,
        ["bench", "artifacts-show", task_id, "--db", str(db_path), "--run-id", str(run_id)],
        color=False,
    )
    assert show_res.exit_code == 0
    assert '"task_id": "python/affine-cipher"' in show_res.stdout
    assert '"suite_entry_name": "polyglot-python"' not in show_res.stdout
    assert '"benchmark": "aider-polyglot"' in show_res.stdout


def test_bench_artifacts_list_defaults_to_latest_run(tmp_path: Path) -> None:
    db_path, _run_id, task_id = _seed_artifact_run(tmp_path)
    runner = CliRunner()

    res = runner.invoke(app, ["bench", "artifacts-list", "--db", str(db_path)], color=False)
    assert res.exit_code == 0
    assert task_id in res.stdout



def test_bench_artifacts_replay_builds_evaluate_run(monkeypatch, tmp_path: Path) -> None:
    db_path, run_id, task_id = _seed_artifact_run(tmp_path)
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_single_benchmark(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("mcode.cli._run_single_benchmark", fake_run_single_benchmark)

    override_root = tmp_path / "polyglot"
    override_artifacts = tmp_path / "override-artifacts"
    override_task_dir = override_artifacts / "aider-polyglot" / "python" / "affine-cipher"
    override_task_dir.mkdir(parents=True)
    source_manifest = (
        tmp_path
        / "artifacts"
        / "aider-polyglot"
        / "python"
        / "affine-cipher"
        / "manifest.json"
    )
    (override_task_dir / "manifest.json").write_text(source_manifest.read_text())
    (override_task_dir / "patch.diff").write_text("diff --git a/foo.py b/foo.py\n+x = 2\n")

    res = runner.invoke(
        app,
        [
            "bench",
            "artifacts-replay",
            task_id,
            "--db",
            str(db_path),
            "--run-id",
            str(run_id),
            "--candidate-index",
            "0",
            "--benchmark-root",
            str(override_root),
            "--artifact-dir",
            str(override_artifacts),
        ],
        color=False,
    )

    assert res.exit_code == 0
    assert captured["benchmark"] == "aider-polyglot"
    assert captured["task_ids"] == task_id
    assert captured["db"] == db_path.with_name("results-replay.db")
    config = captured["config"]
    assert config.phase == "evaluate"
    assert config.artifact_candidate_index == 0
    assert config.artifact_dir == override_artifacts
    assert config.aider_polyglot_root == override_root