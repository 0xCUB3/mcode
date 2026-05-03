from __future__ import annotations

from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from mcode.bench.suite import SuiteEntry, load_suite_manifest
from mcode.cli import app


def test_default_suite_manifest_covers_mixed_benchmarks() -> None:
    manifest = load_suite_manifest()

    benchmarks = {entry.benchmark for entry in manifest.entries}
    polyglot_languages = {
        entry.language for entry in manifest.entries if entry.benchmark == "aider-polyglot"
    }

    assert "swebench-lite" in benchmarks
    assert "swebench-live" in benchmarks
    assert polyglot_languages == {"python", "go", "rust", "javascript", "cpp", "java"}


def test_bench_suite_help_exposes_phase_and_manifest_flags() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["bench", "suite", "--help"], color=False)

    assert res.exit_code == 0
    command = get_command(app)
    bench_command = command.get_command(None, "bench")
    assert bench_command is not None
    suite_command = bench_command.get_command(None, "suite")
    assert suite_command is not None
    option_names = {param.name for param in suite_command.params}
    assert "suite_file" in option_names
    assert "phase" in option_names
    assert "artifact_dir" in option_names
    assert "shards" in option_names
    assert "shard_count" in option_names
    assert "shard_index" in option_names

def test_bench_suite_runs_each_manifest_entry(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    seen: list[dict[str, object]] = []

    monkeypatch.setattr(
        "mcode.cli.load_suite_manifest",
        lambda _path=None: type(
            "Manifest",
            (),
            {
                "entries": (
                    SuiteEntry(
                        name="lite",
                        benchmark="swebench-lite",
                        limit=2,
                        task_ids=("a", "b"),
                        split="test",
                        dataset="example/lite",
                    ),
                    SuiteEntry(
                        name="polyglot",
                        benchmark="aider-polyglot",
                        limit=1,
                        language="python",
                    ),
                )
            },
        )(),
    )

    def fake_run_suite_entry(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr("mcode.cli._run_suite_entry", fake_run_suite_entry)

    res = runner.invoke(
        app,
        [
            "bench",
            "suite",
            "--model",
            "test-model",
            "--db",
            str(tmp_path / "suite.db"),
            "--phase",
            "generate",
            "--shard-count",
            "4",
            "--shard-index",
            "1",
        ],
        color=False,
    )

    assert res.exit_code == 0
    assert [item["entry"].name for item in seen] == ["lite", "polyglot"]
    assert all(item["phase"] == "generate" for item in seen)
    assert all(item["shard_count"] == 4 for item in seen)
    assert all(item["shard_index"] == 1 for item in seen)
    assert all(isinstance(item["artifact_dir"], Path) for item in seen)


def test_bench_suite_bluevela_forwards_manifest_and_phase(monkeypatch, tmp_path: Path) -> None:
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

    suite_file = tmp_path / "suite.json"
    suite_file.write_text('{"entries": [{"name": "lite", "benchmark": "swebench-lite"}]}')
    db = tmp_path / "suite.db"
    res = runner.invoke(
        app,
        [
            "bench",
            "suite",
            "--model",
            "test-model",
            "--db",
            str(db),
            "--suite-file",
            str(suite_file),
            "--phase",
            "evaluate",
            "--on",
            "bluevela",
        ],
        color=False,
    )

    assert res.exit_code == 0
    bench_argv = captured["bench_argv"]
    assert bench_argv[0] == "suite"
    assert "--suite-file" in bench_argv
    assert "--phase" in bench_argv
    assert "evaluate" in bench_argv
    assert "--artifact-dir" in bench_argv
    assert captured["local_db"] == db
    assert captured["fetch_db"] is True
