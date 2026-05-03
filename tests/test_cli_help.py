from __future__ import annotations

import re
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from mcode.cli import app

EXPECTED_OPTIONS: dict[tuple[str, ...], set[str]] = {
    ("bench", "list"): {"benchmark", "status", "artifacts_only", "limit", "json_mode"},
    ("bench", "swebench-lite"): {
        "shards",
        "n_samples",
        "sampling",
        "sampling_budget",
        "selection_attempts",
        "on",
        "fetch_db",
        "fetch_artifacts",
        "diagnostic_traces",
        "phase",
        "artifact_dir",
    },
    ("bench", "swebench-live"): {
        "shards",
        "n_samples",
        "sampling",
        "sampling_budget",
        "selection_attempts",
        "on",
        "fetch_db",
        "fetch_artifacts",
        "diagnostic_traces",
        "phase",
        "artifact_dir",
    },
    ("bench", "aider-polyglot"): {
        "retry_loop_budget",
        "benchmark_root",
        "language",
        "exercise",
        "no_retry",
        "task_ids",
        "on",
        "fetch_db",
        "fetch_artifacts",
        "shards",
        "shard_count",
        "shard_index",
        "phase",
        "artifact_dir",
    },
    ("bench", "smoke"): {
        "shards",
        "diagnostic_traces",
        "fetch_artifacts",
        "phase",
        "artifact_dir",
    },
    ("bench", "suite"): {
        "suite_file",
        "phase",
        "artifact_dir",
        "fetch_artifacts",
        "shards",
        "shard_count",
        "shard_index",
    },
    ("bench", "artifacts-list"): {"db", "run_id", "task_id", "phase", "json_mode"},
    ("bench", "artifacts-fetch"): {"db", "dest", "json_mode"},
    ("bench", "artifacts-show"): {"db", "run_id", "candidate_index"},
    ("bench", "artifacts-patch"): {"db", "run_id", "candidate_index", "out"},
    ("bench", "artifacts-replay"): {
        "db",
        "run_id",
        "out_db",
        "candidate_index",
        "benchmark_root",
        "artifact_dir",
        "fetch_missing_artifacts",
    },
}

HELP_COMMANDS = [
    ("--help",),
    ("bench", "swebench-lite", "--help"),
    ("report", "--help"),
    ("compare", "--help"),
    ("deps", "sync", "--help"),
]


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _invoke(*args: str):
    return CliRunner().invoke(app, list(args), color=False)


def _invoke_help(*args: str) -> str:
    res = CliRunner().invoke(
        app,
        list(args),
        color=False,
        env={"COLUMNS": "120", "TERM": "xterm-256color"},
    )
    assert res.exit_code == 0
    return res.stdout


def _command_option_names(*args: str) -> set[str]:
    command = get_command(app)
    current = command
    for name in args:
        current = current.get_command(None, name)
        assert current is not None
    return {param.name for param in current.params}


def test_cli_options_are_registered() -> None:
    for args, options in EXPECTED_OPTIONS.items():
        assert options <= _command_option_names(*args)


def test_cli_help() -> None:
    for args in HELP_COMMANDS:
        _invoke_help(*args)


def test_cli_rejects_sampling_budget_without_sampling() -> None:
    res = _invoke(
        "bench",
        "swebench-lite",
        "--model",
        "test-model",
        "--sampling-budget",
        "2",
    )
    assert res.exit_code != 0
    assert "--sampling-budget requires --sampling != none" in _strip_ansi(res.output)


def _fake_sync(captured: dict[str, object]):
    def fake_sync(
        project_root: Path,
        *,
        env=None,
        sync_args=None,
        run_command=None,
    ):
        captured["project_root"] = project_root
        captured["sync_args"] = sync_args

        class Selection:
            source = "github"
            local_path = None

        return Selection()

    return fake_sync


def test_cli_deps_sync_defaults_to_dev_extra(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("mcode.uv_setup.sync_uv_environment", _fake_sync(captured))

    res = _invoke("deps", "sync")

    assert res.exit_code == 0
    assert captured["project_root"] == Path.cwd()
    assert captured["sync_args"] == ["--extra", "dev"]


def test_cli_deps_sync_allows_disabling_dev_and_adding_extras(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("mcode.uv_setup.sync_uv_environment", _fake_sync(captured))

    res = _invoke("deps", "sync", "--no-dev", "--extra", "swebench", "--extra", "datasets")

    assert res.exit_code == 0
    assert captured["project_root"] == Path.cwd()
    assert captured["sync_args"] == ["--extra", "swebench", "--extra", "datasets"]
