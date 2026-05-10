from __future__ import annotations

import re
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from mcode.bench.toolchains import ToolchainCheck
from mcode.cli import app

EXPECTED_OPTIONS: dict[tuple[str, ...], set[str]] = {
    ("bench", "list"): {"benchmark", "status", "artifacts_only", "limit", "json_mode"},
    ("bench", "show"): {"json_mode"},
    ("bench", "swebench-lite"): {
        "shards",
        "n_samples",
        "sampling",
        "sampling_budget",
        "selection_attempts",
        "eval_repair_attempts",
        "on",
        "fetch_db",
        "fetch_artifacts",
        "chunk_size",
        "relaunch_vllm",
        "vllm_tensor_parallel",
        "vllm_max_model_len",
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
        "eval_repair_attempts",
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
    ("compare", "--help"),
    ("deps", "sync", "--help"),
    ("deps", "toolchains", "--help"),
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


def test_cli_aider_polyglot_reports_missing_toolchain(monkeypatch, tmp_path: Path) -> None:
    from mcode.bench.toolchains import PolyglotToolchainError

    def fail_toolchain(self, benchmark, *, limit=None, task_ids=None):
        raise PolyglotToolchainError(
            "polyglot toolchain unavailable\n"
            "- go: go not found on PATH\n"
            "  next: install Go\n"
            "or run: mcode deps toolchains --benchmark aider-polyglot --install"
        )

    monkeypatch.setattr("mcode.bench.runner.BenchmarkRunner.run_benchmark", fail_toolchain)

    res = _invoke(
        "bench",
        "aider-polyglot",
        "--model",
        "test-model",
        "--language",
        "go",
        "--exercise",
        "beer-song",
        "--db",
        str(tmp_path / "results.db"),
    )

    output = _strip_ansi(res.output)
    assert res.exit_code == 2
    assert "polyglot toolchain unavailable" in output
    assert "mcode deps toolchains --benchmark aider-polyglot --install" in output


def test_cli_deps_toolchains_reports_missing_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "mcode.bench.toolchains.check_polyglot_toolchains",
        lambda languages: (
            ToolchainCheck("go", "go", False, "go not found on PATH", "install Go"),
        ),
    )

    res = _invoke("deps", "toolchains", "--language", "go")

    assert res.exit_code == 1
    assert "go not found on PATH" in _strip_ansi(res.output)


def test_cli_deps_toolchains_install_invokes_installer(monkeypatch) -> None:
    installed: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "mcode.bench.toolchains.install_polyglot_toolchains",
        lambda languages: installed.append(tuple(languages)),
    )
    monkeypatch.setattr(
        "mcode.bench.toolchains.check_polyglot_toolchains",
        lambda languages: (ToolchainCheck("go", "go", True, "/usr/bin/go", ""),),
    )

    res = _invoke("deps", "toolchains", "--language", "go", "--install")

    assert res.exit_code == 0
    assert installed == [("go",)]
