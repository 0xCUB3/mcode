from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mcode.bench import remote
from mcode.cli import app
from mcode.launch import config as launch_config
from mcode.launch import state as launch_state
from mcode.launch.models import RunRecord, RunStatus, Target


class _Result:
    def __init__(self, *, ok: bool = True, stdout: str = "", stderr: str = "") -> None:
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr


class _Ssh:
    last: _Ssh | None = None

    def __init__(self, login: str) -> None:
        self.login = login
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.download_trees: list[tuple[str, Path, float]] = []
        type(self).last = self

    def run(self, cmd: str, *, timeout: float = 60.0):
        del timeout
        self.commands.append(cmd)
        if cmd.startswith("bsub -G "):
            return _Result(stdout="Job <4242> is submitted to queue <normal>.\n")
        if cmd.startswith("cat "):
            return _Result(stdout="0\n")
        if cmd.startswith("test -f "):
            return _Result(stdout="128\n")
        if cmd.startswith("test -d "):
            return _Result(stdout="ok\n")
        return _Result()

    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        del timeout
        self.uploads.append((src.read_text(), dst))

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        del timeout
        self.downloads.append((src, dst))

    def download_tree(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        self.download_trees.append((src, dst, timeout))


def _cfg(tmp_path: Path) -> launch_config.LaunchConfig:
    return launch_config.LaunchConfig(
        bluevela=launch_config.BluevelaConfig(
            login="testuser@example.test",
            workspace_root=str(tmp_path / "remote-workspace"),
            shared_root=str(tmp_path / "remote-shared"),
            group="grp_runtime",
            hf_env=str(tmp_path / "hf-env.sh"),
        )
    )


def _patch_remote(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _cfg(tmp_path))
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _Ssh)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000000.0)
    monkeypatch.setattr(remote.os, "getpid", lambda: 1234)


def test_remote_json_helpers(capsys) -> None:
    remote._emit_remote_event(True, "remote_stdout", "plain line", line="plain line")

    assert json.loads(capsys.readouterr().out) == {
        "kind": "remote_stdout",
        "data": {"line": "plain line"},
    }
    assert remote._drop_remote_json_line("tail: cannot open 'log' for reading")
    assert remote._drop_remote_json_line("WARNING: benchmark finished but LSF job 1 is still RUN")
    assert not remote._drop_remote_json_line("podman storage host=h")


def test_run_bench_on_bluevela_submits_and_uploads_script(tmp_path: Path, monkeypatch) -> None:
    _patch_remote(monkeypatch, tmp_path)
    local_db = tmp_path / "results.db"

    assert (
        remote.run_bench_on_bluevela(
            bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
            model="Qwen/Qwen3.5-35B-A3B",
            local_db=local_db,
        )
        == 0
    )

    ssh = _Ssh.last
    assert ssh is not None
    submit = next(cmd for cmd in ssh.commands if cmd.startswith("bsub -G "))
    assert "-G grp_runtime" in submit
    assert "-q normal" in submit
    assert "bench-1777000000000-1234.sh" in submit
    script, dst = ssh.uploads[0]
    assert dst.endswith("bench-1777000000000-1234.sh")
    assert "export OPENAI_BASE_URL=http://host:8321/v1" in script
    assert str(tmp_path / "remote-workspace") in script
    assert str(tmp_path / "remote-shared") in script


def test_bench_artifacts_fetch_downloads_saved_remote_dir(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    launch_state.update(
        None,
        lambda s: s.upsert_run(
            RunRecord(
                id="run-1",
                target=Target.BLUEVELA,
                benchmark="suite",
                status=RunStatus.DONE,
                remote={
                    "login": "testuser@example.test",
                    "remote_artifact_dir": "/remote/artifacts",
                    "local_artifact_dir": "artifacts",
                },
            )
        ),
    )
    monkeypatch.setattr("mcode.launch.ssh.SshClient", _Ssh)

    res = CliRunner().invoke(
        app,
        ["bench", "artifacts-fetch", "run-1", "--dest", str(tmp_path / "out"), "--json"],
        color=False,
    )

    assert res.exit_code == 0
    assert json.loads(res.stdout)["local_artifact_dir"] == str(tmp_path / "out")
    ssh = _Ssh.last
    assert ssh is not None
    assert ssh.download_trees == [("/remote/artifacts", tmp_path / "out", 300)]
