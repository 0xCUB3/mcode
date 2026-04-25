from __future__ import annotations

from pathlib import Path

from mcode.bench import remote
from mcode.launch import config as launch_config


class _FakeResult:
    def __init__(self, *, ok: bool = True, stdout: str = "", stderr: str = "") -> None:
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr


class _FakeSshClient:
    last: _FakeSshClient | None = None

    def __init__(self, login: str) -> None:
        self.login = login
        self.commands: list[str] = []
        self.downloads: list[tuple[str, Path, float]] = []
        type(self).last = self

    def run(self, cmd: str, *, timeout: float = 60.0):
        del timeout
        self.commands.append(cmd)
        if cmd.startswith("mkdir -p "):
            return _FakeResult()
        if cmd.startswith("nohup bash -lc "):
            return _FakeResult(stdout="4242\n")
        if cmd.startswith("cat "):
            return _FakeResult(stdout="0\n")
        if cmd.startswith("test -f "):
            return _FakeResult(stdout="128\n")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        self.downloads.append((src, dst, timeout))


def _bluevela_cfg() -> launch_config.LaunchConfig:
    return launch_config.LaunchConfig(
        bluevela=launch_config.BluevelaConfig(
            login="skula@login3.bluevela.rmf.ibm.com",
            workspace_root="/u/skula/mcode-launch",
            shared_root="/u/skula/mcode-shared",
            group="grp_runtime",
            hf_env="/u/skula/.config/mcode/hf-env.sh",
        )
    )


def test_run_bench_on_bluevela_uses_tmp_for_podman_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, pid: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000000)
    fake_uuid = type("FakeUUID", (), {"hex": "abcdef123456"})()
    monkeypatch.setattr(remote.uuid, "uuid4", lambda: fake_uuid)

    exit_code = remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    assert exit_code == 0
    ssh = _FakeSshClient.last
    assert ssh is not None
    launch_cmd = next(cmd for cmd in ssh.commands if cmd.startswith("nohup bash -lc "))
    run_id = "bench-1777000000-abcdef12-Qwen-Qwen3.5-35B-A3B"

    assert f"export XDG_RUNTIME_DIR=/tmp/mcode-bench-{run_id}" in launch_cmd
    assert 'WORKSPACE_TMP="$XDG_RUNTIME_DIR/tmp"' in launch_cmd
    assert 'GRAPHROOT="$XDG_RUNTIME_DIR/graphroot"' in launch_cmd
    assert 'RUNROOT="$XDG_RUNTIME_DIR/runroot"' in launch_cmd
    assert "trap cleanup EXIT" in launch_cmd
    assert "/u/skula/mcode-shared" not in launch_cmd


def test_run_bench_on_bluevela_forwards_context_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, pid: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000001)
    monkeypatch.setenv("MCODE_CONTEXT_WINDOW", "262144")
    monkeypatch.setenv("MCODE_MAX_NEW_TOKENS", "4096")
    monkeypatch.setenv("MCODE_HARNESS_EXPERIMENTS", "mellea_loop_detect_v1")
    monkeypatch.setenv("MCODE_REACT_TIMEOUT", "2400")

    remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.6-35B-A3B"],
        model="Qwen/Qwen3.6-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    ssh = _FakeSshClient.last
    assert ssh is not None
    launch_cmd = next(cmd for cmd in ssh.commands if cmd.startswith("nohup bash -lc "))
    assert "export MCODE_CONTEXT_WINDOW=262144" in launch_cmd
    assert "export MCODE_MAX_NEW_TOKENS=4096" in launch_cmd
    assert "export MCODE_HARNESS_EXPERIMENTS=mellea_loop_detect_v1" in launch_cmd
    assert "export MCODE_REACT_TIMEOUT=2400" in launch_cmd



def test_run_bench_on_bluevela_prefers_openai_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, pid: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000002)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.6-35B-A3B"],
        model="Qwen/Qwen3.6-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    ssh = _FakeSshClient.last
    assert ssh is not None
    launch_cmd = next(cmd for cmd in ssh.commands if cmd.startswith("nohup bash -lc "))
    assert "export OPENAI_BASE_URL=https://example.test/v1" in launch_cmd
    assert "export OPENAI_API_KEY=secret-key" in launch_cmd



def test_run_bench_on_bluevela_sets_up_aider_polyglot_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, pid: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000003)

    remote.run_bench_on_bluevela(
        bench_argv=[
            "aider-polyglot",
            "--model",
            "Qwen/Qwen3.6-35B-A3B",
            "--benchmark-root",
            "/Users/skula/Documents/polyglot-benchmark",
        ],
        model="Qwen/Qwen3.6-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    ssh = _FakeSshClient.last
    assert ssh is not None
    launch_cmd = next(cmd for cmd in ssh.commands if cmd.startswith("nohup bash -lc "))
    remote_root = "/u/skula/mcode-launch/benchmarks/polyglot-benchmark"
    assert "git clone --depth=1 https://github.com/Aider-AI/polyglot-benchmark.git" in launch_cmd
    assert ") 9>/u/skula/mcode-launch/benchmarks/.polyglot-benchmark.lock" in launch_cmd
    assert f"--benchmark-root {remote_root}" in launch_cmd
    assert "/Users/skula/Documents/polyglot-benchmark" not in launch_cmd
    toolchain_root = "/u/skula/mcode-shared/toolchains/aider-polyglot"
    assert f"TOOLCHAIN_ROOT={toolchain_root}" in launch_cmd
    assert 'export GOROOT="$TOOLCHAIN_ROOT/go"' in launch_cmd
    assert 'export JAVA_HOME="$TOOLCHAIN_ROOT/jdk"' in launch_cmd
    assert 'export RUSTUP_HOME="$TOOLCHAIN_ROOT/rustup"' in launch_cmd
    assert 'export CARGO_HOME="$TOOLCHAIN_ROOT/cargo"' in launch_cmd
    assert "$TOOLCHAIN_ROOT/node/bin" in launch_cmd
    assert "$TOOLCHAIN_ROOT/cmake/bin" in launch_cmd