from __future__ import annotations

from pathlib import Path

from mcode.bench import remote
from mcode.launch import config as launch_config
from mcode.launch import state as launch_state


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
        self.uploads: list[tuple[str, str, float]] = []
        type(self).last = self

    def run(self, cmd: str, *, timeout: float = 60.0):
        del timeout
        self.commands.append(cmd)
        if cmd.startswith("mkdir -p "):
            return _FakeResult()
        if cmd.startswith("bsub -G "):
            return _FakeResult(stdout="Job <4242> is submitted to queue <normal>.\n")
        if cmd.startswith("cat "):
            return _FakeResult(stdout="0\n")
        if cmd.startswith("test -f "):
            return _FakeResult(stdout="128\n")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        self.downloads.append((src, dst, timeout))

    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        self.uploads.append((src.read_text(), dst, timeout))


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


def test_bluevela_bench_submits_lsf_job_with_podman_on_compute(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, job_id: None)
    local_db = tmp_path / "results.db"
    exit_code = remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=local_db,
    )

    assert exit_code == 0
    ssh = _FakeSshClient.last
    assert ssh is not None
    launch_cmd = next(cmd for cmd in ssh.commands if cmd.startswith("bsub -G "))
    run_id = remote._remote_run_key(
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=local_db,
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        forwarded_env={},
    )

    assert "nohup setsid" not in launch_cmd
    assert "-G grp_runtime" in launch_cmd
    assert "-q normal" in launch_cmd
    assert "-n 8" in launch_cmd
    assert "span[hosts=1]" in launch_cmd
    assert "rusage[mem=16000]" in launch_cmd
    assert launch_cmd.endswith(f"bash /u/skula/mcode-launch/bench-runs/{run_id}/bench.sh")
    assert ssh.uploads
    script_text = ssh.uploads[0][0]
    assert 'if [ -z "${LSB_JOBID:-}" ]; then' in script_text
    assert "refusing to start podman outside an LSF compute job" in script_text
    assert f"export XDG_RUNTIME_DIR=/u/skula/mcode-shared/podman-runtime/{run_id}" in script_text
    assert f"WORKSPACE_TMP=/u/skula/mcode-shared/podman-tmp/{run_id}" in script_text
    assert 'GRAPHROOT="$XDG_RUNTIME_DIR/graphroot"' in script_text
    assert 'RUNROOT="$XDG_RUNTIME_DIR/runroot"' in script_text
    assert 'CONTAINERS_CONF="$XDG_RUNTIME_DIR/containers.conf"' in script_text
    assert "keyring=false" in script_text
    assert "export CONTAINERS_CONF" in script_text
    assert 'podman unshare rm -rf "$XDG_RUNTIME_DIR"' in script_text
    assert "trap cleanup EXIT" in script_text
    assert 'export MCODE_PODMAN_LOCK_DIR="$XDG_RUNTIME_DIR"' in script_text
    assert "max_infra_retries=1" in script_text
    assert 'if [ "$rc" = "86" ]' in script_text
    assert "resetting runtime" in script_text
    assert f"/u/skula/mcode-shared/podman-runtime/{run_id}" in script_text
    assert 'REGISTRY_AUTH_FILE=/u/skula/mcode-shared/containers-auth.json' in script_text


def test_run_bench_on_bluevela_forwards_context_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, job_id: None)
    monkeypatch.setattr(remote.time, "time", lambda: 1777000001)
    monkeypatch.setenv("MCODE_CONTEXT_WINDOW", "262144")
    monkeypatch.setenv("MCODE_MAX_NEW_TOKENS", "4096")
    monkeypatch.setenv("MCODE_REACT_TIMEOUT", "2400")

    remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.6-35B-A3B"],
        model="Qwen/Qwen3.6-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    ssh = _FakeSshClient.last
    assert ssh is not None
    script_text = ssh.uploads[0][0]
    assert "export MCODE_CONTEXT_WINDOW=262144" in script_text
    assert "export MCODE_MAX_NEW_TOKENS=4096" in script_text
    assert "export MCODE_REACT_TIMEOUT=2400" in script_text


def test_run_bench_on_bluevela_prefers_openai_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, job_id: None)
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
    script_text = ssh.uploads[0][0]
    assert "export OPENAI_BASE_URL=https://example.test/v1" in script_text
    assert "export OPENAI_API_KEY=secret-key" in script_text


def test_run_bench_on_bluevela_sets_up_aider_polyglot_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", lambda ssh, remote_log, job_id: None)
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
    script_text = ssh.uploads[0][0]
    remote_root = "/u/skula/mcode-launch/benchmarks/polyglot-benchmark"
    assert "git clone --depth=1 https://github.com/Aider-AI/polyglot-benchmark.git" in script_text
    assert ") 9>/u/skula/mcode-launch/benchmarks/.polyglot-benchmark.lock" in script_text
    assert f"--benchmark-root {remote_root}" in script_text
    assert "/Users/skula/Documents/polyglot-benchmark" not in script_text
    toolchain_root = "/u/skula/mcode-shared/toolchains/aider-polyglot"
    assert f"TOOLCHAIN_ROOT={toolchain_root}" in script_text
    assert 'export GOROOT="$TOOLCHAIN_ROOT/go"' in script_text
    assert 'export JAVA_HOME="$TOOLCHAIN_ROOT/jdk"' in script_text
    assert 'export RUSTUP_HOME="$TOOLCHAIN_ROOT/rustup"' in script_text
    assert 'export CARGO_HOME="$TOOLCHAIN_ROOT/cargo"' in script_text
    assert "$TOOLCHAIN_ROOT/node/bin" in script_text
    assert "$TOOLCHAIN_ROOT/cmake/bin" in script_text


class _RecoverableMetadataSshClient(_FakeSshClient):
    def run(self, cmd: str, *, timeout: float = 60.0):
        del timeout
        self.commands.append(cmd)
        if cmd.startswith("mkdir -p "):
            return _FakeResult()
        if cmd.startswith("bsub -G "):
            return _FakeResult(stdout="Job <4242> is submitted to queue <normal>.\n")
        if cmd.startswith("cat "):
            return _FakeResult(ok=False, stderr="sentinel missing")
        if cmd.startswith("test -f "):
            return _FakeResult(stdout="123\n")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        super().download(src, dst, timeout=timeout)
        raise RuntimeError("scp failed")

    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        self.uploads.append((src.read_text(), dst, timeout))


def test_run_bench_on_bluevela_keeps_recoverable_metadata_on_fetch_failure(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _RecoverableMetadataSshClient)
    monkeypatch.setattr(
        remote,
        "_stream_remote_log",
        lambda ssh, remote_log, job_id: (_ for _ in ()).throw(RuntimeError("ssh dropped")),
    )
    local_db = tmp_path / "results.db"

    exit_code = remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=local_db,
    )

    assert exit_code == 99
    snap = launch_state.load(state_path)
    assert len(snap.runs) == 1
    rec = snap.runs[0]
    assert rec.remote["remote_db"].endswith("/results.db")
    assert rec.remote["remote_log"].endswith("/bench.log")
    assert rec.remote["exit_sentinel"].endswith("/exit_code")
    assert rec.updated_at is not None
    ssh = _RecoverableMetadataSshClient.last
    assert ssh is not None
    assert ssh.downloads
