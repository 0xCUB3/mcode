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
        self.download_trees: list[tuple[str, Path, float]] = []
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
        if cmd.startswith("test -d "):
            return _FakeResult(stdout="ok\n")
        if cmd.startswith("STAT=$(bjobs -noheader -o stat "):
            return _FakeResult()
        if cmd.startswith("bkill "):
            return _FakeResult(stdout="Job <4242> is being terminated\n")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        self.downloads.append((src, dst, timeout))

    def download_tree(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        self.download_trees.append((src, dst, timeout))

    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        self.uploads.append((src.read_text(), dst, timeout))


def _bluevela_cfg(
    *,
    graphroot_base: str | None = None,
    runroot_base: str | None = None,
 ) -> launch_config.LaunchConfig:
    return launch_config.LaunchConfig(
        bluevela=launch_config.BluevelaConfig(
            login="skula@login3.bluevela.rmf.ibm.com",
            workspace_root="/u/skula/mcode-launch",
            shared_root="/u/skula/mcode-shared",
            group="grp_runtime",
            hf_env="/u/skula/.config/mcode/hf-env.sh",
            podman=launch_config.BluevelaPodmanConfig(
                graphroot_base=graphroot_base,
                runroot_base=runroot_base,
            ),
        )
    )


def _set_attempt_context(monkeypatch, *, timestamp: float, pid: int = 4242) -> str:
    monkeypatch.setattr(remote.time, "time", lambda: timestamp)
    monkeypatch.setattr(remote.os, "getpid", lambda: pid)
    return f"{int(timestamp * 1000)}-{pid}"

def _ignore_stream(ssh, remote_log, *, exit_sentinel, job_id) -> None:
    del ssh, remote_log, exit_sentinel, job_id


def test_bluevela_bench_submits_lsf_job_with_podman_on_compute(tmp_path, monkeypatch) -> None:
    streamed: dict[str, str] = {}
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    attempt_token = _set_attempt_context(monkeypatch, timestamp=1777000000.0)

    def fake_stream(ssh, remote_log, *, exit_sentinel, job_id) -> None:
        del ssh
        streamed.update(
            remote_log=remote_log,
            exit_sentinel=exit_sentinel,
            job_id=job_id,
        )

    monkeypatch.setattr(remote, "_stream_remote_log", fake_stream)
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
    remote_log = f"/u/skula/mcode-launch/bench-runs/{run_id}/logs/bench-{attempt_token}.log"
    exit_sentinel = f"/u/skula/mcode-launch/bench-runs/{run_id}/exit-{attempt_token}.code"

    assert "nohup setsid" not in launch_cmd
    assert "-G grp_runtime" in launch_cmd
    assert "-q normal" in launch_cmd
    assert "-n 8" in launch_cmd
    assert "span[hosts=1]" in launch_cmd
    assert "rusage[mem=16000]" in launch_cmd
    assert f"-o {remote_log} -e {remote_log}" in launch_cmd
    assert launch_cmd.endswith(
        f"bash /u/skula/mcode-launch/bench-runs/{run_id}/bench-{attempt_token}.sh"
    )
    assert streamed == {
        "remote_log": remote_log,
        "exit_sentinel": exit_sentinel,
        "job_id": "4242",
    }
    assert ssh.uploads
    script_text = ssh.uploads[0][0]
    assert ssh.uploads[0][1].endswith(f"/bench-{attempt_token}.sh")
    assert 'if [ -z "${LSB_JOBID:-}" ]; then' in script_text
    assert "refusing to start podman outside an LSF compute job" in script_text
    assert f"export XDG_RUNTIME_DIR=/u/skula/mcode-shared/podman-runtime/{run_id}" in script_text
    assert f"export WORKSPACE_TMP=/u/skula/mcode-shared/podman-tmp/{run_id}" in script_text
    assert 'export TMPDIR="$WORKSPACE_TMP"' in script_text
    assert 'GRAPHROOT_BASE=/u/skula/mcode-shared/podman-graphroot' in script_text
    assert 'RUNROOT_BASE=/u/skula/mcode-shared/podman-runroot' in script_text
    assert 'LOCKROOT_BASE="$GRAPHROOT_BASE/locks"' in script_text
    assert 'HOST_TAG="$(hostname -s)"' in script_text
    assert 'GRAPHROOT="$GRAPHROOT_BASE/$HOST_TAG"' in script_text
    assert 'RUNROOT="$RUNROOT_BASE/$HOST_TAG"' in script_text
    assert 'CONTAINERS_CONF="$XDG_RUNTIME_DIR/containers.conf"' in script_text
    assert "keyring=false" in script_text
    assert "export CONTAINERS_CONF" in script_text
    assert 'cleanup_dir() {' in script_text
    assert 'cleanup_runtime_dir() {' in script_text
    assert 'cleanup_dir "$XDG_RUNTIME_DIR" "runtime"' in script_text
    assert 'cleanup_dir "$GRAPHROOT" "graphroot"' in script_text
    assert 'cleanup_dir "$RUNROOT" "runroot"' in script_text
    assert 'run_with_timeout 5 mv "$target" "$cleanup_target"' in script_text
    assert 'run_with_timeout 20 rm -rf "$cleanup_target"' in script_text
    assert 'run_with_timeout 20 podman unshare rm -rf "$cleanup_target"' in script_text
    assert 'wait "$PODMAN_PID"' not in script_text
    assert 'setsid podman' in script_text
    assert 'kill -TERM -- "-$pid"' in script_text
    assert 'kill -KILL -- "-$pid"' in script_text
    assert "trap cleanup EXIT" in script_text
    assert 'reset_persistent_podman_store() {' in script_text
    assert 'grep -q "database configuration mismatch"' in script_text
    assert 'podman store mismatch under $GRAPHROOT, clearing persistent store once' in script_text
    assert 'if [ "$rc" = "86" ]' in script_text
    assert 'resetting runtime' in script_text
    assert f"/u/skula/mcode-shared/podman-runtime/{run_id}" in script_text
    assert 'REGISTRY_AUTH_FILE=/u/skula/mcode-shared/containers-auth.json' in script_text
    assert "/tmp" not in script_text


def test_bluevela_bench_uses_graphroot_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        remote.launch_config,
        "load",
        lambda: _bluevela_cfg(graphroot_base="/proj/custom/podman-graphroot"),
    )
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    _set_attempt_context(monkeypatch, timestamp=1777000000.5)

    remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    ssh = _FakeSshClient.last
    assert ssh is not None
    script_text = ssh.uploads[0][0]
    assert 'GRAPHROOT_BASE=/proj/custom/podman-graphroot' in script_text
    assert 'GRAPHROOT="$GRAPHROOT_BASE/$HOST_TAG"' in script_text


def test_run_bench_on_bluevela_forwards_context_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    _set_attempt_context(monkeypatch, timestamp=1777000001.0)
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
def test_run_bench_on_bluevela_fetches_artifacts_when_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    _set_attempt_context(monkeypatch, timestamp=1777000001.5)

    local_db = tmp_path / "results.db"
    exit_code = remote.run_bench_on_bluevela(
        bench_argv=[
            "smoke",
            "--model",
            "Qwen/Qwen3.5-35B-A3B",
            "--artifact-dir",
            "experiments/results/smoke/artifacts",
        ],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=local_db,
        fetch_artifacts=True,
    )

    assert exit_code == 0
    ssh = _FakeSshClient.last
    assert ssh is not None
    assert ssh.download_trees == [
        (
            "/u/skula/mcode-launch/experiments/results/smoke/artifacts",
            Path("experiments/results/smoke/artifacts"),
            300,
        )
    ]




def test_run_bench_on_bluevela_prefers_openai_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "SshClient", _FakeSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    _set_attempt_context(monkeypatch, timestamp=1777000002.0)
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
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    _set_attempt_context(monkeypatch, timestamp=1777000003.0)

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
        if cmd.startswith("STAT=$(bjobs -noheader -o stat "):
            return _FakeResult()
        if cmd.startswith("bkill "):
            return _FakeResult(stdout="Job <4242> is being terminated\n")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def download(self, src: str, dst: Path, *, timeout: float = 300.0) -> None:
        super().download(src, dst, timeout=timeout)
        raise RuntimeError("scp failed")

    def upload(self, src: Path, dst: str, *, timeout: float = 300.0) -> None:
        self.uploads.append((src.read_text(), dst, timeout))


class _LingeringJobSshClient(_FakeSshClient):
    def __init__(self, login: str) -> None:
        super().__init__(login)
        self.active_checks = 0

    def run(self, cmd: str, *, timeout: float = 60.0):
        if cmd.startswith("STAT=$(bjobs -noheader -o stat "):
            self.commands.append(cmd)
            self.active_checks += 1
            return _FakeResult(ok=self.active_checks > 1)
        if cmd.startswith("bkill "):
            self.commands.append(cmd)
            return _FakeResult(stdout="Job <4242> is being terminated\n")
        return super().run(cmd, timeout=timeout)


def test_run_bench_on_bluevela_keeps_recoverable_metadata_on_fetch_failure(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "state.json"
    attempt_token = _set_attempt_context(monkeypatch, timestamp=1777000004.0)
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(state_path))
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _RecoverableMetadataSshClient)
    monkeypatch.setattr(
        remote,
        "_stream_remote_log",
        lambda ssh, remote_log, *, exit_sentinel, job_id: (_ for _ in ()).throw(
            RuntimeError("ssh dropped")
        ),
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
    assert rec.remote["attempt_token"] == attempt_token
    assert rec.remote["remote_db"].endswith("/results.db")
    assert rec.remote["remote_log"].endswith(f"/logs/bench-{attempt_token}.log")
    assert rec.remote["exit_sentinel"].endswith(f"/exit-{attempt_token}.code")
    assert rec.remote["podman_svc_log"].endswith(
        f"/logs/podman-svc-{attempt_token}.log"
    )
    assert rec.remote["remote_script"].endswith(f"/bench-{attempt_token}.sh")
    assert rec.updated_at is not None
    ssh = _RecoverableMetadataSshClient.last
    assert ssh is not None
    assert ssh.downloads




def test_run_bench_on_bluevela_bkills_lingering_lsf_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", _LingeringJobSshClient)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    _set_attempt_context(monkeypatch, timestamp=1777000005.0)

    exit_code = remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    assert exit_code == 0
    ssh = _LingeringJobSshClient.last
    assert ssh is not None
    assert not any(cmd.startswith("bkill 4242") for cmd in ssh.commands)
    assert ssh.active_checks == 2


def test_stream_remote_log_stops_after_sentinel_even_if_bjob_stays_running(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        def __init__(self, argv, *, stdout, stderr) -> None:
            captured["argv"] = argv
            captured["stdout"] = stdout
            captured["stderr"] = stderr

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        remote.subprocess,
        "Popen",
        lambda argv, stdout, stderr: _FakeProc(argv, stdout=stdout, stderr=stderr),
    )

    ssh = type("_Ssh", (), {"login": "skula@login3.bluevela.rmf.ibm.com"})()
    remote_log = "/proj/dmfexp/skula/mcode-launch/bench-runs/r1/logs/bench-attempt.log"
    exit_sentinel = "/proj/dmfexp/skula/mcode-launch/bench-runs/r1/exit-attempt.code"
    remote._stream_remote_log(
        ssh,
        remote_log,
        exit_sentinel=exit_sentinel,
        job_id="4242",
    )

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "ssh"
    remote_cmd = argv[-1]
    assert f"tail -n 0 -F {remote_log}" in remote_cmd
    assert 'tail -F -n +1' not in remote_cmd
    assert f"test -f {exit_sentinel}" in remote_cmd
    assert 'bjobs -noheader -o stat 4242' in remote_cmd
    assert 'SENTINEL_SEEN=1' in remote_cmd
    assert 'WARNING: benchmark finished but LSF job 4242 is still $STAT; ' in remote_cmd
    assert f'"inspect {remote_log}" >&2' in remote_cmd

def test_bluevela_bench_waits_for_lsf_exit_before_bkill(tmp_path, monkeypatch) -> None:
    class SettlingSsh(_FakeSshClient):
        def __init__(self, login: str) -> None:
            super().__init__(login)
            self.active_checks = 0

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
            if cmd.startswith("STAT=$(bjobs -noheader -o stat "):
                self.active_checks += 1
                if self.active_checks == 1:
                    return _FakeResult(ok=False)
                return _FakeResult(ok=True)
            if cmd.startswith("bkill "):
                return _FakeResult(stdout="Job <4242> is being terminated\n")
            raise AssertionError(f"unexpected ssh command: {cmd}")

    monkeypatch.setattr(remote.launch_config, "load", lambda: _bluevela_cfg())
    monkeypatch.setattr(remote.launch_config, "validate_for_bluevela", lambda cfg: [])
    monkeypatch.setattr(remote, "_resolve_endpoint", lambda model, cfg: "http://host:8321/v1")
    monkeypatch.setattr(remote, "SshClient", SettlingSsh)
    monkeypatch.setattr(remote, "_stream_remote_log", _ignore_stream)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    _set_attempt_context(monkeypatch, timestamp=1777000002.0)

    exit_code = remote.run_bench_on_bluevela(
        bench_argv=["smoke", "--model", "Qwen/Qwen3.5-35B-A3B"],
        model="Qwen/Qwen3.5-35B-A3B",
        local_db=tmp_path / "results.db",
    )

    assert exit_code == 0
    ssh = SettlingSsh.last
    assert ssh is not None
    assert any(cmd.startswith("STAT=$(bjobs -noheader -o stat ") for cmd in ssh.commands)
    assert not any(cmd.startswith("bkill ") for cmd in ssh.commands)