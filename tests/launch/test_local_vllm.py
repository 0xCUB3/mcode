from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcode.launch import local_vllm, profiles, state
from mcode.launch.config import LaunchConfig
from mcode.launch.models import LaunchError, LaunchSpec, Target
from mcode.launch.progress import NullReporter


def _spec(model: str = "Qwen/Qwen2.5-0.5B") -> LaunchSpec:
    return LaunchSpec(
        target=Target.LOCAL_VLLM,
        model=model,
        profile=profiles.resolve(model),
    )


def test_build_vllm_argv_contains_profile_flags() -> None:
    spec = _spec("Qwen/Qwen3.5-27B")
    argv = local_vllm._build_vllm_argv(spec, port=8000)
    # Core args
    assert "--model" in argv
    assert "Qwen/Qwen3.5-27B" in argv
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"  # Qwen3.5-27B profile
    # Tool-call parser flags from profile.flags
    assert "--tool-call-parser" in argv
    assert "qwen3_coder" in argv


def test_build_vllm_argv_injects_chat_template_when_file_exists(tmp_path: Path) -> None:
    """Gemma4's chat_template must show up as a --chat-template flag when the
    bundled resource exists locally."""
    # Create a fake bundled template at the resources dir
    resources = Path(local_vllm.__file__).parent / "resources"
    resources.mkdir(exist_ok=True)
    tmpl = resources / "tool_chat_template_gemma4.jinja"
    created_here = False
    if not tmpl.exists():
        tmpl.write_text("{# placeholder #}")
        created_here = True
    try:
        spec = _spec("google/gemma-4-31B-it")
        argv = local_vllm._build_vllm_argv(spec, port=8000)
        assert "--chat-template" in argv
        assert str(tmpl) in argv
    finally:
        if created_here:
            tmpl.unlink()


def test_config_hash_is_stable_across_calls() -> None:
    a = local_vllm._config_hash(_spec("Qwen/Qwen3.5-27B"))
    b = local_vllm._config_hash(_spec("Qwen/Qwen3.5-27B"))
    assert a == b
    c = local_vllm._config_hash(_spec("google/gemma-4-31B-it"))
    assert a != c


def test_launch_rejects_wrong_target() -> None:
    spec = _spec()
    spec.target = Target.BLUEVELA
    with pytest.raises(LaunchError) as ei:
        local_vllm.launch(spec, NullReporter.create(local_vllm.PHASES))
    assert "wrong target" in ei.value.what


@patch("mcode.launch.local_vllm._process_identity", return_value="fake-identity")
@patch("mcode.launch.local_vllm.subprocess.Popen")
@patch("mcode.launch.local_vllm._health_check")
def test_launch_happy_path_writes_server_record(
    mock_health, mock_popen, _mock_id, tmp_path: Path
) -> None:
    # Port is free initially, then /v1/models returns 200 on the first poll.
    mock_health.side_effect = [(False, 0), (True, 200)]
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None  # still alive
    mock_popen.return_value = proc

    state_path = tmp_path / "state.json"
    cfg = LaunchConfig()
    spec = _spec()
    reporter = NullReporter.create(local_vllm.PHASES)
    server = local_vllm.launch(spec, reporter, cfg=cfg, state_path=state_path)

    assert server.target == Target.LOCAL_VLLM
    assert server.endpoint == f"http://127.0.0.1:{cfg.local_vllm.port}/v1"
    assert server.job_id == "12345"
    # Codex fix: proc_identity is captured at launch for PID-reuse safety.
    assert server.metadata.get("proc_identity") == "fake-identity"

    # Persisted
    loaded = state.load(state_path)
    assert len(loaded.servers) == 1
    assert loaded.servers[0].endpoint == server.endpoint


@patch("mcode.launch.local_vllm._health_check")
def test_launch_refuses_when_port_already_serving(mock_health, tmp_path: Path) -> None:
    mock_health.return_value = (True, 200)  # port busy
    with pytest.raises(LaunchError) as ei:
        local_vllm.launch(
            _spec(),
            NullReporter.create(local_vllm.PHASES),
            cfg=LaunchConfig(),
            state_path=tmp_path / "s.json",
        )
    assert "already in use" in ei.value.what


@patch("mcode.launch.local_vllm._process_identity", return_value="fake-id")
@patch("mcode.launch.local_vllm.subprocess.Popen")
@patch("mcode.launch.local_vllm._health_check")
def test_launch_fails_cleanly_when_vllm_exits_early(
    mock_health, mock_popen, _mock_id, tmp_path: Path
) -> None:
    mock_health.return_value = (False, 0)  # never ready
    proc = MagicMock()
    proc.pid = 999
    proc.returncode = 2
    proc.poll.return_value = 2  # exited early
    mock_popen.return_value = proc

    with pytest.raises(LaunchError) as ei:
        local_vllm.launch(
            _spec(),
            NullReporter.create(local_vllm.PHASES),
            cfg=LaunchConfig(),
            state_path=tmp_path / "s.json",
        )
    assert "exited before" in ei.value.what
    assert ei.value.next  # hint populated


@patch("mcode.launch.local_vllm.subprocess.Popen", side_effect=FileNotFoundError("vllm"))
def test_launch_hint_when_vllm_not_on_path(_mock_popen, tmp_path: Path) -> None:
    with (
        patch("mcode.launch.local_vllm._health_check", return_value=(False, 0)),
        pytest.raises(LaunchError) as ei,
    ):
        local_vllm.launch(
            _spec(),
            NullReporter.create(local_vllm.PHASES),
            cfg=LaunchConfig(),
            state_path=tmp_path / "s.json",
        )
    assert "vllm binary not found" in ei.value.what
    assert "uv pip install vllm" in ei.value.next


def test_startup_hint_classifies_oom() -> None:
    hint = local_vllm._startup_hint("error: no available memory for the cache blocks", _spec())
    assert "too large" in hint or "smaller model" in hint


def test_startup_hint_classifies_parser() -> None:
    hint = local_vllm._startup_hint("ValueError: invalid tool call parser", _spec())
    assert "parser" in hint and "profiles.py" in hint


def test_stop_returns_false_for_missing_server(tmp_path: Path) -> None:
    assert local_vllm.stop("no-such-id", state_path=tmp_path / "s.json") is False


def test_stop_removes_record_and_signals_pid(tmp_path: Path) -> None:
    # Use a short-lived real subprocess to avoid patching os.kill.
    import subprocess
    import sys

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state_path = tmp_path / "s.json"
        from mcode.launch.models import ServerRecord

        server = ServerRecord(
            id="server-test",
            target=Target.LOCAL_VLLM,
            endpoint="http://127.0.0.1:8000/v1",
            model="x",
            config_hash="h",
            job_id=str(sleeper.pid),
        )
        state.update(state_path, lambda s: s.upsert_server(server))
        assert local_vllm.stop("server-test", state_path=state_path, grace_s=2.0) is True
        # Record gone from state
        assert state.load(state_path).server("server-test") is None
        # Child process should be dead within a short window
        try:
            sleeper.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            sleeper.kill()
            pytest.fail("local_vllm.stop did not terminate the child process")
        assert sleeper.returncode is not None
    finally:
        if sleeper.poll() is None:
            sleeper.kill()


def test_refresh_marks_stopped_when_pid_gone() -> None:
    from mcode.launch.models import ServerRecord

    server = ServerRecord(
        id="s",
        target=Target.LOCAL_VLLM,
        endpoint="x",
        model="m",
        config_hash="h",
        # A pid that's extremely unlikely to exist
        job_id="2147480000",
        status="healthy",
    )
    updated = local_vllm.refresh(server)
    assert isinstance(updated, ServerRecord)
    assert updated.status == "stopped"


def test_refresh_detects_pid_reuse() -> None:
    """Codex fix: a recorded proc_identity that doesn't match the live pid's
    identity means the pid was reused. Must flip to stopped and NOT interfere
    with the unrelated process."""
    from mcode.launch.models import ServerRecord

    server = ServerRecord(
        id="s",
        target=Target.LOCAL_VLLM,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="1",  # init — always exists on Unix but identity won't match
        status="healthy",
        metadata={"proc_identity": "some-different-lstart-etime-value"},
    )
    updated = local_vllm.refresh(server)
    assert isinstance(updated, ServerRecord)
    assert updated.status == "stopped"


def test_chat_template_missing_hard_fails() -> None:
    """Codex fix: if profile requires a chat_template but it's not found,
    launch must fail closed, not silently drop the flag."""
    resources = Path(local_vllm.__file__).parent / "resources"
    tmpl = resources / "tool_chat_template_gemma4.jinja"
    created_by_other_test = tmpl.exists()
    if created_by_other_test:
        # Temporarily move it aside
        backup = tmpl.with_suffix(".jinja.bak")
        tmpl.rename(backup)
    try:
        with pytest.raises(LaunchError) as ei:
            local_vllm._build_vllm_argv(_spec("google/gemma-4-31B-it"), port=8000)
        assert "chat template" in ei.value.what
        assert "not found" in ei.value.what
    finally:
        if created_by_other_test:
            backup.rename(tmpl)


def test_launch_tears_down_child_on_persistence_failure(tmp_path: Path) -> None:
    """Codex fix: if state.update() fails after the server is healthy, the
    detached child must be killed — otherwise we orphan a live vLLM process."""
    killed: list[int] = []

    def fake_terminate(pid, *, expected_identity, grace_s):
        killed.append(pid)
        return True

    with (
        patch("mcode.launch.local_vllm._health_check", side_effect=[(False, 0), (True, 200)]),
        patch("mcode.launch.local_vllm.subprocess.Popen") as mock_popen,
        patch("mcode.launch.local_vllm._process_identity", return_value="id-xyz"),
        patch("mcode.launch.local_vllm._terminate_pid", side_effect=fake_terminate),
        patch("mcode.launch.local_vllm.state.update", side_effect=OSError("disk full")),
    ):
        proc = MagicMock()
        proc.pid = 55555
        proc.poll.return_value = None
        mock_popen.return_value = proc

        with pytest.raises(OSError):
            local_vllm.launch(
                _spec(),
                NullReporter.create(local_vllm.PHASES),
                cfg=LaunchConfig(),
                state_path=tmp_path / "s.json",
            )
    assert 55555 in killed


def test_doctor_reports_port_status_and_vllm_presence() -> None:
    cfg = LaunchConfig()
    checks = local_vllm.doctor(cfg)
    # Structure: at least vllm-importable + port-free
    names = [c.name for c in checks]
    assert any("vllm importable" in n for n in names)
    assert any("port" in n for n in names)


def test_server_record_metadata_is_json_serializable() -> None:
    """argv can contain values with whitespace; ensure state.py survives."""
    from dataclasses import asdict

    from mcode.launch.models import ServerRecord

    server = ServerRecord(
        id="s",
        target=Target.LOCAL_VLLM,
        endpoint="http://127.0.0.1:8000/v1",
        model="x",
        config_hash="h",
        metadata={"port": 8000, "argv": ["vllm", "--model", "x y z"]},
    )
    # This is what state._save does under the hood
    payload = asdict(server)
    json.dumps(payload, default=str)  # must not raise
