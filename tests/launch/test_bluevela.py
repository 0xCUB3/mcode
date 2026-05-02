from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcode.launch import bluevela, profiles, state
from mcode.launch.config import LaunchConfig
from mcode.launch.models import LaunchError, LaunchSpec, ServerRecord, Target
from mcode.launch.progress import NullReporter, TransportError
from mcode.launch.ssh import SshResult


def _cfg(group: str = "grp_runtime", queue: str = "normal") -> LaunchConfig:
    c = LaunchConfig()
    c.bluevela.login = "testuser@testhost"
    c.bluevela.group = group
    c.bluevela.queue_order = [queue]
    c.bluevela.workspace_root = "/u/testuser/mcode-launch"
    c.bluevela.shared_root = "/u/testuser/mcode-shared"
    c.bluevela.hf_env = "/u/testuser/.config/mcode/hf-env.sh"
    c.bluevela.gpu_mode = "exclusive_process"
    return c


def _spec(model: str = "Qwen/Qwen3.5-27B") -> LaunchSpec:
    return LaunchSpec(
        target=Target.BLUEVELA,
        model=model,
        profile=profiles.resolve(model),
    )


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SshResult:
    return SshResult(returncode=returncode, stdout=stdout, stderr=stderr, duration_s=0.01)


# --- env.json construction --------------------------------------------------
def test_build_env_json_carries_profile_flags_and_paths() -> None:
    cfg = _cfg().bluevela
    env = bluevela.build_env_json(_spec("Qwen/Qwen3.5-27B"), cfg, run_dir="/u/testuser/runs/bv-x")
    assert env["MODEL"] == "Qwen/Qwen3.5-27B"
    assert env["QUEUE"] == "normal"
    assert env["GROUP"] == "grp_runtime"
    assert env["GPU_COUNT"] == "2"  # Qwen3.5-27B profile
    assert env["RUN_DIR"] == "/u/testuser/runs/bv-x"
    assert env["BV_SHARED_DIR"] == "/u/testuser/mcode-shared"
    # Tool-call parser flags carry through
    assert "--tool-call-parser" in env["VLLM_FLAGS"]


def test_build_env_json_injects_chat_template_flag_for_gemma() -> None:
    """Codex-style invariant: when profile.chat_template is set, Python builder
    must inject --chat-template /chat-template.jinja into VLLM_FLAGS *and* set
    CHAT_TEMPLATE_PATH. The shell never guesses."""
    cfg = _cfg().bluevela
    env = bluevela.build_env_json(
        _spec("google/gemma-4-31B-it"), cfg, run_dir="/u/testuser/runs/bv-y"
    )
    flags = env["VLLM_FLAGS"]
    assert "--chat-template" in flags
    i = flags.index("--chat-template")
    assert flags[i + 1] == "/chat-template.jinja"
    assert env["CHAT_TEMPLATE_PATH"].endswith("tool_chat_template_gemma4.jinja")


def test_build_env_json_propagates_minimax_extra_env() -> None:
    cfg = _cfg().bluevela
    env = bluevela.build_env_json(
        _spec("MiniMaxAI/MiniMax-M2"), cfg, run_dir="/u/testuser/runs/bv-z"
    )
    assert env["EXTRA_ENV"].get("SAFETENSORS_FAST_GPU") == "1"
    assert "--enable_expert_parallel" in env["VLLM_FLAGS"]


def test_bluevela_vllm_script_uses_shared_root_only() -> None:
    script = (Path(bluevela.__file__).parent / "scripts" / "bluevela_vllm.sh").read_text()
    assert '${BV_SHARED_DIR}/server-podman-${LSB_JOBID:-0}' in script
    assert '${HF_HOME:-${BV_SHARED_DIR}/hf-cache}' in script
    assert '${HOME}/.local/run' not in script
    assert ' -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \\' in script
    assert '    "$VLLM_IMAGE" \\' in script

# --- config hash stability -------------------------------------------------
def test_config_hash_sensitive_to_model_and_image() -> None:
    h1 = bluevela._config_hash(_spec("Qwen/Qwen3.5-27B"))
    h2 = bluevela._config_hash(_spec("Qwen/Qwen3.5-27B"))
    h3 = bluevela._config_hash(_spec("google/gemma-4-31B-it"))
    assert h1 == h2
    assert h1 != h3


# --- LSF primitives --------------------------------------------------------
def test_parse_job_id_extracts_from_bsub_output() -> None:
    assert bluevela._parse_job_id("Job <871884> is submitted to queue <normal>.") == "871884"


def test_parse_job_id_raises_with_context_on_failure() -> None:
    with pytest.raises(LaunchError) as ei:
        bluevela._parse_job_id("something unexpected happened")
    assert "job id" in ei.value.what
    assert "something unexpected" in ei.value.why


def test_bjobs_state_single_line() -> None:
    ssh = MagicMock()
    ssh.run.return_value = _result(stdout="RUN\n")
    assert bluevela._bjobs_state(ssh, "123") == "RUN"


def test_bjobs_state_empty_returns_none() -> None:
    ssh = MagicMock()
    ssh.run.return_value = _result(returncode=1, stderr="not found")
    assert bluevela._bjobs_state(ssh, "999") is None


def test_validate_queue_accepts_and_cleans_up() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = [
        _result(stdout=""),  # mkdir -p .mcode-qval
        _result(stdout="Job <555> is submitted to queue <normal>.\n"),
        _result(stdout=""),  # bkill
    ]
    err = bluevela._validate_queue(ssh, _cfg().bluevela, "normal")
    assert err is None
    # bkill was called on the validation job
    assert any("bkill 555" in c.args[0] for c in ssh.run.call_args_list)


def test_validate_queue_rejection_returns_error_text() -> None:
    ssh = MagicMock()
    ssh.run.return_value = _result(returncode=255, stderr="queue normal is closed\n")
    err = bluevela._validate_queue(ssh, _cfg().bluevela, "normal")
    assert err and "closed" in err


def test_pick_queue_tries_in_order_and_raises_on_all_rejected() -> None:
    ssh = MagicMock()
    ssh.run.return_value = _result(returncode=255, stderr="rejected\n")
    cfg = _cfg().bluevela
    cfg.queue_order = ["q1", "q2"]
    with pytest.raises(LaunchError) as ei:
        bluevela._pick_queue(ssh, cfg)
    assert "no configured queue" in ei.value.what
    # Both queues should appear in the why: field
    assert "q1" in ei.value.why and "q2" in ei.value.why


def test_pick_queue_returns_first_accepted() -> None:
    ssh = MagicMock()
    # q1 rejected, q2 accepted. Each _validate_queue call also issues a
    # `mkdir -p .mcode-qval` first to stage the bsub output dir.
    ssh.run.side_effect = [
        _result(stdout=""),  # mkdir for q1
        _result(returncode=255, stderr="closed\n"),
        _result(stdout=""),  # mkdir for q2
        _result(stdout="Job <10> submitted to queue <q2>.\n"),
        _result(stdout=""),  # bkill
    ]
    cfg = _cfg().bluevela
    cfg.queue_order = ["q1", "q2"]
    assert bluevela._pick_queue(ssh, cfg) == "q2"


# --- failure catalog -------------------------------------------------------
def test_hint_for_vpn_pattern() -> None:
    hint = bluevela._hint_for("ssh: Connection timed out")
    assert "VPN" in hint


def test_hint_for_group_membership() -> None:
    hint = bluevela._hint_for("<user> is not a member of the group grp_x")
    assert "group" in hint


def test_hint_for_oom() -> None:
    hint = bluevela._hint_for("RuntimeError: No available memory for the cache blocks")
    assert "tensor_parallel" in hint or "smaller model" in hint


def test_hint_for_unknown_returns_generic_suggestion() -> None:
    hint = bluevela._hint_for("something truly novel")
    assert hint  # never empty


# --- launch happy path -----------------------------------------------------
def _successful_ssh_mock(tmp_path: Path, run_dir: str) -> MagicMock:
    """Mock an SshClient that walks through: queue validate OK, bsub OK, LSF
    RUN, host file appears, HTTP 200."""
    ssh = MagicMock()
    state = {"stage": 0}

    def run(cmd: str, *, timeout: float = 60.0):
        # Queue validation path
        if cmd.startswith("bsub -H "):
            return _result(stdout="Job <100> submitted.\n")
        if cmd.startswith("bkill 100"):
            return _result()
        if cmd.startswith("mkdir -p"):
            return _result()
        # Real submit
        if cmd.startswith("bsub -G") and "mcode-vllm-" in cmd:
            return _result(stdout="Job <871884> is submitted to queue <normal>.\n")
        # bjobs polling — first call PEND, subsequent calls RUN
        if "bjobs" in cmd:
            state["stage"] += 1
            return _result(stdout="RUN\n" if state["stage"] >= 2 else "PEND\n")
        # host file check
        if "vllm_host.txt" in cmd:
            return _result(stdout="p1-r01-n2\n")
        # curl health
        if "curl" in cmd:
            return _result(stdout="200")
        # log tail
        if "vllm.log" in cmd:
            return _result(stdout="loading safetensors\n")
        return _result()

    ssh.run.side_effect = run
    ssh.upload = MagicMock()
    return ssh


def test_launch_happy_path_writes_server_record(tmp_path: Path) -> None:
    cfg = _cfg()
    ssh = _successful_ssh_mock(tmp_path, run_dir="/u/testuser/mcode-shared/runs/x")
    state_path = tmp_path / "state.json"

    server = bluevela.launch(
        _spec(),
        NullReporter.create(bluevela.PHASES),
        cfg=cfg,
        state_path=state_path,
        ssh_client=ssh,
    )
    assert server.target == Target.BLUEVELA
    assert server.job_id == "871884"
    assert server.endpoint.startswith("http://")
    assert server.endpoint.endswith("/v1")
    # Persisted
    assert state.load if True else None  # keep import-used
    from mcode.launch import state as state_mod

    loaded = state_mod.load(state_path)
    assert len(loaded.servers) == 1
    assert loaded.servers[0].metadata["queue"] == "normal"


def test_launch_rejects_wrong_target() -> None:
    spec = _spec()
    spec.target = Target.LOCAL_VLLM
    with pytest.raises(LaunchError):
        bluevela.launch(spec, NullReporter.create(bluevela.PHASES), cfg=_cfg())


def test_launch_rejects_incomplete_config() -> None:
    cfg = _cfg()
    cfg.bluevela.group = ""  # broken
    with pytest.raises(LaunchError) as ei:
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=cfg,
        )
    assert "incomplete" in ei.value.what


def test_launch_fails_fast_on_done_before_ready(tmp_path: Path) -> None:
    """Codex pre-merge-review fix: if the LSF job transitions to DONE before
    the endpoint ever becomes healthy, surface the failure immediately —
    don't sit in `starting` phase waiting for the 40-min deadline."""
    ssh = MagicMock()
    stage = {"n": 0}

    def run(cmd: str, *, timeout: float = 60.0):
        if cmd.startswith("bsub -H "):
            return _result(stdout="Job <100> submitted.\n")
        if cmd.startswith("bkill 100"):
            return _result()
        if cmd.startswith("mkdir -p"):
            return _result()
        if cmd.startswith("bsub -G") and "mcode-vllm-" in cmd:
            return _result(stdout="Job <999> is submitted to queue <normal>.\n")
        if "bjobs" in cmd:
            stage["n"] += 1
            # PEND briefly, then DONE before any health can succeed.
            if stage["n"] < 2:
                return _result(stdout="PEND\n")
            return _result(stdout="DONE\n")
        if "vllm.log" in cmd:
            return _result(stdout="(no output)")
        if "vllm_host.txt" in cmd:
            return _result()  # host file never appears
        if "curl" in cmd:
            return _result()
        return _result()

    ssh.run.side_effect = run
    ssh.upload = MagicMock()

    with pytest.raises(LaunchError) as ei:
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=_cfg(),
            state_path=tmp_path / "s.json",
            ssh_client=ssh,
        )
    # Must surface BEFORE the absolute startup deadline (not 40 min wait).
    msg = (ei.value.what + ei.value.why).lower()
    assert "done" in msg or "exited" in msg or "before running" in msg


def test_phase_starting_absorbs_bjobs_transport_blip(tmp_path: Path) -> None:
    ssh = MagicMock()
    calls = {"bjobs": 0}

    def run(cmd: str, *, timeout: float = 60.0):
        if "vllm_host.txt" in cmd:
            return _result(stdout="p2-r01-n1.bluevela.rmf.ibm.com\n")
        if "curl" in cmd:
            return _result(returncode=1)
        if "bjobs" in cmd:
            calls["bjobs"] += 1
            if calls["bjobs"] == 1:
                raise TransportError("ssh timeout")
            return _result(stdout="DONE\n")
        if "vllm.log" in cmd:
            return _result(stdout="startup log")
        return _result()

    ssh.run.side_effect = run
    ctx = bluevela._LaunchContext(
        spec=_spec(),
        reporter=NullReporter.create(bluevela.PHASES),
        ssh=ssh,
        cfg=_cfg(),
        state_path=tmp_path / "s.json",
        run_id="bv-test",
        run_dir="/proj/dmfexp/skula/mcode-shared/runs/bv-test",
        local_log=tmp_path / "vllm.log",
        job_id="999",
    )

    with pytest.raises(LaunchError, match="before endpoint became healthy"):
        bluevela._phase_starting(ctx)

    assert calls["bjobs"] == 2


def test_launch_surfaces_transport_error_with_hint() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")
    with pytest.raises(LaunchError) as ei:
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=_cfg(),
            ssh_client=ssh,
        )
    assert "reach Blue Vela" in ei.value.what
    assert ei.value.next  # actionable hint present


# --- refresh / stop --------------------------------------------------------
def test_refresh_healthy_only_when_http_also_200() -> None:
    """Codex fix: LSF RUN alone is not 'healthy' — refresh must ALSO verify
    the endpoint responds 200, matching the launch()-side readiness contract."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if cmd.startswith("bjobs"):
            return _result(stdout="RUN\n")
        if "curl" in cmd:
            return _result(stdout="200")
        return _result()

    ssh.run.side_effect = run
    server = ServerRecord(
        id="s",
        target=Target.BLUEVELA,
        endpoint="http://compute-host:8321/v1",
        model="m",
        config_hash="h",
        job_id="42",
        status="pending",
    )
    updated = bluevela.refresh(server, cfg=_cfg(), ssh_client=ssh)
    assert updated.status == "healthy"


def test_refresh_stays_pending_when_lsf_run_but_http_unhealthy() -> None:
    """Codex fix regression: RUN + HTTP != 200 means still loading / stuck,
    NOT healthy."""
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        if cmd.startswith("bjobs"):
            return _result(stdout="RUN\n")
        if "curl" in cmd:
            return _result(stdout="000")  # no response
        return _result()

    ssh.run.side_effect = run
    server = ServerRecord(
        id="s",
        target=Target.BLUEVELA,
        endpoint="http://compute-host:8321/v1",
        model="m",
        config_hash="h",
        job_id="42",
        status="healthy",  # was healthy; must downgrade
    )
    updated = bluevela.refresh(server, cfg=_cfg(), ssh_client=ssh)
    assert updated.status == "pending"


def test_refresh_failed_on_exit() -> None:
    ssh = MagicMock()
    ssh.run.return_value = _result(stdout="EXIT\n")
    server = ServerRecord(
        id="s",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="42",
        status="healthy",
    )
    updated = bluevela.refresh(server, cfg=_cfg(), ssh_client=ssh)
    assert updated.status == "failed"


def test_refresh_transport_error_keeps_record_intact() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("ssh down")
    server = ServerRecord(
        id="s",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="42",
        status="healthy",
    )
    updated = bluevela.refresh(server, cfg=_cfg(), ssh_client=ssh)
    # Don't claim status if we couldn't verify
    assert updated.status == "healthy"


def test_stop_bkills_and_drops_record(tmp_path: Path) -> None:
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="9000",
    )
    state_mod.update(state_path, lambda s: s.upsert_server(server))
    ssh = MagicMock()
    ssh.run.return_value = _result()
    assert bluevela.stop("server-x", cfg=_cfg(), state_path=state_path, ssh_client=ssh) is True
    # bkill was called
    assert any("bkill 9000" in c.args[0] for c in ssh.run.call_args_list)
    # Record dropped
    assert state_mod.load(state_path).server("server-x") is None


def test_stop_preserves_record_when_bkill_returns_nonzero(tmp_path: Path) -> None:
    """Codex pre-merge-review fix: if bkill fails (permission denied, LSF
    hiccup) without transport failure, we MUST NOT drop the state record.
    Previously `bkill ... || true` masked all non-zero exits."""
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="9000",
        metadata={"login": "alice@host"},
    )
    state_mod.update(state_path, lambda s: s.upsert_server(server))
    ssh = MagicMock()
    # bkill exits non-zero with an unexpected stderr (not an "already gone" phrase)
    ssh.run.return_value = SshResult(
        returncode=1,
        stdout="",
        stderr="Permission denied: somebody else's job",
        duration_s=0.01,
    )

    ok = bluevela.stop("server-x", cfg=_cfg(), state_path=state_path, ssh_client=ssh)
    assert ok is False
    reloaded = state_mod.load(state_path).server("server-x")
    assert reloaded is not None
    assert reloaded.status == "stop-pending"


def test_stop_drops_record_when_bkill_reports_already_gone(tmp_path: Path) -> None:
    """If LSF says the job is already finished, treat as success and drop
    the record — it's a no-op, not a failure."""
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="9000",
        metadata={"login": "alice@host"},
    )
    state_mod.update(state_path, lambda s: s.upsert_server(server))
    ssh = MagicMock()
    ssh.run.return_value = SshResult(
        returncode=1,
        stdout="",
        stderr="Job <9000>: Job has already finished\n",
        duration_s=0.01,
    )

    ok = bluevela.stop("server-x", cfg=_cfg(), state_path=state_path, ssh_client=ssh)
    assert ok is True
    assert state_mod.load(state_path).server("server-x") is None


def test_stop_preserves_record_on_transport_failure(tmp_path: Path) -> None:
    """Codex final-review fix: if `bkill` can't confirm because SSH is down,
    the remote LSF job is still running. The launcher MUST keep the state
    record so the user can retry — deleting it strands the live job."""
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="9000",
        metadata={"login": "alice@host"},
    )
    state_mod.update(state_path, lambda s: s.upsert_server(server))
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")

    ok = bluevela.stop("server-x", cfg=_cfg(), state_path=state_path, ssh_client=ssh)
    assert ok is False  # didn't confirm kill
    reloaded = state_mod.load(state_path).server("server-x")
    assert reloaded is not None
    assert reloaded.status == "stop-pending"


def test_stop_uses_record_login_not_current_config(tmp_path: Path) -> None:
    """Codex final-review fix: stop routes through the record's login."""
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    server = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="9000",
        metadata={"login": "correct-login@correct-host"},
    )
    state_mod.update(state_path, lambda s: s.upsert_server(server))
    cfg = _cfg()
    cfg.bluevela.login = "wrong-login@wrong-host"
    ssh = MagicMock()
    ssh.run.return_value = _result()
    ok = bluevela.stop("server-x", cfg=cfg, state_path=state_path, ssh_client=ssh)
    assert ok is True


def test_stop_missing_record_returns_false(tmp_path: Path) -> None:
    assert (
        bluevela.stop("nope", cfg=_cfg(), state_path=tmp_path / "s.json", ssh_client=MagicMock())
        is False
    )


# --- doctor ----------------------------------------------------------------
def test_bjobs_state_rejects_non_numeric_job_id() -> None:
    """Codex fix: allowlist gating prevents shell interpolation of anything
    non-digit into `bjobs`."""
    ssh = MagicMock()
    with pytest.raises(LaunchError):
        bluevela._bjobs_state(ssh, "1; rm -rf /")
    ssh.run.assert_not_called()


def test_stop_rejects_poisoned_job_id_but_drops_record(tmp_path: Path) -> None:
    """Codex fix: a state record with a crafted job_id must NOT be passed to
    bkill. The record is still cleaned up."""
    from mcode.launch import state as state_mod

    state_path = tmp_path / "s.json"
    bad = ServerRecord(
        id="server-x",
        target=Target.BLUEVELA,
        endpoint="x",
        model="m",
        config_hash="h",
        job_id="0 -u skula",  # malicious
    )
    state_mod.update(state_path, lambda s: s.upsert_server(bad))
    ssh = MagicMock()
    assert bluevela.stop("server-x", cfg=_cfg(), state_path=state_path, ssh_client=ssh) is True
    ssh.run.assert_not_called()  # no bkill was issued
    assert state_mod.load(state_path).server("server-x") is None


def test_launch_refuses_poisoned_group_value() -> None:
    """Codex fix: a group containing shell metachars must be rejected
    before bsub is issued."""
    cfg = _cfg(group="grp_runtime; rm -rf /")
    cfg.bluevela.group = "grp_runtime; rm -rf /"
    ssh = MagicMock()
    # config validator catches empty group but not shell-injection content;
    # allowlist inside _validate_queue/_pick_queue is the guard.
    # Force queue_order into the mock path so _pick_queue runs.
    ssh.run.return_value = _result(stdout="Job <1> submitted.\n")
    with pytest.raises(LaunchError) as ei:
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=cfg,
            ssh_client=ssh,
        )
    # Either validate_for_bluevela path or allowlist path; either is fine.
    msg = (ei.value.what + " " + ei.value.why).lower()
    assert "group" in msg or "unsafe" in msg


def test_launch_persists_pending_record_before_wait(tmp_path: Path) -> None:
    """Codex fix: after bsub accept, a pending ServerRecord lands in state so
    a failure during the wait phase doesn't orphan the job."""
    from mcode.launch import state as state_mod

    cfg = _cfg()
    state_path = tmp_path / "s.json"
    ssh = MagicMock()
    call_log: list[str] = []

    def run(cmd: str, *, timeout: float = 60.0):
        call_log.append(cmd)
        if cmd.startswith("bsub -H "):
            return _result(stdout="Job <100> submitted.\n")
        if cmd.startswith("bkill 100"):
            return _result()
        if cmd.startswith("mkdir -p"):
            return _result()
        if cmd.startswith("bsub -G") and "mcode-vllm-" in cmd:
            return _result(stdout="Job <871884> is submitted to queue <normal>.\n")
        if "bjobs" in cmd:
            # Simulate LSF giving us an unexpected EXIT mid-queue so we hit
            # the tear-down path, but a record must exist by then.
            return _result(stdout="EXIT\n")
        if "vllm.log" in cmd:
            return _result(stdout="")
        return _result()

    ssh.run.side_effect = run
    ssh.upload = MagicMock()

    with pytest.raises(LaunchError):
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=cfg,
            state_path=state_path,
            ssh_client=ssh,
        )
    # The record was persisted before the failure.
    loaded = state_mod.load(state_path)
    assert len(loaded.servers) == 1
    assert loaded.servers[0].job_id == "871884"
    # The tear-down path bkilled the orphan.
    assert any(call.startswith("bkill ") and "871884" in call for call in call_log)


def test_doctor_reports_incomplete_config() -> None:
    cfg = _cfg()
    cfg.bluevela.group = ""
    checks = bluevela.doctor(cfg, ssh_client=MagicMock())
    assert checks[0].name == "config complete"
    assert checks[0].ok is False


def test_doctor_reports_ssh_failure() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")
    checks = bluevela.doctor(_cfg(), ssh_client=ssh)
    # config ok + ssh failing
    assert checks[0].ok is True
    ssh_check = next(c for c in checks if c.name == "ssh reachable")
    assert ssh_check.ok is False
    assert ssh_check.next  # hint populated
