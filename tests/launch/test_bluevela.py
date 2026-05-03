from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcode.launch import bluevela, profiles, state
from mcode.launch.config import LaunchConfig
from mcode.launch.models import LaunchError, LaunchSpec, ServerRecord, Target
from mcode.launch.progress import NullReporter, TransportError
from mcode.launch.ssh import SshResult


def _cfg(tmp_path: Path, *, group: str = "grp_runtime", queue: str = "normal") -> LaunchConfig:
    c = LaunchConfig()
    c.bluevela.login = "testuser@example.test"
    c.bluevela.group = group
    c.bluevela.queue_order = [queue]
    c.bluevela.workspace_root = str(tmp_path / "workspace")
    c.bluevela.shared_root = str(tmp_path / "shared")
    c.bluevela.hf_env = str(tmp_path / "hf-env.sh")
    c.bluevela.gpu_mode = "exclusive_process"
    return c


def _spec(model: str = "Qwen/Qwen3.5-27B") -> LaunchSpec:
    return LaunchSpec(target=Target.BLUEVELA, model=model, profile=profiles.resolve(model))


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SshResult:
    return SshResult(returncode=returncode, stdout=stdout, stderr=stderr, duration_s=0.01)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Qwen/Qwen3.5-27B", {"GPU_COUNT": "2", "flag": "qwen3_coder"}),
        ("google/gemma-4-31B-it", {"GPU_COUNT": "2", "flag": "--chat-template"}),
        ("MiniMaxAI/MiniMax-M2", {"GPU_COUNT": "4", "flag": "--enable_expert_parallel"}),
        ("ibm-granite/granite-4.0-h-small", {"GPU_COUNT": "1", "flag": "hermes"}),
    ],
)
def test_build_env_json_carries_profile_contract(
    tmp_path: Path, model: str, expected: dict
) -> None:
    env = bluevela.build_env_json(_spec(model), _cfg(tmp_path).bluevela, run_dir=str(tmp_path))
    assert env["MODEL"] == model
    assert env["GPU_COUNT"] == expected["GPU_COUNT"]
    assert expected["flag"] in env["VLLM_FLAGS"]


def test_parse_job_id_extracts_from_bsub_output() -> None:
    assert bluevela._parse_job_id("Job <871884> is submitted to queue <normal>.") == "871884"
    with pytest.raises(LaunchError, match="job id"):
        bluevela._parse_job_id("unexpected")


def test_bjobs_state_rejects_non_numeric_job_id() -> None:
    ssh = MagicMock()
    with pytest.raises(LaunchError):
        bluevela._bjobs_state(ssh, "1; rm -rf /")
    ssh.run.assert_not_called()


def test_pick_queue_tries_until_one_validates(tmp_path: Path) -> None:
    ssh = MagicMock()
    ssh.run.side_effect = [
        _result(),
        _result(returncode=255, stderr="closed\n"),
        _result(),
        _result(stdout="Job <10> submitted to queue <q2>.\n"),
        _result(),
    ]
    cfg = _cfg(tmp_path).bluevela
    cfg.queue_order = ["q1", "q2"]

    assert bluevela._pick_queue(ssh, cfg) == "q2"


def _successful_ssh() -> MagicMock:
    ssh = MagicMock()
    polls = {"n": 0}

    def run(cmd: str, *, timeout: float = 60.0):
        del timeout
        if cmd.startswith("bsub -H "):
            return _result(stdout="Job <100> submitted.\n")
        if cmd.startswith("bkill 100"):
            return _result()
        if cmd.startswith("mkdir -p"):
            return _result()
        if cmd.startswith("bsub -G"):
            return _result(stdout="Job <871884> is submitted to queue <normal>.\n")
        if "bjobs" in cmd:
            polls["n"] += 1
            return _result(stdout="RUN\n" if polls["n"] > 1 else "PEND\n")
        if "vllm_host.txt" in cmd:
            return _result(stdout="compute-host\n")
        if "curl" in cmd:
            return _result(stdout="200")
        return _result()

    ssh.run.side_effect = run
    ssh.upload = MagicMock()
    return ssh


def test_launch_happy_path_writes_server_record(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    server = bluevela.launch(
        _spec(),
        NullReporter.create(bluevela.PHASES),
        cfg=_cfg(tmp_path),
        state_path=state_path,
        ssh_client=_successful_ssh(),
    )

    assert server.target == Target.BLUEVELA
    assert server.job_id == "871884"
    assert server.endpoint == "http://compute-host:8321/v1"
    assert state.load(state_path).servers[0].metadata["queue"] == "normal"


@pytest.mark.parametrize(
    "target,group", [(Target.LOCAL_VLLM, "grp_runtime"), (Target.BLUEVELA, "")]
)
def test_launch_rejects_bad_target_or_config(tmp_path: Path, target: Target, group: str) -> None:
    spec = _spec()
    spec.target = target
    with pytest.raises(LaunchError):
        bluevela.launch(spec, NullReporter.create(bluevela.PHASES), cfg=_cfg(tmp_path, group=group))


def test_launch_fails_when_job_exits_before_ready(tmp_path: Path) -> None:
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        del timeout
        if cmd.startswith("bsub -H "):
            return _result(stdout="Job <100> submitted.\n")
        if cmd.startswith("bsub -G"):
            return _result(stdout="Job <999> is submitted to queue <normal>.\n")
        if "bjobs" in cmd:
            return _result(stdout="DONE\n")
        return _result()

    ssh.run.side_effect = run
    ssh.upload = MagicMock()
    with pytest.raises(LaunchError, match="before running|before endpoint became healthy|exited"):
        bluevela.launch(
            _spec(),
            NullReporter.create(bluevela.PHASES),
            cfg=_cfg(tmp_path),
            state_path=tmp_path / "state.json",
            ssh_client=ssh,
        )


def test_launch_surfaces_transport_error_with_hint(tmp_path: Path) -> None:
    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")
    with pytest.raises(LaunchError) as ei:
        bluevela.launch(
            _spec(), NullReporter.create(bluevela.PHASES), cfg=_cfg(tmp_path), ssh_client=ssh
        )
    assert "reach Blue Vela" in ei.value.what
    assert ei.value.next


@pytest.mark.parametrize(
    "lsf,http,initial,expected",
    [
        ("RUN", "200", "pending", "healthy"),
        ("RUN", "000", "healthy", "pending"),
        ("EXIT", "", "healthy", "failed"),
    ],
)
def test_refresh_maps_lsf_and_http_state(
    tmp_path: Path, lsf: str, http: str, initial: str, expected: str
) -> None:
    ssh = MagicMock()

    def run(cmd: str, *, timeout: float = 60.0):
        del timeout
        if cmd.startswith("bjobs"):
            return _result(stdout=f"{lsf}\n")
        if "curl" in cmd:
            return _result(stdout=http)
        return _result()

    ssh.run.side_effect = run
    server = ServerRecord(
        id="s",
        target=Target.BLUEVELA,
        endpoint="http://compute-host:8321/v1",
        model="m",
        config_hash="h",
        job_id="42",
        status=initial,
    )

    assert bluevela.refresh(server, cfg=_cfg(tmp_path), ssh_client=ssh).status == expected


def test_stop_bkills_and_drops_record(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state.update(
        state_path,
        lambda s: s.upsert_server(
            ServerRecord(
                id="server-x",
                target=Target.BLUEVELA,
                endpoint="x",
                model="m",
                config_hash="h",
                job_id="9000",
            )
        ),
    )
    ssh = MagicMock()
    ssh.run.return_value = _result()

    assert bluevela.stop("server-x", cfg=_cfg(tmp_path), state_path=state_path, ssh_client=ssh)
    assert any("bkill 9000" in c.args[0] for c in ssh.run.call_args_list)
    assert state.load(state_path).server("server-x") is None


def test_doctor_reports_config_and_ssh_errors(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.bluevela.group = ""
    assert bluevela.doctor(cfg, ssh_client=MagicMock())[0].ok is False

    ssh = MagicMock()
    ssh.run.side_effect = TransportError("Connection timed out")
    checks = bluevela.doctor(_cfg(tmp_path), ssh_client=ssh)
    assert checks[0].ok is True
    assert next(c for c in checks if c.name == "ssh reachable").ok is False
