from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcode.launch import state
from mcode.launch.cli import app
from mcode.launch.models import LaunchError, ServerRecord, Target


@pytest.fixture
def runner() -> CliRunner:
    # Typer 0.21 CliRunner doesn't support mix_stderr; stderr and stdout
    # both land in result.output.
    return CliRunner()


def _all_output(result) -> str:
    """Concatenate every captured stream into one string for substring asserts."""
    parts: list[str] = []
    for name in ("stdout", "stderr", "output"):
        try:
            val = getattr(result, name)
        except (AttributeError, ValueError):
            continue
        if val:
            parts.append(val)
    return "\n".join(parts)


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "state.json"
    monkeypatch.setenv("MCODE_LAUNCH_STATE", str(p))
    return p


def test_status_empty(runner: CliRunner, isolated_state: Path) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "no servers or runs" in result.stdout


def test_status_lists_server(runner: CliRunner, isolated_state: Path) -> None:
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="server-abc",
                target=Target.LOCAL_VLLM,
                endpoint="http://127.0.0.1:8000/v1",
                model="Qwen/Qwen2.5-0.5B",
                config_hash="h",
                status="healthy",
            )
        ),
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "server-abc" in result.stdout
    assert "Qwen/Qwen2.5-0.5B" in result.stdout


def test_status_json_mode(runner: CliRunner, isolated_state: Path) -> None:
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="server-xyz",
                target=Target.BLUEVELA,
                endpoint="http://host:8321/v1",
                model="m",
                config_hash="h",
            )
        ),
    )
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    import json as _json

    data = _json.loads(result.stdout)
    assert len(data["servers"]) == 1
    assert data["servers"][0]["id"] == "server-xyz"


def test_error_formatting_prints_what_why_next(
    runner: CliRunner, isolated_state: Path, monkeypatch
) -> None:
    """The LaunchError contract: what / why / next / logs to stderr, exit 1."""

    def fake_launch(*a, **kw):
        raise LaunchError(
            what="simulated failure",
            why="testing",
            next="run the test again",
            logs="/tmp/fake.log",
        )

    monkeypatch.setattr("mcode.launch.cli.local_vllm.launch", fake_launch)
    result = runner.invoke(app, ["local-vllm", "--model", "Qwen/Qwen2.5-0.5B"])
    assert result.exit_code == 1
    err = _all_output(result)
    assert "✗" in err
    assert "simulated failure" in err
    assert "why:" in err and "testing" in err
    assert "next:" in err and "run the test again" in err
    assert "logs:" in err and "/tmp/fake.log" in err


def test_mcode_debug_env_raises_traceback(runner: CliRunner, monkeypatch) -> None:
    monkeypatch.setenv("MCODE_DEBUG", "1")

    def fake_launch(*a, **kw):
        raise LaunchError(what="x", why="y", next="z")

    monkeypatch.setattr("mcode.launch.cli.local_vllm.launch", fake_launch)
    result = runner.invoke(app, ["local-vllm", "--model", "Qwen/Qwen2.5-0.5B"])
    # Typer/CliRunner captures the exception rather than formatting it.
    assert result.exit_code != 0
    assert result.exception is not None
    assert isinstance(result.exception, LaunchError)


def test_stop_all_scoped_to_local_servers(
    runner: CliRunner, isolated_state: Path, monkeypatch
) -> None:
    """--all must only act on recorded servers, never a blanket bkill."""
    stopped: list[str] = []

    def _fake_stop(sid):
        stopped.append(sid)
        return True

    monkeypatch.setattr("mcode.launch.cli.local_vllm.stop", _fake_stop)
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="s1",
                target=Target.LOCAL_VLLM,
                endpoint="x",
                model="m",
                config_hash="h",
            )
        ),
    )
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="s2",
                target=Target.LOCAL_VLLM,
                endpoint="y",
                model="m",
                config_hash="h",
            )
        ),
    )
    result = runner.invoke(app, ["stop", "--all"])
    assert result.exit_code == 0
    assert set(stopped) == {"s1", "s2"}


def test_stop_nonexistent_id_exits_with_hint(runner: CliRunner, isolated_state: Path) -> None:
    result = runner.invoke(app, ["stop", "nope"])
    assert result.exit_code == 1
    assert "nope" in _all_output(result)


def test_doctor_local_vllm_reports_checks(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "local-vllm"])
    # Doctor can exit 1 if vllm isn't installed on the host; we just check
    # the output shape (at least one ✓ or ✗ marker).
    assert any(mark in result.stdout for mark in ("✓", "✗"))


def test_doctor_unknown_target_errors(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "totally-not-a-target"])
    assert result.exit_code == 1
    assert "unknown target" in _all_output(result)


def test_logs_for_bluevela_prints_ssh_hint(runner: CliRunner, isolated_state: Path) -> None:
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="bv-1",
                target=Target.BLUEVELA,
                endpoint="http://h:8321/v1",
                model="m",
                config_hash="h",
                log_path="/u/user/runs/bv-1/vllm.log",
                metadata={"login": "user@host"},
            )
        ),
    )
    result = runner.invoke(app, ["logs", "bv-1"])
    assert result.exit_code == 0
    assert "ssh user@host" in result.stdout
    assert "/vllm.log" in result.stdout


def test_stop_bluevela_transport_failure_exits_nonzero(
    runner: CliRunner, isolated_state: Path, monkeypatch
) -> None:
    """Codex verification-pass fix: when bluevela.stop() returns False
    (transport failure, record kept as stop-pending), the CLI must surface
    that as a failure — not print 'stopped: ...' and exit 0."""
    monkeypatch.setattr("mcode.launch.cli.bluevela.stop", lambda sid, **_kw: False)
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="bv-1",
                target=Target.BLUEVELA,
                endpoint="x",
                model="m",
                config_hash="h",
                job_id="123",
                metadata={"login": "a@b"},
            )
        ),
    )
    # Write a valid TOML so config loads — the point is the False return path.
    cfg_path = isolated_state.parent / "launch.toml"
    cfg_path.write_text(
        '[bluevela]\nlogin = "a@b"\nworkspace_root = "/u/x"\nshared_root = "/u/y"\n'
        'queue_order = ["normal"]\ngroup = "g"\n'
    )
    monkeypatch.setenv("MCODE_LAUNCH_CONFIG", str(cfg_path))
    result = runner.invoke(app, ["stop", "bv-1"])
    assert result.exit_code == 1
    out = _all_output(result)
    assert "could not confirm" in out or "stop-pending" in out
    assert "stopped: bv-1" not in out


def test_stop_local_target_works_even_when_bluevela_config_broken(
    runner: CliRunner, isolated_state: Path, tmp_path: Path, monkeypatch
) -> None:
    """Codex verification-pass fix: a malformed [bluevela] TOML must NOT
    block stopping a local-vllm server."""
    bad_cfg = tmp_path / "launch.toml"
    bad_cfg.write_text("not = = toml\n")  # malformed
    monkeypatch.setenv("MCODE_LAUNCH_CONFIG", str(bad_cfg))

    stopped: list[str] = []

    def _fake_stop(sid):
        stopped.append(sid)
        return True

    monkeypatch.setattr("mcode.launch.cli.local_vllm.stop", _fake_stop)
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="lv-1",
                target=Target.LOCAL_VLLM,
                endpoint="x",
                model="m",
                config_hash="h",
            )
        ),
    )
    result = runner.invoke(app, ["stop", "lv-1"])
    assert result.exit_code == 0
    assert stopped == ["lv-1"]


def test_doctor_init_writes_config(
    runner: CliRunner, isolated_state: Path, tmp_path: Path, monkeypatch
) -> None:
    written: dict[str, Path] = {}

    def fake_init(*, login, cfg_path=None, **_):
        p = tmp_path / "launch.toml"
        p.write_text("[bluevela]\nlogin = '" + login + "'\n")
        written["path"] = p
        return p

    monkeypatch.setattr("mcode.launch.cli.bluevela.doctor_init", fake_init)
    result = runner.invoke(app, ["doctor", "bluevela", "--init", "--login", "alice@host"])
    assert result.exit_code == 0
    assert "wrote" in result.stdout
    assert written["path"].exists()


def test_doctor_init_rejects_non_bluevela_target(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "local-vllm", "--init"])
    assert result.exit_code == 1
    assert "only supported for `bluevela`" in _all_output(result)


def test_refresh_walks_state(runner: CliRunner, isolated_state: Path, monkeypatch) -> None:
    refreshed: list[str] = []

    def fake_refresh(srv, *a, **kw):
        refreshed.append(srv.id)
        srv.status = "healthy"
        return srv

    monkeypatch.setattr("mcode.launch.cli.local_vllm.refresh", fake_refresh)
    state.update(
        isolated_state,
        lambda s: s.upsert_server(
            ServerRecord(
                id="s1",
                target=Target.LOCAL_VLLM,
                endpoint="x",
                model="m",
                config_hash="h",
                status="pending",
            )
        ),
    )
    result = runner.invoke(app, ["refresh"])
    assert result.exit_code == 0
    assert "refreshed 1" in result.stdout
    assert "s1" in refreshed
