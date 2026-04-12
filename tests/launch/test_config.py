from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcode.launch import config as cfg_mod
from mcode.launch.models import LaunchError


def test_load_missing_returns_defaults(tmp_path: Path) -> None:
    cfg = cfg_mod.load(tmp_path / "nope.toml")
    assert cfg.bluevela.queue_order == ["normal"]
    assert cfg.bluevela.gpu_mode == "exclusive_process"
    assert cfg.local_vllm.port == 8000
    assert cfg.source is None


def test_load_parses_all_sections(tmp_path: Path) -> None:
    p = tmp_path / "launch.toml"
    p.write_text(
        """
[bluevela]
login = "alice@login3.bluevela.rmf.ibm.com"
workspace_root = "~/mcode-launch"
shared_root = "~/mcode-shared"
queue_order = ["normal", "short"]
group = "grp_runtime"
gpu_mode = "shared"
hf_env = "~/.config/mcode/hf-env.sh"

[bluevela.podman]
graphroot_base = "/proj/x/podman/graph"

[local_vllm]
port = 8321

[local_ollama]
host = "0.0.0.0"
port = 11500
"""
    )
    cfg = cfg_mod.load(p)
    assert cfg.bluevela.login == "alice@login3.bluevela.rmf.ibm.com"
    assert cfg.bluevela.queue_order == ["normal", "short"]
    assert cfg.bluevela.group == "grp_runtime"
    assert cfg.bluevela.gpu_mode == "shared"
    # ~ is expanded
    assert cfg.bluevela.workspace_root == os.path.expanduser("~/mcode-launch")
    assert cfg.bluevela.podman.graphroot_base == "/proj/x/podman/graph"
    assert cfg.local_vllm.port == 8321
    assert cfg.local_ollama.host == "0.0.0.0"
    assert cfg.local_ollama.port == 11500
    assert cfg.source == p


def test_malformed_toml_raises_launch_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("not = = toml\n")
    with pytest.raises(LaunchError) as ei:
        cfg_mod.load(p)
    err = ei.value
    assert "not valid TOML" in err.what
    assert "doctor --init" in err.next


def test_save_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "launch.toml"
    src.write_text(
        """
[bluevela]
login = "alice@host"
workspace_root = "/tmp/ws"
shared_root = "/tmp/shared"
queue_order = ["normal"]
group = "grp_x"
gpu_mode = "exclusive_process"
hf_env = "/tmp/hf.sh"
"""
    )
    cfg = cfg_mod.load(src)
    dst = tmp_path / "out.toml"
    cfg_mod.save(cfg, dst)
    reloaded = cfg_mod.load(dst)
    assert reloaded.bluevela.login == "alice@host"
    assert reloaded.bluevela.group == "grp_x"
    assert reloaded.bluevela.queue_order == ["normal"]
    assert reloaded.bluevela.gpu_mode == "exclusive_process"


def test_save_rejects_unquotable_queue_name(tmp_path: Path) -> None:
    cfg = cfg_mod.LaunchConfig()
    cfg.bluevela.queue_order = ['bad"queue']
    with pytest.raises(LaunchError):
        cfg_mod.save(cfg, tmp_path / "x.toml")


def test_validate_reports_missing_fields() -> None:
    cfg = cfg_mod.LaunchConfig()
    errs = cfg_mod.validate_for_bluevela(cfg)
    assert any("login" in e for e in errs)
    assert any("group" in e for e in errs)
    assert any("workspace_root" in e for e in errs)


def test_validate_catches_bad_gpu_mode() -> None:
    cfg = cfg_mod.LaunchConfig()
    cfg.bluevela.login = "a@b"
    cfg.bluevela.group = "g"
    cfg.bluevela.workspace_root = "/x"
    cfg.bluevela.shared_root = "/y"
    cfg.bluevela.gpu_mode = "magic"
    errs = cfg_mod.validate_for_bluevela(cfg)
    assert any("gpu_mode" in e for e in errs)


def test_validate_clean_config_has_no_errors() -> None:
    cfg = cfg_mod.LaunchConfig()
    cfg.bluevela.login = "alice@host"
    cfg.bluevela.group = "grp_runtime"
    cfg.bluevela.workspace_root = "/home/alice/mcode-launch"
    cfg.bluevela.shared_root = "/home/alice/mcode-shared"
    assert cfg_mod.validate_for_bluevela(cfg) == []


def test_env_var_override(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "overridden.toml"
    p.write_text(
        "\n".join(
            [
                "[bluevela]",
                'login = "x@y"',
                'workspace_root = "/a"',
                'shared_root = "/b"',
                'queue_order = ["normal"]',
                'group = "g"',
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("MCODE_LAUNCH_CONFIG", str(p))
    cfg = cfg_mod.load()
    assert cfg.bluevela.login == "x@y"
