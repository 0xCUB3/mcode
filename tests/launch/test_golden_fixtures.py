"""Golden fixtures for every supported model.

Each target model has a committed JSON snapshot of its rendered `env.json`
under tests/launch/fixtures/env_json/. Changes to profiles.py that shift the
wire-level contract must regenerate the fixtures — `MCODE_UPDATE_FIXTURES=1
pytest tests/launch/test_golden_fixtures.py` rewrites them. Without that env
var, any mismatch is a hard failure.

Purpose (per plan M8): snapshot the exact bytes sent to the cluster, so a
silent regression like the "Gemma --chat-template dropped" bug cannot ship.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mcode.launch import bluevela, profiles
from mcode.launch.config import BluevelaConfig
from mcode.launch.models import LaunchSpec, Target

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "env_json"

# (profile_name, model_id). All 5 Blue Vela target models + the local smoke.
MODELS: list[tuple[str, str]] = [
    ("qwen3.5-27b", "Qwen/Qwen3.5-27B"),
    ("qwen3.5-a3b", "Qwen/Qwen3.5-35B-A3B"),
    ("gemma4-31b", "google/gemma-4-31B-it"),
    ("granite4", "ibm-granite/granite-4.0-h-small"),
    ("minimax-m2", "MiniMaxAI/MiniMax-M2.5"),
    ("local-tiny", "Qwen/Qwen2.5-0.5B"),
]


def _fixed_cfg() -> BluevelaConfig:
    """Canonical Blue Vela config for fixture rendering — no user-specific
    values (portability grep-gate lines up). If this function ever learns a
    real username, the test suite is broken."""
    c = BluevelaConfig()
    c.login = "testuser@testhost"
    c.workspace_root = "/u/testuser/mcode-launch"
    c.shared_root = "/u/testuser/mcode-shared"
    c.queue_order = ["normal"]
    c.group = "grp_runtime"
    c.gpu_mode = "exclusive_process"
    c.hf_env = "/u/testuser/.config/mcode/hf-env.sh"
    return c


def _render(model: str) -> dict:
    spec = LaunchSpec(
        target=Target.BLUEVELA,
        model=model,
        profile=profiles.resolve(model),
    )
    return bluevela.build_env_json(
        spec,
        _fixed_cfg(),
        run_dir="/u/testuser/mcode-shared/runs/bv-FIXED",
    )


@pytest.mark.parametrize("profile_name,model", MODELS)
def test_env_json_matches_golden(profile_name: str, model: str) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture = FIXTURES_DIR / f"{profile_name}.json"
    rendered = _render(model)
    rendered_text = json.dumps(rendered, indent=2, sort_keys=True) + "\n"

    if os.environ.get("MCODE_UPDATE_FIXTURES"):
        fixture.write_text(rendered_text)
        return

    if not fixture.exists():
        fixture.write_text(rendered_text)
        pytest.fail(f"missing fixture {fixture} — wrote a baseline; review and re-run tests")

    golden = fixture.read_text()
    assert rendered_text == golden, (
        f"env.json for {profile_name} drifted — if intentional, "
        f"MCODE_UPDATE_FIXTURES=1 pytest tests/launch/test_golden_fixtures.py "
        f"to refresh {fixture}"
    )


def test_gemma_fixture_has_chat_template_flag() -> None:
    """Bug-specific regression: Gemma4's env.json MUST carry
    `--chat-template /chat-template.jinja` in VLLM_FLAGS, otherwise tool
    calls silently fail (vLLM issue #39043)."""
    rendered = _render("google/gemma-4-31B-it")
    flags = rendered["VLLM_FLAGS"]
    assert "--chat-template" in flags
    i = flags.index("--chat-template")
    assert flags[i + 1] == "/chat-template.jinja"
    assert "CHAT_TEMPLATE_PATH" in rendered
    assert rendered["CHAT_TEMPLATE_PATH"].endswith("tool_chat_template_gemma4.jinja")


def test_minimax_fixture_has_safetensors_fast_gpu_env() -> None:
    """MiniMax M2 is known to produce illegal memory errors without the
    SAFETENSORS_FAST_GPU=1 env. Lock it in."""
    rendered = _render("MiniMaxAI/MiniMax-M2.5")
    assert rendered["EXTRA_ENV"].get("SAFETENSORS_FAST_GPU") == "1"
    assert "--enable_expert_parallel" in rendered["VLLM_FLAGS"]


def test_granite_uses_hermes_not_granite_parser() -> None:
    """Granite 4.x uses the `hermes` tool-call parser. `granite` is the 3.x
    parser; vLLM will reject it for 4.x weights."""
    rendered = _render("ibm-granite/granite-4.0-h-small")
    flags = rendered["VLLM_FLAGS"]
    i = flags.index("--tool-call-parser")
    assert flags[i + 1] == "hermes"


def test_qwen3_a3b_fixture_has_expected_tp() -> None:
    rendered = _render("Qwen/Qwen3.5-35B-A3B")
    assert rendered["GPU_COUNT"] == "2"


def test_no_fixture_carries_developer_user_info() -> None:
    """Portability grep-gate for the committed fixtures: no real usernames
    or project paths. The canonical fake user is `testuser` / `testhost`.
    """
    import re

    forbidden = re.compile(r"skula|login\d+\.bluevela|grp_(?!runtime\b)[a-z_]+")
    for path in FIXTURES_DIR.glob("*.json"):
        text = path.read_text()
        m = forbidden.search(text)
        assert not m, f"{path}: forbidden token {m.group(0) if m else ''}"
