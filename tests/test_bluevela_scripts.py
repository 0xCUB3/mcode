from pathlib import Path

BLUEVELA_DIR = Path(__file__).resolve().parents[1] / "deploy" / "bluevela"


def test_bluevela_scripts_use_dotvenv() -> None:
    for path in BLUEVELA_DIR.glob("*.sh"):
        text = path.read_text()
        assert "venv/bin/activate" not in text.replace(".venv/bin/activate", "")


def test_bluevela_setup_uses_mcode_deps_sync() -> None:
    text = (BLUEVELA_DIR / "setup.sh").read_text()
    assert "uv run mcode deps sync --no-dev --extra swebench --extra datasets" in text
    assert "uv venv --python 3.11 venv" not in text
    assert "uv pip install -e" not in text


def test_bluevela_bench_scripts_use_uv_run() -> None:
    text = (BLUEVELA_DIR / "run-swebench-live.sh").read_text()
    assert "uv run mcode bench swebench-live" in text


def test_bluevela_bench_scripts_wait_for_docker() -> None:
    text = (BLUEVELA_DIR / "run-swebench-live.sh").read_text()
    assert "Docker socket did not become ready" in text
    assert "client = docker.from_env()" in text
    assert "client.ping()" in text


def test_bluevela_live_launcher_sources_hf_env() -> None:
    text = (BLUEVELA_DIR / "env.sh").read_text()
    assert "BV_HF_ENV" in text
    assert "HF_DATASETS_CACHE" in text
    assert 'source "${BV_HF_ENV}"' in text


def test_bluevela_env_defaults_follow_user_environment() -> None:
    text = (BLUEVELA_DIR / "env.sh").read_text()
    assert "BV_USER=${BV_USER:-${USER:-user}}" in text
    assert "BV_LOGIN=${BV_LOGIN:-${BV_USER}@login3.bluevela.rmf.ibm.com}" in text
    assert "BV_HOME=${BV_HOME:-/u/${BV_USER}}" in text
    assert "BV_SHARED_DIR=${BV_SHARED_DIR:-/proj/dmfexp/${BV_USER}}" in text


def test_bluevela_live_launcher_drops_strategy_flag() -> None:
    text = (BLUEVELA_DIR / "run-swebench-live.sh").read_text()
    assert "--strategy" not in text
    assert "HF_DATASETS_CACHE" in text


def test_bluevela_vllm_launcher_uses_shared_hf_home() -> None:
    text = (BLUEVELA_DIR / "start-vllm.sh").read_text()
    assert 'HF_CACHE="${HF_HOME}"' in text
    assert "HUGGINGFACE_HUB_TOKEN" in text
