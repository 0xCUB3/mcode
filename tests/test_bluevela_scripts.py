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
