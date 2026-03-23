from pathlib import Path

BLUEVELA_DIR = Path(__file__).resolve().parents[1] / "deploy" / "bluevela"


def test_bluevela_scripts_use_dotvenv() -> None:
    for path in BLUEVELA_DIR.glob("*.sh"):
        text = path.read_text()
        assert "venv/bin/activate" not in text.replace(".venv/bin/activate", "")


def test_bluevela_setup_uses_uv_sync() -> None:
    text = (BLUEVELA_DIR / "setup.sh").read_text()
    assert "uv sync --extra swebench --extra datasets" in text
    assert "uv venv --python 3.11 venv" not in text
    assert 'uv pip install -e ".[evalplus,datasets]"' not in text
