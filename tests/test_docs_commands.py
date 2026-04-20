from __future__ import annotations

from pathlib import Path


def test_command_docs_cover_sampling_and_compare() -> None:
    command_docs = Path("docs/COMMANDS.md").read_text()
    assert "uv run mcode bench swebench-lite" in command_docs
    assert "--shards" in command_docs
    assert "--sampling {none,rejection,repair,sofai}" in command_docs
    assert "uv run mcode compare" in command_docs


def test_readmes_match_command_contract() -> None:
    root_readme = Path("README.md").read_text()
    bluevela_readme = Path("deploy/bluevela/README.md").read_text()

    assert "--shards" in root_readme
    assert "--sampling" in root_readme
    assert "uv run mcode compare" in root_readme
    assert "MELLEA_TRACE_APPLICATION" in root_readme
    assert "--sampling rejection --n-samples 3" in bluevela_readme
    assert "--shards 4" in bluevela_readme
