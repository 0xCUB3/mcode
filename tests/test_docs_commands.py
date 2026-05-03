from __future__ import annotations

from pathlib import Path


def test_command_docs_cover_sampling_and_compare() -> None:
    command_docs = Path("docs/COMMANDS.md").read_text()
    assert "uv run mcode bench swebench-lite" in command_docs
    assert "uv run mcode bench suite" in command_docs
    assert "uv run mcode bench list [--json] [--benchmark NAME]" in command_docs
    assert "[--limit N]" in command_docs
    assert "uv run mcode bench artifacts-list" in command_docs
    assert "--task-id python/affine-cipher --phase evaluate --json" in command_docs
    assert "uv run mcode bench artifacts-patch" in command_docs
    assert "uv run mcode bench artifacts-replay" in command_docs
    assert "uv run mcode bench artifacts-fetch" in command_docs
    assert "uv run mcode compare --baseline-dir A --candidate-dir B [--benchmark X]" in command_docs
    assert "DB files or directories" in command_docs
    assert "selected verified candidate count" in command_docs
    assert "--shards" in command_docs
    assert "--sampling {none,multiturn}" in command_docs
    assert "--phase {run,generate,evaluate}" in command_docs
    assert "--phase generate" in command_docs
    assert "--phase evaluate" in command_docs
    assert "--artifact-dir DIR" in command_docs
    assert "uv run mcode compare" in command_docs
def test_readmes_match_command_contract() -> None:
    """The README is intentionally a high-level pointer to docs/. Detailed
    flag documentation lives in docs/local.md, docs/bluevela.md, and
    docs/COMMANDS.md, so we assert against those instead."""
    root_readme = Path("README.md").read_text()
    local_doc = Path("docs/local.md").read_text()
    bluevela_doc = Path("docs/bluevela.md").read_text()
    commands_doc = Path("docs/COMMANDS.md").read_text()
    bluevela_readme = Path("deploy/bluevela/README.md").read_text()

    # Root README points at the right places.
    assert "docs/local.md" in root_readme
    assert "docs/bluevela.md" in root_readme
    assert "docs/COMMANDS.md" in root_readme

    # Local + Blue Vela docs cover the canonical flags.
    assert "--shards" in local_doc
    assert "--sampling" in local_doc
    assert "--phase {run,generate,evaluate}" in local_doc
    assert "--artifact-dir DIR" in local_doc
    assert "uv run mcode bench suite" in local_doc
    assert "--shards" in bluevela_doc
    assert "--sampling" in bluevela_doc
    assert "--phase generate" in bluevela_doc
    assert "--artifact-dir DIR" in bluevela_doc
    assert "--fetch-artifacts" in bluevela_doc
    assert "uv run mcode bench suite" in bluevela_doc
    assert "uv run mcode bench artifacts-fetch" in bluevela_doc
    assert "artifacts-fetch --db <results.db>" in bluevela_doc

    # Reference doc covers compare and observability env vars.
    assert "uv run mcode compare" in commands_doc
    assert "uv run mcode bench suite" in commands_doc
    assert "uv run mcode bench artifacts-list" in commands_doc
    assert "uv run mcode bench artifacts-patch" in commands_doc
    assert "uv run mcode bench artifacts-replay" in commands_doc
    assert "uv run mcode bench artifacts-fetch" in commands_doc
    assert "MELLEA_TRACE_APPLICATION" in commands_doc

    # Legacy Blue Vela shell scripts still match their pinned shape.
    assert "--sampling multiturn --n-samples 3" in bluevela_readme
    assert "--shards 4" in bluevela_readme
