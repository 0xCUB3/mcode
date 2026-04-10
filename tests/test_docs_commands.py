from __future__ import annotations

import shlex
from pathlib import Path

from typer.main import get_command

from mcode.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "COMMANDS.md",
    REPO_ROOT / "deploy" / "bluevela" / "README.md",
]
PREFERRED_DOC_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "COMMANDS.md",
]
RAW_CLUSTER_PREFIXES = ("ssh ", "rsync ", "bsub ", "bjobs", "podman ", "./")


def _bash_blocks(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    blocks: list[list[str]] = []
    in_bash = False
    current: list[str] = []
    for line in lines:
        if line.startswith("```bash"):
            in_bash = True
            current = []
            continue
        if in_bash and line.startswith("```"):
            blocks.append(current)
            in_bash = False
            current = []
            continue
        if in_bash:
            current.append(line)
    return blocks


def _commands_from_block(lines: list[str]) -> list[str]:
    commands: list[str] = []
    current = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                commands.append(current.strip())
                current = ""
            continue
        current = f"{current} {line}".strip() if current else line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        commands.append(current)
        current = ""
    if current:
        commands.append(current.strip())
    return commands


def _documented_commands(path: Path) -> list[str]:
    commands: list[str] = []
    for block in _bash_blocks(path):
        commands.extend(_commands_from_block(block))
    return commands


def _assert_documented_command_exists(command_text: str) -> None:
    argv = shlex.split(command_text)
    assert argv[:3] == ["uv", "run", "mcode"]
    if len(argv) == 4 and argv[3].startswith("-"):
        return
    current = get_command(app)
    idx = 3
    consumed = 0
    while idx < len(argv) and not argv[idx].startswith("-"):
        if not hasattr(current, "get_command"):
            break
        subcommand = current.get_command(None, argv[idx])
        if subcommand is None:
            break
        current = subcommand
        idx += 1
        consumed += 1
    assert consumed >= 1, f"Documented command does not resolve in CLI: {command_text}"


def test_documented_uv_run_mcode_commands_exist() -> None:
    for path in DOC_PATHS:
        for command in _documented_commands(path):
            if command.startswith("uv run mcode "):
                _assert_documented_command_exists(command)


def test_preferred_docs_use_uv_run_mcode_for_cli_examples() -> None:
    for path in PREFERRED_DOC_PATHS:
        for command in _documented_commands(path):
            assert not command.startswith("mcode "), (
                f"Use 'uv run mcode ...' in {path.name}: {command}"
            )


def test_preferred_docs_do_not_default_to_raw_cluster_commands() -> None:
    for path in PREFERRED_DOC_PATHS:
        for command in _documented_commands(path):
            assert not command.startswith(RAW_CLUSTER_PREFIXES), (
                f"Use mcode commands instead of raw cluster commands in {path.name}: {command}"
            )
