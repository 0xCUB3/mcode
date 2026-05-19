# mCode agent guide

Repo-level guidance for agents working in `mcode`. Read the docs before changing code; keep docs and behavior in sync.

## Read the relevant docs first

- `README.md`: project purpose, setup, common commands, state locations.
- `docs/architecture.md`: runner, adapters, agent loop, artifacts, DB, launch state, remote execution.
- `docs/COMMANDS.md`: CLI behavior, flags, examples, and command semantics.
- `docs/local.md`: local Ollama/vLLM workflows.
- `docs/bluevela.md`: Blue Vela launch, sync, remote bench, fetch, cancellation, LSF behavior.
- `docs/terminalbench.md`: Terminal-Bench 2.0, Harbor integration, result import, terminal agent.

If docs and code disagree, verify against code and update docs in the same patch when practical.

## Architecture boundaries

mCode is a Python benchmark harness for agentic coding and terminal tasks. It wraps a Mellea ReACT loop, exposes guarded tools, stores benchmark rows in SQLite, and tracks launch/run state for long local or remote jobs.

Keep changes close to the owner:

- CLI: `src/mcode/cli.py`, `src/mcode/bench/cli.py`
- Runner/adapters: `src/mcode/bench/runner.py`, `src/mcode/bench/adapters.py`
- Results/artifacts: `src/mcode/bench/results*.py`, `src/mcode/bench/artifacts*.py`
- Launch/remote execution: `src/mcode/launch/`, `src/mcode/bench/remote*.py`
- Tool policy and coding agent: `src/mcode/agent/tool_policy.py`, `src/mcode/agent/coding_agent.py`, `src/mcode/agent/tooling.py`
- ReACT/LLM session: `src/mcode/llm/react_driver.py`, `src/mcode/llm/session.py`
- Terminal-Bench: `src/mcode/bench/terminalbench.py`, `src/mcode/bench/terminalbench_agent.py`, `src/mcode/agent/terminal_agent.py`

The runner should stay boring: load/filter/shard/resume tasks, call adapters, save results/artifacts, and update progress. Benchmark-specific execution belongs in adapters or execution modules. Tool safety belongs in policy/tooling modules.

## Common checks

```bash
uv sync --extra dev
uv run mcode --help
uv run mcode doctor
uv run ruff check .
uv run pytest -q
```

Focused checks:

```bash
uv run pytest tests/test_cli_help.py -q
uv run pytest tests/test_cli_shards.py tests/bench -q
uv run pytest tests/test_react_driver.py tests/test_agent_generate.py tests/test_agent_tools.py tests/test_tool_policy.py -q
uv run pytest tests/launch tests/bench/test_remote.py -q
uv run pytest tests/test_terminalbench.py tests/test_terminal_agent.py tests/test_doctor.py -q
```

Do not run live Docker, Harbor, Blue Vela, or model-server tests unless asked or clearly required. If skipped, say so.

## State invariants

Keep these stores distinct:

- SQLite results DB: benchmark truth.
- Artifact dirs: patches, manifests, verifier previews, Harbor trial references, replay inputs.
- Launch state: operational state for listing, watching, canceling, server records, remote process info, and fetch metadata.

Do not treat launch state as benchmark truth. Tests must not touch real launch state. Bad `--task-ids` filters should fail early rather than create empty successful runs.

## Terminal-Bench / Harbor

Terminal-Bench 2.0 is Harbor-native. Harbor owns task download, environments, verifier injection, rewards, concurrency, and trial logs. mCode owns CLI UX, DB import, artifact manifests, and the optional mCode terminal agent.

Do not reimplement Harbor's evaluator unless explicitly asked. Preserve Harbor job/trial directories and import `result.json`, verifier logs, and rewards.

The mCode Terminal-Bench agent is stateful-terminal oriented, not patch oriented. Keep it separate from the SWE-bench patch agent.

Harbor currently requires Python 3.12+. Prefer `uv tool install harbor` or `--harbor-executable "uv run --with harbor harbor"`; do not add Harbor as a normal project extra unless the dependency conflict is intentionally resolved.

## Blue Vela

Read `docs/bluevela.md` before changing remote behavior. Blue Vela workflows must use mCode commands from the local checkout; do not manually copy code, venvs, benchmark data, DBs, artifacts, or scripts to login nodes or `/tmp`.

Use SSH only for light inspection when mCode tells you to or while diagnosing leaks. Do not hide remote failures by marking runs stopped if the remote process may still be alive.

## Git and responses

Use small, incremental commits. Before committing, inspect `git status --short`, `git diff --stat`, and relevant diffs. Stage only intended files. Do not push unless asked.

Keep responses concise: what changed, files/areas touched, checks run, commits made, and known follow-up.
