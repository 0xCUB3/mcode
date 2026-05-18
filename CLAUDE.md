# mCode agent guide

This file is repo-level guidance for agents working in `mcode`. Read it before changing code, then follow the relevant docs linked below. The docs are the source of truth for this repo. If a doc describes a command, architecture boundary, workflow, test, or operational caveat, use that guidance instead of guessing.

## Documentation first

Before nontrivial work, read the docs for the area you are touching:

- Start with `README.md` for the project purpose, quick starts, common commands, state locations, and usage notes.
- Read `docs/architecture.md` before changing the runner, adapters, agent loop, artifacts, results DB, launch state, remote execution, or command layout.
- Read `docs/COMMANDS.md` before changing CLI behavior, flags, output, docs examples, or command semantics.
- Read `docs/local.md` before changing local Ollama/vLLM workflows.
- Read `docs/bluevela.md` before changing Blue Vela launch, sync, remote bench, fetch, cancellation, or LSF behavior.
- Read `docs/terminalbench.md` before changing Terminal-Bench 2.0, Harbor command construction, Harbor result import, or the mCode Terminal-Bench agent.

If docs and code disagree, verify with the code and update the docs in the same patch when practical. If you add a new workflow, flag, artifact shape, DB behavior, remote path, or operational pitfall, document it before calling the change done.

## What this repo is

mCode is a Python benchmark harness for agentic coding and terminal tasks. It wraps a Mellea ReACT loop, exposes guarded tools, records benchmark rows in SQLite, and keeps enough launch/run state to resume, inspect, cancel, fetch, and compare long runs.

Supported benchmark areas include:

- SWE-bench Lite and Verified
- SWE-bench Live
- Aider Polyglot
- Terminal-Bench 2.0 through Harbor
- mixed suites and smoke runs

Model backends include Ollama, local vLLM, OpenAI-compatible endpoints, and vLLM jobs on IBM Blue Vela.

## Key files and boundaries

Keep changes close to the code that owns the behavior:

- Top-level Typer app: `src/mcode/cli.py`
- Benchmark CLI facade: `src/mcode/bench/cli.py`
- Shared runner lifecycle: `src/mcode/bench/runner.py`
- Benchmark adapter registry: `src/mcode/bench/adapters.py`
- Suite and smoke commands: `src/mcode/bench/suite_cli.py`
- Results DB facade/schema/export/merge: `src/mcode/bench/results*.py`
- Artifact manifests and artifact CLI: `src/mcode/bench/artifacts.py`, `src/mcode/bench/runner_artifacts.py`, `src/mcode/bench/artifacts_cli.py`
- Launch state and server launchers: `src/mcode/launch/`
- Blue Vela remote bench planning: `src/mcode/bench/remote.py`, `src/mcode/bench/remote_script.py`
- Tool policy: `src/mcode/agent/tool_policy.py`
- Coding agent tools/prompt: `src/mcode/agent/coding_agent.py`, `src/mcode/agent/tooling.py`
- ReACT driver and LLM session: `src/mcode/llm/react_driver.py`, `src/mcode/llm/session.py`
- Terminal-Bench integration: `src/mcode/bench/terminalbench.py`, `src/mcode/bench/terminalbench_agent.py`, `src/mcode/agent/terminal_agent.py`

The runner should stay boring: load tasks, shard/filter/resume, call the adapter, save results/artifacts, and update progress. Benchmark-specific container or evaluation behavior belongs in adapters or execution modules. Tool safety belongs in policy/tooling modules, not scattered across benchmark code.

## Commands agents should know

Install and basic checks:

```bash
uv sync --extra dev
uv run mcode --help
uv run mcode doctor
```

Focused local smoke:

```bash
uv run mcode doctor local-ollama
uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <server-id> --timeout 120
uv run mcode bench smoke --backend openai --model granite4 --shards 4
uv run mcode bench show --latest
```

Blue Vela rhythm:

```bash
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
uv run mcode launch sync bluevela
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json
uv run mcode launch wait <server-id> --timeout 1800
uv run mcode bench smoke --backend openai --model Qwen/Qwen3.6-35B-A3B --on bluevela --shards 4
uv run mcode bench show --latest
uv run mcode launch stop <server-id>
```

Terminal-Bench local smoke:

```bash
uv tool install harbor
uv run mcode doctor terminal-bench
uv run mcode bench terminal-bench --agent oracle --model unused --limit 1
```

Use `--harbor-executable "uv run --with harbor harbor"` when Harbor needs to run from a project-aware environment, especially for the custom mCode Harbor agent.

## Testing expectations

Run focused tests first, then broader checks when the touched area warrants it.

General final check for code changes:

```bash
uv run ruff check .
uv run pytest -q
```

Focused checks by area:

```bash
# CLI shape and command registration
uv run pytest tests/test_cli_help.py -q

# Runner/results/sharding behavior
uv run pytest tests/test_cli_shards.py tests/bench -q

# Artifacts and replay plumbing
uv run pytest tests/test_bench_artifacts_cli.py -q

# Agent loop and tool behavior
uv run pytest tests/test_react_driver.py tests/test_agent_generate.py tests/test_agent_tools.py tests/test_tool_policy.py -q

# Launch and Blue Vela remote planning
uv run pytest tests/launch tests/bench/test_remote.py -q

# Terminal-Bench integration
uv run pytest tests/test_terminalbench.py tests/test_terminal_agent.py tests/test_doctor.py -q
```

Do not run live Docker, Harbor, Blue Vela, or model-server tests unless the user asks or the task clearly requires it. If skipped, say so.

## Results, artifacts, and state invariants

Keep these stores distinct:

- SQLite results DB is benchmark truth: runs, task rows, diagnostics, artifact metadata.
- Artifact directories hold generated patches, manifests, verifier previews, Harbor trial references, and replay inputs.
- Launch state is operational state for listing, watching, cancellation, server records, run records, and fetch metadata.

Do not treat launch state as benchmark truth. Do not make tests touch the real launch state. Use the isolated test fixtures and existing helpers.

Reruns against the same DB should resume completed work. Bad `--task-ids` filters should fail early rather than creating an empty successful run. Retryable infrastructure failures can be retried, but ordinary failed task rows are benchmark results.

## Terminal-Bench and Harbor notes

Terminal-Bench 2.0 is Harbor-native. Harbor owns task download, container setup, verifier injection, reward parsing, concurrency, and trial logs. mCode owns CLI UX, DB import, artifact manifests, and the optional mCode terminal agent.

Do not reimplement Harbor's evaluator unless explicitly asked. Preserve Harbor job/trial directories and import their `result.json`, verifier logs, and rewards into mCode artifacts/results.

The custom mCode Terminal-Bench agent is stateful-terminal oriented, not patch oriented. It may create files and change container state. Keep it separate from the SWE-bench coding agent, which is git-diff/patch based and intentionally constrained.

Harbor currently requires Python 3.12+. Avoid adding Harbor as a normal project extra unless the `datasets` dependency conflict is intentionally resolved. Prefer `uv tool install harbor` or `--harbor-executable "uv run --with harbor harbor"`.

## Blue Vela notes

Read `docs/bluevela.md` before changing remote behavior. Remote execution uses rsync, LSF, per-run workspaces, process groups, Podman socket setup, DB fetch, and optional artifact fetch. Keep quoting and script construction in `remote_script.py` where it can be tested without submitting jobs.

Do not hide remote failures by marking a run stopped if the remote process may still be alive. Cancellation and fetch behavior must stay visible and recoverable.

## Git and commits

Use small, incremental commits for multi-step work. Before committing, inspect status and diff:

```bash
git status --short
git diff --stat
git diff
```

Stage only intended files. Commit messages should be short, imperative, and specific. Do not use conventional prefixes unless the repo starts doing so. Do not push unless the user asks.

## Writing and responses

Keep user-facing responses concise and concrete. Mention:

- what changed
- files or area touched
- tests/checks run
- commits made, if any
- known issues or follow-up

Do not paste large file contents into responses. Do not narrate every tool call. If a bug is unclear, gather evidence with logs/tests/instrumentation instead of speculating.
