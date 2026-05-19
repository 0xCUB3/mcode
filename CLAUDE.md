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

## Human handoff overview

Use this section when onboarding a new human maintainer. The short mental model is:

```text
benchmark task -> Mellea-backed agent generates a patch -> benchmark evaluator tests it -> mCode saves DB rows and artifacts
```

Most of the repository is scaffolding around that loop: benchmark-specific task prep, Docker/Podman isolation, guarded agent tools, verification, retry/resume/sharding, result persistence, artifact replay, and Blue Vela operations.

### Folder map

- `src/mcode/bench/`: benchmark orchestration. Important files: `bench/cli.py` turns CLI flags into `BenchConfig` and dispatches local/sharded/remote runs; `bench/runner.py` is the main lifecycle; `bench/adapters.py` maps benchmark names to task loaders/environments.
- `src/mcode/agent/`: coding-agent layer around Mellea. Important files: `agent/coding_agent.py` builds prompts/tools; `agent/tool_policy.py` enforces edit/test/final-answer rules; `agent/verification.py` builds the `run_tests` tool and failure feedback.
- `src/mcode/llm/`: Mellea integration. Important files: `llm/session.py` starts sessions, snapshots repos, runs attempts, and selects a patch; `llm/react_driver.py` is the custom ReACT loop with tool filtering, final-answer gating, malformed-call recovery, and trace collection.
- `src/mcode/execution/`: benchmark sandboxes/evaluators. Important files: `execution/swebench.py` handles SWE-bench Lite/Verified image prep, repo context, patch apply, and official eval; `execution/swebench_live.py` does the same for SWE-bench Live.
- `src/mcode/launch/`: model-server and remote-run operations. Important files: `launch/cli.py` exposes local/Blue Vela launch/status/wait/stop/sync commands; `launch/bluevela.py` submits and monitors vLLM LSF jobs; `launch/state.py` stores atomic server/run records.
- `src/mcode/ui/`: human/JSON output, dashboards, and friendly errors. Important files: `ui/task_reporter.py`, `ui/dashboard.py`, and `ui/errors.py`.
- `src/mcode/util/`: small shared helpers, mainly temp directories and retry/backoff.
- `research/`: experiment reports and result DBs. Treat each `research/*/README.md` as the index for which DBs are final vs partial/recovery.
- `experiments/`, `results*`, and `logs/`: generated local outputs. Useful for debugging history, but not core source.
- `benchmarks/`: local benchmark checkouts such as Aider Polyglot when present.

### What one task looks like

A SWE-bench Lite/Verified task is conceptually `repo`, `instance_id`, `base_commit`, `problem_statement`, optional `hints_text`, and the raw SWE-bench row containing test/eval metadata. The adapter loads this into `SWEbenchLiteTask`, the sandbox prepares a repo from the task image, the agent produces a git diff, and `SWEbenchSandbox.evaluate_patch()` runs the official eval in a clean container.

An Aider Polyglot task is simpler: `language`, `exercise`, `task_id` like `python/proverb`, and a source directory. `PreparedPolyglotTask` adds a temp work dir, editable stub paths, test paths, commands, and timeout; mCode restricts the agent to those implementation files.

### Why Docker/Podman appears everywhere

For SWE-bench, containers are for task environments and evaluation, not for hosting the model. Each task comes from an old real-world repo commit with specific dependencies, Python/system packages, and official eval scripts; containers make those reproducible and isolated.

There are two distinct phases: `repo_context()` gives the agent a copied repo mounted into an execution container for exploration and verification commands, while `evaluate_patch()` later applies only the generated patch in a clean container for official scoring.

### Vanilla Mellea vs mCode loop

Vanilla Mellea provides the model/session/tool framework and a general agent loop. mCode adds benchmark-specific scaffolding: constrained prompts, guarded tools, edit-before-final-answer behavior, mandatory verification, retry with test output, editable-path restrictions, patch/artifact capture, official evaluation, and SQLite metrics.

The key files for this distinction are `llm/session.py`, `llm/react_driver.py`, `agent/coding_agent.py`, `agent/tool_policy.py`, and for Polyglot specifically `bench/runner.py` plus `bench/aider_polyglot.py`.

### Data and result locations

- Local default DB: `experiments/results/results.db`, unless `--db` is passed.
- Important local experiment DBs: usually under `research/<date>-<experiment>/`.
- Local artifacts: `<db-stem>-artifacts/` beside the DB unless `--artifact-dir` is passed.
- Local operational state: `~/.config/mcode/launch-state.json`, or `MCODE_LAUNCH_STATE`.
- Blue Vela remote benchmark runs: `$HOME/mcode-launch/bench-runs/<run-id>/`, including `results.db`, `logs/bench-<attempt>.log`, `exit-<attempt>.code`, and the generated bench script.
- Blue Vela vLLM server logs: usually `$HOME/mcode-shared/runs/<server-run-id>/vllm.log`.
- When `--fetch-db` is enabled, the remote Blue Vela DB is copied back to the local `--db` path; `--fetch-artifacts` similarly fetches the remote artifact directory.

### Reading research reports

Research reports generally follow: what was tested, setup, exact commands, result tables, important files, and notes/gotchas. Start with the README in the experiment directory; it should identify the final merged DB and separate it from shard, partial, smoke, or recovery DBs.

Token counts are stored per task in SQLite (`task_results.prompt_tokens`, `completion_tokens`, `total_tokens`) and per artifact candidate (`artifact_candidates.*_tokens`). `uv run mcode results --db <db> --time` surfaces average token/task, but there is not a full dollar-cost accounting model because most runs were local or Blue Vela vLLM rather than paid APIs.

### Future experiment runner idea

If future work needs many experiment variants, consider adding Hydra as a separate experiment entrypoint rather than replacing Typer. Keep the boundary `Hydra config -> BenchConfig -> BenchmarkRunner`; do not make the runner depend on Hydra.

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
