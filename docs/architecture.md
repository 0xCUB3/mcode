# Architecture

The reason I wrote this doc is to share the current basic structure of the code and the few invariants I try not to break.

TL;DR: the CLI builds a `BenchConfig`, the runner asks a benchmark adapter for tasks, the agent generates a patch with a small tool set, the adapter verifies it, and `ResultsDB` records what happened. Around that path are launch state, shard management, artifact storage, and Blue Vela remote execution.

## Main code paths

`src/mcode/cli.py` owns the top-level Typer app. It registers `bench`, `launch`, `doctor`, `results`, `compare`, `deps`, and a few compatibility aliases. Most benchmark work starts in `src/mcode/bench/cli.py`, which is still intentionally the public bench facade. It parses the command line, prints the run plan, starts or resumes run state, handles local and remote execution, and then delegates real work to the runner.

The runner lives in `src/mcode/bench/runner.py`. It is the shared path for SWE-bench, Aider Polyglot, smoke, and suite runs. It should stay boring: choose tasks, skip completed rows on resume, call the adapter, write result rows, save artifacts, and update progress. When that file starts collecting glue that belongs somewhere else, split the helper out but keep the facade stable.

Adapters are wired in `src/mcode/bench/adapters.py`. The point of an adapter is to hide benchmark-specific details from the runner. SWE-bench knows about repos, Docker images, official evaluation, and task ids. Aider Polyglot knows about exercise directories, language toolchains, and retry feedback. The runner should not know those details beyond the common adapter methods.

The agent loop is under `src/mcode/agent/` and `src/mcode/llm/`. `tool_policy.py` is where command and tool-call policy belongs. `react_driver.py` handles the Mellea interaction and calls the policy before tools run. Keep safety checks centralized there instead of scattering path and shell checks across benchmark code.


## Module map

|Area|Files|Notes|
|-|-|-|
|Top-level CLI|`src/mcode/cli.py`|Registers the Typer app and top-level commands|
|Bench CLI|`src/mcode/bench/cli.py`|Public bench facade, run planning, local and remote dispatch|
|Runner|`src/mcode/bench/runner.py`|Shared task loop across benchmarks|
|Run-state updates|`src/mcode/bench/runner_state.py`|Best-effort metadata and progress patches into launch state|
|Artifacts|`src/mcode/bench/runner_artifacts.py`, `src/mcode/bench/artifacts/`|Manifest writing, patch storage, replay, fetch, and inspection|
|Adapters|`src/mcode/bench/adapters.py`, `aider_polyglot.py`, `src/mcode/bench/swebench/`|Benchmark-specific task loading and Docker-based SWE-bench verification|
|Terminal-Bench|`src/mcode/bench/terminalbench.py`, `terminalbench_agent.py`, `src/mcode/agent/terminal_agent.py`|Harbor-backed Terminal-Bench 2.0 execution, result import, and mCode terminal agent|
|Sharding|`src/mcode/bench/shards.py`|Worker process launch, shard DB paths, merge, retryable infra exit code|
|Remote bench|`src/mcode/bench/remote.py`, `src/mcode/bench/remote_script.py`|Blue Vela bench plan, shell script generation, fetch planning|
|Summaries|`src/mcode/bench/summary.py`|Run plan, failure hints, footer, rerun metadata|
|Suites|`src/mcode/bench/suite_cli.py`|Mixed-suite command and suite manifest handling|
|Results|`src/mcode/bench/results/`|SQLite facade, schema, ingest, metrics, export, merge|
|Launch|`src/mcode/launch/*.py`|Config, state file, Blue Vela launch, local launchers, status, stop, sync|
|Agent policy|`src/mcode/agent/tool_policy.py`|Tool-call checks and denial reasons|
|Mellea driver|`src/mcode/llm/react_driver.py`|ReACT loop integration and tool-call recovery|

The split is not about hiding complexity. It is about keeping each failure mode close to the code that can explain it. Remote script bugs belong in `remote_script.py`; SQLite shape belongs in the results modules; tool denial behavior belongs in `tool_policy.py` and its tests.

## Results, artifacts, and launch state

This distinction matters because a lot of bugs come from treating them as one blob.

The SQLite results DB is the durable record of benchmark runs and task results. `src/mcode/bench/results/__init__.py` is the public facade, with schema, ingest, metrics, exports, artifact-copy helpers, and merge logic split into smaller modules in the same package. If you change the schema, update the schema module, exports, tests, and docs in the same patch.

Artifacts are the saved per-task generation records: manifests, candidate patches, selected candidate metadata, and evaluation inputs. They live in an artifact directory, usually next to the DB as `<db-stem>-artifacts`. The runner writes them through `runner_artifacts.py`, and the artifact CLI in `bench/artifacts/cli.py` lets you list, show, patch, replay, or fetch them later.

Launch state is operational state, not benchmark truth. It lives in `~/.config/mcode/launch-state.json` unless `MCODE_LAUNCH_STATE` points somewhere else. It records servers and bench runs so commands like `launch status`, `bench list`, `bench show`, `bench cancel`, `bench prune`, and `watch` have something to read. Tests must never touch the real state file; `tests/conftest.py` isolates it for every test.

A good rule of thumb: if a number is part of the benchmark result, it belongs in SQLite. If it helps you find, cancel, resume, or fetch a run, it belongs in launch state. If it is a patch or manifest generated by the model, it belongs in artifacts.


## SQLite schema overview

The schema is append-only because old research DBs stay useful. Schema setup creates missing tables and adds newer columns without destructive migrations.

|Table|Purpose|
|-|-|
|`runs`|One benchmark run configuration and timestamp|
|`task_results`|One evaluated task row per run and task id|
|`diagnostic_events`|Optional compact per-turn or per-tool trace events|
|`artifact_tasks`|One artifact-backed task manifest per run and task id|
|`artifact_candidates`|Candidate patch metadata, selected flag, token counts, and failure counters|
|`artifact_verification_evidence`|Verification commands and output previews attached to a candidate|
|`artifact_evaluations`|Evaluation rows produced while replaying or evaluating artifacts|

`ResultsDB` in `bench/results/__init__.py` is the public API. The smaller modules in that package do the schema setup, artifact copying, metrics, ingest, export, and merge work. Callers should go through the facade unless there is a good reason not to.

## Artifact layout

The default artifact root is `<db-stem>-artifacts`. Under that, each task gets a safe path based on benchmark and task id:

```text
<artifact-root>/
  swebench-lite/
    astropy__astropy-12907/
      manifest.json
      candidate-0/
        patch.diff
        submission.json
        trace.json
        verification-0.txt
```

The manifest records schema version, phase, run config digest, code sha, model, backend, task reference, candidates, evaluations, and metadata. Candidate directories hold the patch and optional trace, submission, and verification previews. The DB stores enough paths and digests to find these files later and to export artifact summaries without reading every patch.

## Launch-state records

The state file is JSON with two arrays: `servers` and `runs`. Writes are protected by a sibling lock file and saved atomically with a temp file plus rename.

Server records store id, target, endpoint, model, config hash, job id, log path, status, and metadata. Run records store id, target, benchmark, status, server id, log paths, DB path, shard pids, remote metadata, progress, and timestamps. Blue Vela run records also carry remote process and fetch information.

Bench run ids are generated so a human can recognize them and the state file can distinguish overlapping work. The compact ids shown by `bench list` are display shortcuts only; the full id stays in state.

## Benchmark lifecycle

The bench command prints a run plan before doing expensive work. That plan should answer the questions a tired human has before hitting enter on a long job: which benchmark, which model, where the DB goes, how many shards, which phase, and whether remote fetching is on.

For local non-sharded runs, the CLI calls the runner in the current process. For local sharded runs, it starts worker processes, each with a shard DB, then merges the shard DBs into the requested output DB. Shard DBs stay under `<db-stem>-shards/`, which makes retries and manual recovery possible.

For Blue Vela runs, the local CLI builds a remote bench plan, syncs enough metadata, starts the remote command under `setsid`, streams logs, and records remote process information in launch state. When the remote process exits, it fetches the DB by default and artifacts if requested.

Each task follows the same rough sequence: prepare the task, let the agent generate a patch, save the artifact manifest, run verification if the phase calls for it, then write the task result. SWE-bench has extra progress events for repo prep and official evaluation because those stages can take long enough that silence feels like a hang.

## Resume behavior

Rerunning a benchmark against the same DB should not throw away good work. Completed task rows are skipped. Retryable infrastructure failures can be retried. Sharded runs reuse their stable shard DBs and merge completed rows back into the main DB.

This is why DB paths matter. If you change `--db`, you are asking for a new run. If you keep `--db`, you are usually asking to continue or re-read the existing run. The run state also stores enough rerun metadata for `bench show` to print a useful command instead of making you reconstruct it from memory.

Zero-task filters should fail early. A bad `--task-ids` value should not create a successful empty run.

## Phase runs and artifact replay

The default phase is `run`, which generates and evaluates in one pass. `generate` saves candidates and manifests without official evaluation. `evaluate` reads those artifacts and writes result rows. `prepare` exists for commands that need to do benchmark prep without a normal solve pass.

This phase split is useful when generation is expensive, evaluation is flaky, or evaluation needs to happen on a different machine. It is also useful for comparing generated patches before paying for official eval. Artifact replay should keep working across machines as long as the benchmark root and artifact paths are supplied when the adapter needs them.


## Resume and merge invariants

Resume is task-id based. A completed task row should not be recomputed just because the process restarted. Retryable infrastructure rows can be retried, but normal failed task rows are benchmark results and should remain visible.

Shard DB paths are stable for the parent DB and shard index. The merge code writes task rows into the requested output DB and reports how many shards were used or ignored. Manual recovery should use `mcode bench merge-shards`; normal `--shards N` runs should merge automatically.

Commands that inspect runs should not guess by `MAX(id)` from SQLite. `bench show` reads launch state and resolves full or compact run ids there. The DB is the result store, not the run registry.

## Blue Vela code boundaries

The launch side of Blue Vela lives under `src/mcode/launch/`. It knows about config files, profiles, LSF jobs, SSH, rsync, server health, logs, and stopping jobs. It should not know benchmark internals.

The remote benchmark side lives under `src/mcode/bench/remote.py` and `src/mcode/bench/remote_script.py`. `remote.py` is the public facade used by the CLI. `remote_script.py` builds the shell plan and LSF submit command. Keep script construction there so remote behavior can be tested without starting a cluster job.

`src/mcode/bench/cancel.py` bridges run state and process cleanup. Local sharded runs get SIGTERM and then SIGKILL for worker pids. Blue Vela runs get process-group termination over SSH, followed by verification. If verification fails, the run stays `running` so the user can retry. That is intentional.


## Remote bench plans

Blue Vela benchmark execution is split so the shell script can be tested without submitting a job. `RemoteBenchPlan` describes the remote command, environment, paths, fetch behavior, and identity. `RemoteArtifactFetch` describes a later artifact fetch. `build_remote_bench_plan` and `build_lsf_submit_command` live in `remote_script.py` so quoting, environment forwarding, Podman setup, and LSF submission do not sprawl through the CLI.

The remote script sets up a per-run workspace tmp dir, Podman socket, graphroot and runroot, and lock dir. It runs the bench under `setsid`, which gives cancellation a process group to terminate. The local side records the remote pid and paths before streaming output.

## Command structure

The user-facing command tree should make the common path obvious:

```text
mcode doctor
mcode launch ...
mcode bench smoke|swebench-lite|aider-polyglot|suite
mcode bench list|show|cancel|prune
mcode bench artifacts list|show|patch|replay|fetch
mcode results
mcode compare
```

Compatibility aliases can stay hidden when they prevent old scripts from breaking, but new docs should use the grouped commands. For example, `mcode bench artifacts list` is the documented form. The older `mcode bench artifacts-list` command still works as a hidden alias.

Top-level manual recovery commands should not crowd the default help. `mcode bench merge-shards` is the documented shard merge command. The older top-level `mcode merge-shards` remains hidden for compatibility.


## Tool policy

`tool_policy.py` decides whether a tool call is allowed. It covers paths the agent may edit, dangerous shell prefixes, evasive verification commands, final answers before verification, and diagnostic denial reasons. `react_driver.py` calls the policy before tool execution and records counters for malformed calls, invalid calls, blocked finalizers, repeated failed test commands, and post-edit exploration.

When adding a new policy rule, add a direct unit test in `tests/test_tool_policy.py` and at least one driver test when the rule changes loop behavior. The policy should stay auditable. A denial reason that only exists in a prompt is not enough.

## Test coverage

For a small CLI or docs-adjacent change, run the focused tests first:

```bash
uv run pytest tests/test_cli_help.py tests/test_bench_artifacts_cli.py tests/bench/test_remote.py -q
```

For runner, results, or launch-state changes, run the full suite before committing:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

The full suite is fast enough now that skipping it usually costs more time than it saves.


More focused checks by area:

```bash
# CLI shape and aliases
uv run pytest tests/test_cli_help.py -q

# Artifact commands and replay plumbing
uv run pytest tests/test_bench_artifacts_cli.py -q

# Blue Vela remote plan and fetch metadata
uv run pytest tests/bench/test_remote.py -q

# Launch config, state, sync, and entrypoints
uv run pytest tests/launch -q

# Results merge/export and shard behavior
uv run pytest tests/test_cli_shards.py tests/bench -q
```

If a change touches the agent loop, run the ReACT driver and agent generation tests too:

```bash
uv run pytest tests/test_react_driver.py tests/test_agent_generate.py -q
```

## Maintenance rules

Do not let tests read or write the real launch state. Do not add a benchmark feature without deciding where its DB rows, artifacts, and state metadata belong. Do not hide a remote failure by marking the run stopped if the process may still be alive. Do not make human output quiet during long work unless JSON mode or `MCODE_LIVE_TRACE=0` asked for that. And when command behavior changes, update these docs in the same patch. Future you will be grateful.
