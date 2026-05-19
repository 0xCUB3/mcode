# mCode commands

This is the command reference I wish I had while building the harness. It is still worth running `mcode <command> --help` when you need the exact Typer output, but this page explains how the commands fit together and which flags matter in real runs.

Examples use `uv run mcode`. If you installed the console script directly, use `mcode` instead.

## Global behavior

These work with any command:

```bash
uv run mcode --help
uv run mcode --version
uv run mcode --verbose <command>
NO_COLOR=1 uv run mcode <command>
MCODE_NO_COLOR=1 uv run mcode <command>
MCODE_DEBUG=1 uv run mcode <command>
MCODE_LAUNCH_STATE=/tmp/mcode-state.json uv run mcode <command>
```

`--verbose` raises mCode and Mellea logging to INFO. `MCODE_DEBUG=1` turns off the formatted error page and lets raw tracebacks through, which is useful when you are fixing the harness rather than using it. `NO_COLOR` and `MCODE_NO_COLOR` are there for CI logs and terminals that do not want ANSI color.

Exit codes are intentionally plain:

|Code|Meaning|
|-|-|
|0|The command succeeded|
|1|A user-actionable failure, usually printed with what happened and what to try next|
|2|Bad command usage, or a bench cancel that cannot work from another shell|
|3|`mcode launch wait` could not find that id|
|86|A sharded benchmark hit retryable infrastructure trouble|
|130|The process was interrupted with Ctrl+C|

## Doctor

The doctor is the first command to run on a new machine. It checks the local basics and, when you pass a target, asks that target to check what it needs.

```bash
uv run mcode doctor
uv run mcode doctor local-ollama
uv run mcode doctor local-vllm
uv run mcode doctor bluevela
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
uv run mcode doctor bluevela --deep
uv run mcode doctor terminal-bench
uv run mcode doctor terminal-bench --deep
```

With no target, the doctor checks the results directory, a container runtime, Mellea importability, and Ruff. `local-ollama` checks the local Ollama path. `local-vllm` checks the local vLLM path. `bluevela` checks SSH, queues, group membership, and config. `terminal-bench` checks Harbor, Docker, and, with `--deep`, one oracle task. `--init` is Blue Vela only and writes `~/.config/mcode/launch.toml` after probing the cluster.

When a row is red, read the `next:` line. It is usually more useful than the exception text.

## Launch commands

Launch commands manage model servers and the remote workspace. They do not run benchmarks by themselves, but benchmark commands use their state to find healthy OpenAI-compatible endpoints.

```bash
uv run mcode launch local-ollama --model granite4
uv run mcode launch local-vllm --model Qwen/Qwen2.5-0.5B
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
uv run mcode launch status
uv run mcode launch wait <server-id> --timeout 600
uv run mcode launch logs <server-id>
uv run mcode launch stop <server-id>
uv run mcode launch stop --all
uv run mcode launch refresh
uv run mcode launch sync bluevela
```

`local-ollama` records an existing Ollama model server and exposes its OpenAI-compatible endpoint to the rest of mCode. Use the exact model name from `ollama list`.

`local-vllm` starts a local vLLM server and records it. It accepts the same `--model` shape you would pass to vLLM.

`bluevela` submits a vLLM server job to Blue Vela. The required flag is `--model`. You can also pass `--tensor-parallel N`, `--max-model-len N`, and `--json`. The built-in model profiles live in `src/mcode/launch/profiles.py`.

`launch wait` exits 0 when a server is healthy, 1 if it failed or stopped, 2 on timeout, and 3 if the id is unknown. I use it in scripts because it lets the next line assume the endpoint is ready.

`launch stop --all` only stops servers recorded in your state file. It does not run a broad kill for your whole user account.

`launch sync bluevela` rsyncs the local repo to the configured Blue Vela workspace. Useful flags are `--dry-run`, `--src DIR`, and `--bootstrap`. The bootstrap flag claims a populated remote directory and should be used deliberately.


## Watch and operational status

`watch` is a read-only dashboard. It combines server records from `launch status` and run records from `bench list`, then refreshes every two seconds until Ctrl+C.

```bash
uv run mcode watch
```

For scripts, prefer the JSON forms of the underlying commands:

```bash
uv run mcode launch status --json
uv run mcode launch status --json --raw
uv run mcode bench list --json
```

`launch logs <id>` prints the log path for a recorded server or run. For local launcher-owned logs it can tail the file. On Blue Vela, the path points at the remote log recorded for that server or bench run.

## Benchmark commands

The bench command tree now has two kinds of commands: commands that run benchmarks and commands that manage run records.

```bash
uv run mcode bench smoke --model M [flags]
uv run mcode bench swebench-lite --model M [flags]
uv run mcode bench swebench-live --model M [flags]
uv run mcode bench aider-polyglot --model M [flags]
uv run mcode bench terminal-bench --model M [flags]
uv run mcode bench suite --model M [flags]

uv run mcode bench list
uv run mcode bench show --latest
uv run mcode bench show <run-id>
uv run mcode bench cancel <run-id>
uv run mcode bench prune
uv run mcode bench merge-shards --out merged.db shard-a.db shard-b.db
uv run mcode bench artifacts list --db results.db
```

Every real benchmark prints a run plan, writes SQLite rows, updates launch state, and prints a footer at the end. The human view includes compact live progress by default. Add `--json` for line-delimited JSON events. Set `MCODE_LIVE_TRACE=0` to mute the human live trace, or `MCODE_LIVE_TRACE=1` to include live trace events in JSON mode.

### Common benchmark flags

These flags are shared by most benchmark commands:

|Flag|Use|
|-|-|
|`--model M`|Model name used by Mellea and endpoint discovery|
|`--backend B`|Mellea backend. Local SWE-bench defaults to `ollama`; smoke, suite, and Aider Polyglot usually use `openai`|
|`--loop-budget N`|Agent turn budget per task|
|`--temperature F`|Sampling temperature|
|`--seed N`|Random seed|
|`--timeout N`|Evaluation timeout per task|
|`--limit N`|Run the first N selected tasks|
|`--task-ids X`|Comma-separated task ids, or a JSON/text file with task ids|
|`--db PATH`|SQLite DB path. Pick this explicitly for runs you care about|
|`--phase`|Use `run`, `generate`, `evaluate`, or `prepare`|
|`--artifact-dir DIR`|Directory for generated task artifacts. Defaults next to the DB|
|`--shards N`|Start N workers and merge their DBs when they finish|
|`--shard-count C --shard-index I`|Manual shard mode. The automatic `--shards` path uses this internally|
|`--on`|Use `local` or `bluevela`|
|`--fetch-db / --no-fetch-db`|For Blue Vela, copy the DB back when the remote run ends|
|`--fetch-artifacts / --no-fetch-artifacts`|For Blue Vela, copy the artifact directory back too|
|`--diagnostic-traces / --no-diagnostic-traces`|Persist compact diagnostic trace events|
|`--json`|Emit JSON objects instead of the human display|

Rerunning the same command against the same DB resumes work. Finished task rows are skipped. Retryable infrastructure failures can be retried. Sharded runs reuse stable shard DBs under `<db-stem>-shards/` and merge whatever finished.

A bad `--task-ids` filter fails before work starts. If you ask for `python/word-count` and the selected benchmark has no such task, mCode tells you rather than creating an empty successful run.

## Smoke benchmark

`bench smoke` is a 16-task SWE-bench Verified diagnostic slice. It uses the SWE-bench runner underneath, so the same phase, artifact, shard, and remote flags apply.

```bash
uv run mcode bench smoke \
  --backend openai \
  --model granite4 \
  --shards 4 \
  --db experiments/results/smoke.db
```

I use this before any larger run. It is short enough to fail fast when Docker, endpoint discovery, or result writing is broken.

## SWE-bench Lite, Verified, and Live

`bench swebench-lite` is the main SWE-bench command. Despite the name, it can run other SWE-bench datasets with `--dataset`, including Verified.

```bash
uv run mcode bench swebench-lite \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 20 \
  --sampling multiturn \
  --sampling-budget 2 \
  --selection-attempts 3 \
  --timeout 300 \
  --mem-limit 8g \
  --pids-limit 512 \
  --shards 4 \
  --db research/swebench-verified/results.db
```

SWE-bench-specific flags:

|Flag|Use|
|-|-|
|`--split`|Dataset split, usually `test`|
|`--dataset`|Hugging Face dataset name. Defaults to `SWE-bench/SWE-bench_Lite`|
|`--namespace`|Prebuilt image namespace. The default is `swebench`; set `""` to build locally|
|`--arch`|Image architecture: `auto`, `x86_64`, or `arm64`|
|`--max-workers N`|Parallelism for local image building|
|`--force-rebuild`|Rebuild images even when cached|
|`--mem-limit TEXT`|Eval container memory limit|
|`--pids-limit N`|Eval container process limit|
|`--cpu-limit N`|Cap each eval container at N CPU cores|
|`--check-image-digests / --no-check-image-digests`|Check registry digests before reusing cached images|
|`--n-samples N`|Outer attempts, or the fallback sampling budget|
|`--sampling`|Use `none` or `multiturn`|
|`--sampling-budget N`|Override the sampling loop budget|
|`--selection-attempts N`|Run independent full-budget trajectories and select one before official eval|
|`--eval-repair-attempts N`|Retry failed official evaluations with deterministic eval feedback|
|`--chunk-size N`|On Blue Vela, run sequential chunks and merge their DBs|
|`--relaunch-vllm / --no-relaunch-vllm`|With chunks, start a fresh Blue Vela vLLM server when needed|
|`--vllm-tensor-parallel N`|Override tensor parallel for chunk relaunch|
|`--vllm-max-model-len N`|Override model length for chunk relaunch|

`bench swebench-live` runs Microsoft SWE-bench Live. It shares most SWE-bench flags, but it is an advanced command and normally not the first place to debug harness changes.

## Aider Polyglot

Aider Polyglot exercises small tasks across several languages. It has a first attempt and, unless you pass `--no-retry`, a second attempt that sees test output from the first failure.

```bash
uv run mcode bench aider-polyglot \
  --backend openai \
  --model granite4 \
  --db experiments/results/polyglot.db
```

Useful Aider Polyglot flags:

|Flag|Use|
|-|-|
|`--language X`|`all`, or one language such as `python`, `go`, `rust`, `js`, `cpp`, or `java`|
|`--exercise X`|Run one exercise. Use it with a concrete language|
|`--benchmark-root DIR`|Use an existing clone of the Aider Polyglot benchmark|
|`--no-retry`|Disable the second attempt with feedback|
|`--retry-loop-budget N`|Turn budget for the second attempt|
|`--sampling`|Use `none` or `multiturn`|
|`--selection-attempts N`|Generate multiple candidates and select one|

Before a full polyglot run, use:

```bash
uv run mcode deps toolchains --benchmark aider-polyglot
```

## Terminal-Bench 2.0

`bench terminal-bench` is experimental. It uses Harbor, the official Terminal-Bench 2.0 harness. Harbor runs the task containers and verifiers; mCode imports the results into SQLite and writes artifact manifests that point at the Harbor trial directories. The local Harbor path has smoke coverage only; full runs and Blue Vela execution are not validated yet.

Start with the oracle agent to validate Harbor and Docker:

```bash
uv run mcode bench terminal-bench \
  --agent oracle \
  --model unused \
  --limit 1 \
  --db experiments/results/terminal-bench-oracle.db
```

Then run the mCode terminal agent or a built-in Harbor agent:

```bash
uv run mcode bench terminal-bench \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --loop-budget 25 \
  --n-concurrent 2 \
  --limit 5 \
  --db experiments/results/terminal-bench.db
```

Terminal-Bench-specific flags:

|Flag|Use|
|-|-|
|`--agent`|`mcode` by default, or a Harbor agent such as `oracle`, `claude-code`, or `codex`|
|`--dataset`|Harbor dataset id. Defaults to `terminal-bench/terminal-bench-2`|
|`--jobs-dir`|Directory where Harbor writes job outputs|
|`--env`|Harbor environment provider, usually `docker` locally|
|`--n-concurrent N`|Concurrent Harbor trials|
|`--timeout-multiplier F`|Scale Harbor task timeouts|
|`--harbor-arg X`|Append a raw argument to `harbor run`|

See [`terminalbench.md`](terminalbench.md) for setup details and caveats.

## Suite

`bench suite` runs a mixed manifest through the shared runner. It is the command I reach for after changing the harness because it catches more than a single SWE-only smoke.

```bash
uv run mcode bench suite \
  --backend openai \
  --model granite4 \
  --db experiments/results/mixed-suite.db
```

The suite has a bundled manifest, but you can supply your own:

```bash
uv run mcode bench suite \
  --backend openai \
  --model granite4 \
  --suite-file path/to/suite.json \
  --db experiments/results/custom-suite.db
```

Suite-specific flags are `--suite-file`, `--retry-loop-budget`, and the SWE-bench eval controls `--timeout`, `--mem-limit`, `--pids-limit`, `--cpu-limit`, and image digest checking. The normal phase, artifact, shard, remote, and JSON flags also work.

## Phases and artifacts

The default phase is `run`, which generates and evaluates in one pass. Use `generate` when you want to save candidates without running official evaluation. Use `evaluate` to read saved artifacts and write result rows later.

```bash
uv run mcode bench swebench-lite \
  --backend openai --model granite4 \
  --limit 16 \
  --db experiments/results/lite-generate.db \
  --artifact-dir experiments/results/lite-artifacts \
  --phase generate

uv run mcode bench swebench-lite \
  --backend openai --model granite4 \
  --limit 16 \
  --db experiments/results/lite-evaluate.db \
  --artifact-dir experiments/results/lite-artifacts \
  --phase evaluate
```

Artifact commands are grouped under `mcode bench artifacts`:

```bash
uv run mcode bench artifacts list --db experiments/results/lite-evaluate.db
uv run mcode bench artifacts list --db experiments/results/lite-evaluate.db --task-id astropy__astropy-12907 --phase evaluate --json
uv run mcode bench artifacts show astropy__astropy-12907 --db experiments/results/lite-evaluate.db
uv run mcode bench artifacts show astropy__astropy-12907 --db experiments/results/lite-evaluate.db --candidate-index 0
uv run mcode bench artifacts patch astropy__astropy-12907 --db experiments/results/lite-evaluate.db --out candidate.patch
uv run mcode bench artifacts replay astropy__astropy-12907 --db experiments/results/lite-generate.db --out-db replay.db
uv run mcode bench artifacts fetch <run-id> --dest research/run-artifacts
uv run mcode bench artifacts fetch --db research/run/results.db --json
```

`artifacts list` accepts `--db`, `--run-id`, `--task-id`, `--phase`, and `--json`. If you omit `--run-id`, it uses the latest run in that DB.

`artifacts show` takes a task id and accepts `--db`, `--run-id`, and `--candidate-index`. Without a candidate index, it prints the task manifest.

`artifacts patch` takes a task id and accepts `--db`, `--run-id`, `--candidate-index`, and `--out`. Without `--out`, it prints the patch.

`artifacts replay` takes a task id and accepts `--db`, `--run-id`, `--out-db`, `--candidate-index`, `--benchmark-root`, `--artifact-dir`, and `--fetch-missing-artifacts`. It re-evaluates a saved candidate through the benchmark adapter.

`artifacts fetch` fetches a remote artifact directory for a recorded Blue Vela run. Pass a run id, or pass `--db` and let mCode resolve the latest fetchable run for that DB. It also accepts `--dest` and `--json`.

The old dashed commands, such as `mcode bench artifacts-list`, still work as hidden aliases for old scripts. New docs and new scripts should use the grouped form.


## JSON benchmark events

Every benchmark accepts `--json`. The stream is one JSON object per line with a monotonic `seq` field, which makes it safe to pipe through `jq` or append to a log file.

```bash
uv run mcode bench smoke --backend openai --model granite4 --shards 4 --json | jq -c '.'
```

A typical sharded run emits events like these:

```jsonl
{"seq": 1, "kind": "run_start", "data": {"benchmark": "smoke", "model": "granite4", "shards": 4}}
{"seq": 2, "kind": "shard_start", "shard": 0, "data": {"db": "...", "log": "..."}}
{"seq": 3, "kind": "shard_stdout", "shard": 0, "data": {"line": "..."}}
{"seq": 4, "kind": "shard_done", "shard": 0, "data": {"exit_code": 0}}
{"seq": 5, "kind": "merged", "data": {"db": "experiments/results/results.db"}}
{"seq": 6, "kind": "summary", "data": {"passed": 12, "total": 16}}
```

The event names used today include `run_start`, `shard_start`, `shard_stdout`, `shard_done`, `shard_failed`, `shard_infra`, `infra_failure`, `merged`, `summary`, `remote_stdout`, and `info`. Human output includes live task trace by default. JSON stays compact unless `MCODE_LIVE_TRACE=1` is set.

## Bench run records

The launch state file records bench runs so you can find, show, cancel, and prune them without opening JSON by hand.

```bash
uv run mcode bench list
uv run mcode bench list --wide
uv run mcode bench list --status running
uv run mcode bench list --benchmark swebench-lite --limit 10
uv run mcode bench list --artifacts
uv run mcode bench list --json
```

The default list is compact and newest first. Use `--wide` when you need DB paths, remote paths, target, shard counts, artifact status, and fetch status.

```bash
uv run mcode bench show --latest
uv run mcode bench show <run-id>
uv run mcode bench show <compact-id>
uv run mcode bench show <run-id> --json
```

`bench show` prints the run record, DB summary, failed task rows, paths, progress, and follow-up commands. Compact ids from `bench list` are accepted when they are unambiguous.

Prune is deliberately safe by default:

```bash
uv run mcode bench prune
uv run mcode bench prune --status failed --older-than 7d
uv run mcode bench prune --status failed --older-than 7d --yes
uv run mcode bench prune --any-db --older-than 30d --yes
uv run mcode bench prune --json
```

Without `--yes`, prune only prints what it would remove. Without `--any-db`, it only targets records whose DB path is missing.


## Run record details

Run ids are long enough to be unique, but the default table shows compact ids to keep the output readable. `bench show` accepts a compact id if it matches exactly one run. If it is ambiguous, use the full id from `bench list --wide` or `bench list --json`.

`bench show --latest` means the newest run after loading state. It is intentionally a read command; it does not guess a run id from the results DB. The detailed view shows progress for active local runs, DB summaries for fetched DBs, failed task rows, rerun metadata, and artifact fetch commands when they apply.

`bench prune` accepts durations such as `7d`, `12h`, and `30m` for `--older-than`. It is a dry run unless `--yes` is present. By default it only prunes records whose DB path is missing, which keeps real result records from disappearing just because the state file is noisy.

## Benchmark cancellation

```bash
uv run mcode bench cancel <run-id>
```

Cancellation depends on how the run was started. Local sharded runs get SIGTERM for each worker pid, then SIGKILL for stragglers after a short grace period. Blue Vela runs get process-group termination over SSH, followed by a check that the process is gone. Single in-process local runs are not cancellable from another shell, so the command exits 2 and tells you to use Ctrl+C in the original terminal.

If Blue Vela kill verification fails, the run stays marked as running. That is safer than claiming success while a remote job may still be alive.

## Manual shard merge

Most sharded runs merge automatically. Use this command when you are recovering by hand:

```bash
uv run mcode bench merge-shards \
  --out merged.db \
  shard-a.db shard-b.db shard-c.db
```

Add `--force` to overwrite an existing output DB. The older top-level `mcode merge-shards` command still exists as a hidden compatibility alias, but the documented command is `mcode bench merge-shards`.

## Results, compare, and CSV export

`results` reads one or more SQLite DBs and prints pass-rate summaries:

```bash
uv run mcode results --db experiments/results/results.db
uv run mcode results --db-dir experiments/results --benchmark swebench-lite
uv run mcode results --db-glob 'research/*/results.db' --time
uv run mcode results --db-dir research --compare-configs --json
```

Results flags:

|Flag|Use|
|-|-|
|`--db PATH`|Read a DB. Repeat it for more DBs|
|`--db-glob TEXT`|Read DBs matching a quoted glob|
|`--db-dir DIR`|Recursively scan a directory for `*.db`|
|`--benchmark X`|Filter by benchmark name|
|`--model M`|Filter by model|
|`--backend B`|Filter by backend|
|`--suite S`|Filter by suite name|
|`--suite-entry E`|Filter by suite entry|
|`--loop-budget N`|Filter by loop budget|
|`--timeout N`|Filter by timeout|
|`--compare-configs`|Group by backend, timeout, and loop budget|
|`--time`|Include sec/solve, solves/hour, and p95 timing|
|`--json`|Print JSON|

`compare` is the regression gate:

```bash
uv run mcode compare \
  --baseline-dir experiments/results/baseline.db \
  --candidate-dir experiments/results/candidate.db \
  --max-lost 0
```

It accepts DB files or directories on either side. Useful gate flags are `--max-lost`, `--min-net`, `--min-candidate-pass-rate`, and `--min-candidate-passed`. You can also filter with `--task-ids`, `--benchmark`, `--suite`, and `--suite-entry`.

For generate-only experiments, JSON compare output includes artifact counts such as generated tasks, evaluated tasks, selected verified candidates, and selected patch bytes. That lets you compare scaffolds before running official evaluation.

CSV export writes runs, task results, diagnostic events, and artifact tables:

```bash
uv run mcode export-csv \
  -i experiments/results \
  --out-dir experiments/results \
  --prefix mcode
```

Add `--include-logs` only when you really want stdout, stderr, and error text in the CSV. Those columns can get large.


`export-csv -i` accepts both files and directories. A directory input exports top-level `*.db` files in that directory and skips shard DBs, so a normal results directory does not double-count `results.db-shards/`. Repeat `-i` when you want to combine several explicit DBs or directories in one export.

Compatibility note: older scripts can still call `mcode bench artifacts-list` and the other dashed artifact commands. They are hidden from help. The top-level `mcode merge-shards` alias also still works but is hidden; use `mcode bench merge-shards` in new commands.

## Dependency management

`deps sync` wraps the repo's `uv` setup. With no flags it installs the default dev extra.

```bash
uv run mcode deps sync
uv run mcode deps sync --extra swebench --extra datasets
uv run mcode deps sync --extra swebench --extra datasets --extra observability
uv run mcode deps sync --no-dev
MCODE_MELLEA_PATH=/path/to/mellea uv run mcode deps sync
```

`MCODE_MELLEA_PATH` replaces the pinned Mellea dependency with a local checkout. I use that when changing Mellea and mCode together.

`deps toolchains` checks or installs Aider Polyglot language runtimes:

```bash
uv run mcode deps toolchains --benchmark aider-polyglot
uv run mcode deps toolchains --benchmark aider-polyglot --language go --language rust
uv run mcode deps toolchains --benchmark aider-polyglot --install
```

With `--install`, mCode uses the platform package manager when supported: Homebrew on macOS, winget or choco on Windows, and apt, dnf, or pacman on Linux.

## Environment variables

The env vars below are the ones that change behavior often enough to document.

|Variable|Use|
|-|-|
|`MCODE_DEBUG`|Show raw tracebacks|
|`MCODE_LAUNCH_STATE`|Use a different launch and bench state file|
|`MCODE_LAUNCH_CONFIG`|Use a different launch config file|
|`MCODE_RESULTS_DIR`|Directory checked by doctor for results writability|
|`MCODE_CACHE_DIR`|Bench cache dir. Defaults to XDG cache or `/tmp/mcode-cache`|
|`MCODE_KEEP_IMAGES`|Skip post-task image cleanup|
|`MCODE_SKIP_IMAGE_PULL`|Use cached SWE-bench images instead of pulling first|
|`MCODE_CONTEXT_WINDOW`|Override model context window|
|`MCODE_MAX_NEW_TOKENS`|Override max output tokens|
|`MCODE_REACT_TIMEOUT`|Timeout for the ReACT loop|
|`MCODE_LIVE_TRACE`|Control live progress trace. `0` mutes human trace, `1` adds trace to JSON|
|`MCODE_AIDER_POLYGLOT_ROOT`|Aider Polyglot benchmark checkout override|
|`MCODE_PODMAN_PULL_ATTEMPTS`|Retry count for Podman pulls|
|`MCODE_PODMAN_PULL_RETRY_DELAY`|Seconds between Podman pull retries|
|`MCODE_NO_COLOR` / `NO_COLOR`|Disable ANSI color|
|`MCODE_MELLEA_PATH`|Use a local Mellea checkout for deps sync|
|`OPENAI_BASE_URL`|Override endpoint discovery for `--backend openai`|
|`OPENAI_API_KEY`|API token for an OpenAI-compatible endpoint. Defaults to `dummy` for local servers|
|`MELLEA_METRICS_ENABLED` / `MELLEA_METRICS_CONSOLE`|Mellea metrics when the observability extra is installed|
|`MELLEA_TRACE_APPLICATION` / `MELLEA_TRACE_BACKEND` / `MELLEA_LOGS_OTLP`|Mellea tracing settings|


Additional environment variables used by lower-level paths:

|Variable|Use|
|-|-|
|`MCODE_TMPDIR`|Base temporary directory for mCode temp work|
|`MCODE_BASH_TIMEOUT`|Default timeout for shell tool execution inside the agent|
|`MCODE_FUZZY_EDIT`|Set to `0` to disable fuzzy fallback for edit matching|
|`MCODE_DOCKER_CONNECT_RETRIES`|Docker client connection retry count|
|`MCODE_DOCKER_RETRY_DELAY`|Delay between Docker connection retries|
|`MCODE_PODMAN_LOCK_DIR`|Lock directory for Podman image operations|
|`MCODE_SWEBENCH_CPU_LIMIT`|Env equivalent of SWE-bench `--cpu-limit`|
|`MCODE_SWEBENCH_CHECK_IMAGE_DIGESTS`|Env equivalent of `--check-image-digests`|
|`MCODE_GIT_SHA` / `GITHUB_SHA`|Code revision stamped into artifact metadata|

Most users only need the shorter table above. These are here so cluster scripts and CI jobs do not have to learn them from source.

## Local workflow example

```bash
uv run mcode doctor local-ollama
uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <server-id> --timeout 120

uv run mcode bench smoke \
  --backend openai \
  --model granite4 \
  --shards 4 \
  --db experiments/results/local-smoke.db

uv run mcode bench show --latest
uv run mcode results --db experiments/results/local-smoke.db --time
uv run mcode launch stop <server-id>
```

## Blue Vela workflow example

```bash
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
uv run mcode launch sync bluevela
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json
uv run mcode launch wait <server-id> --timeout 1800

MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --shards 4 \
  --db research/bluevela-smoke/results.db

uv run mcode bench show --latest
uv run mcode results --db research/bluevela-smoke/results.db --time
uv run mcode launch stop <server-id>
```
