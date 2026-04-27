# mCode

mCode is an agentic coding benchmark harness built on [Mellea](https://mellea.ai). It runs SWE-bench Verified, SWE-bench Live, Aider Polyglot, and a smoke slice end-to-end on a local model or on a remote vLLM server (Blue Vela LSF or any OpenAI-compatible endpoint).

## Top results

Best mCode harness results to date, on Blue Vela with Qwen3.6-35B-A3B served via vLLM:

|Benchmark|Baseline|mCode|Δ|
|-|-|-|-|
|SWE-bench Verified (271 tasks, partial run)|–|**169/271 = 62.4%**|new|
|Aider Polyglot (225 tasks, full run)|103/225 = 45.8%|**190/225 = 84.4%**|+87 solves|
|SWE-bench Verified, 16-task smoke slice|4/16 = 25.0%|**9/16 = 56.3%**|+5 solves|

Baseline = single-pass run without selection / multiturn / control-loop fixes. mCode = full harness with selection, multiturn sampling, workspace-context discovery, generic control-loop nudges, and the verification policy. Source data and per-language breakdowns live under [`research/`](research/).

## Install

```bash
uv sync --extra dev
uv run mcode --help
```

To pin upstream Mellea + the SWE-bench / datasets extras:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

Add the Mellea telemetry / hooks extra for token metrics + tracing:

```bash
uv run mcode deps sync --extra swebench --extra datasets --extra observability
```

To override the pinned Mellea with a local checkout, set `MCODE_MELLEA_PATH=/path/to/mellea` before the command runs.

## Quick start

```bash
# system + per-target diagnostics (results dir writable, podman/docker, mellea importable, ruff)
uv run mcode doctor

# launch a vLLM server somewhere
uv run mcode launch local-vllm  --model Qwen/Qwen2.5-0.5B
uv run mcode launch bluevela    --model Qwen/Qwen3.6-35B-A3B
uv run mcode launch local-ollama --model granite4

# run a benchmark; auto-routes to a healthy server matching --model
uv run mcode bench smoke --backend openai --model Qwen/Qwen3.6-35B-A3B --shards 4

# inspect / manage runs
uv run mcode launch status
uv run mcode bench list
uv run mcode watch
```

## Commands

`mcode --help` lists the full surface. Highlights:

|Command|Purpose|
|-|-|
|`mcode --version`|Print mcode version|
|`mcode doctor [target]`|System + launch diagnostics. With no target: results dir writable, podman/docker on PATH, mellea importable, ruff. With `bluevela` / `local-vllm` / `local-ollama`: per-target probes. `--init --login <user>@<host>` bootstraps `launch.toml` for Blue Vela|
|`mcode watch`|Live Rich dashboard combining `launch status` + `bench list`. Refreshes every 2s. Recovers from transient state-file read failures|
|`mcode bench list`|List historical bench runs from the launch state file. `--json` for machine-readable|
|`mcode bench cancel <run-id>`|Cancel a running bench. Local sharded → SIGTERM/SIGKILL each shard pid. Blue Vela → SSH `kill -TERM/-KILL -<pid>` against the captured process group. In-process single runs are not cancellable from another shell|
|`mcode bench {swebench-live,swebench-lite,aider-polyglot,smoke}`|Run a benchmark. `--shards N` parallelizes locally. `--on bluevela` runs the bench remotely. `--json` emits one JSON event per state change with strictly monotonic `seq` so downstream consumers can reconstruct timing|
|`mcode launch wait <id> [--timeout N]`|Block until the server is healthy / failed / stopped. Exit codes: 0 healthy, 1 failed-or-stopped, 2 timeout, 3 no-such-id. `--json` for scripts|
|`mcode launch {bluevela,local-vllm,local-ollama}`|Spin up a vLLM server and record its endpoint|
|`mcode launch status [--json] [--raw]`|List recorded servers and runs|
|`mcode launch logs <id>`|Print the log path (or for Blue Vela the `ssh tail -f` command)|
|`mcode launch stop <id> | --all`|Stop a server. `--all` is scoped to recorded servers, never `bkill 0`|
|`mcode launch sync bluevela`|Rsync the local repo to `[bluevela].workspace_root`. Refuses `--delete` into a remote dir without our marker; pass `--bootstrap` to claim a populated dir|
|`mcode launch refresh`|Re-query each server / run against its target and persist updated status|
|`mcode launch doctor <target>`|Alias for `mcode doctor <target>` (kept for back-compat)|
|`mcode results [--benchmark X] [--time]`|Query pass rates from the results DB|
|`mcode compare --baseline-dir A --candidate-dir B`|Run-to-run diff|
|`mcode report --db-dir D --out report.html`|Generate a Plotly HTML report|
|`mcode merge-shards --out out.db shard-0.db shard-1.db`|Merge shard SQLite DBs|
|`mcode export-csv -i DIR --out-dir DIR`|Export results DBs to CSV|

Full reference with every flag: [`docs/COMMANDS.md`](docs/COMMANDS.md).

## Bench

```bash
# local
uv run mcode bench swebench-lite --model granite3.3:8b --limit 5
uv run mcode bench swebench-lite --model granite3.3:8b --limit 16 --shards 4

# Blue Vela (auto-resolves a healthy server matching --model)
uv run mcode bench swebench-live --backend openai --model Qwen/Qwen3.6-35B-A3B --limit 10
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench smoke --backend openai --model Qwen/Qwen3.6-35B-A3B --on bluevela --shards 4

# JSON event stream — one line per state change with monotonic seq
uv run mcode bench smoke --backend openai --model Qwen/Qwen3.6-35B-A3B --shards 4 --json | jq -c '.'
```

Useful flags:

- `--loop-budget N` — agent retry budget per task
- `--timeout N` — per-task eval timeout (seconds)
- `--limit N` — first N tasks
- `--shards N` — auto-spawn N workers, merge per-shard DBs back into `--db`
- `--shard-count C --shard-index I` — manual single-shard mode for external fan-out
- `--sampling {none,rejection,repair,multiturn,sofai}` — Mellea sampling strategy
- `--sampling-budget N` — sampling-loop budget override
- `--selection-attempts N` — independent full-budget trajectories; pick one patch before official eval
- `--n-samples N` — outer attempts when `--sampling none`
- `--task-ids` — comma-separated ids or path to JSON / text file
- `--on {local,bluevela}` — where to run the bench
- `--json` — machine-readable event stream

For SWE-bench Lite, `--namespace ""` forces local image builds instead of pulling prebuilt ones.

## Launch

`mcode launch` spins up a vLLM server and records its endpoint so the bench can route to it.

First time on Blue Vela:

```bash
uv run mcode doctor bluevela --init --login <user>@<login-host>
uv run mcode doctor bluevela
```

Push the local repo:

```bash
uv run mcode launch sync bluevela              # subsequent syncs
uv run mcode launch sync bluevela --bootstrap  # first sync into a non-empty remote dir
uv run mcode launch sync bluevela --dry-run    # preview only
```

Launch a server:

```bash
uv run mcode launch bluevela     --model Qwen/Qwen3.6-35B-A3B
uv run mcode launch local-vllm   --model Qwen/Qwen2.5-0.5B
uv run mcode launch local-ollama --model granite4
```

Wait for it to come up, then point the bench at it:

```bash
uv run mcode launch wait server-bv-abc123 --timeout 1200
uv run mcode bench swebench-live --backend openai --model Qwen/Qwen3.6-35B-A3B --limit 10
```

Manage running servers and runs:

```bash
uv run mcode launch status
uv run mcode launch refresh
uv run mcode launch stop <server-id>
uv run mcode launch stop --all
uv run mcode bench list
uv run mcode bench cancel <run-id>
uv run mcode watch
```

Blue Vela profiles ship with correct vLLM flags per model: Qwen3.5 (27B / 35B-A3B), Qwen3.6-35B-A3B, Gemma-4-31B-it, Granite 4.0, MiniMax-M2.5. Add new ones in `src/mcode/launch/profiles.py`.

`--backend openai` auto-resolves the endpoint from `mcode launch status` when a healthy server matches `--model`. `OPENAI_BASE_URL` and `OPENAI_API_KEY` still override if set, including `--on bluevela` runs.

The legacy shell scripts under `deploy/bluevela/` still work as a fallback. See [`deploy/bluevela/README.md`](deploy/bluevela/README.md).

## Results

```bash
uv run mcode results --benchmark swebench-live
uv run mcode results --benchmark swebench-live --time
uv run mcode compare --baseline-dir ./results/baseline --candidate-dir ./results/candidate
uv run mcode report --db-dir ./results --benchmark swebench-live --out ./results/report.html
uv run mcode merge-shards --out ./results/merged.db ./results/swebench-live-shard-0.db ./results/swebench-live-shard-1.db
uv run mcode export-csv -i experiments/results --out-dir experiments/results --prefix mcode
```

`--include-logs` on `export-csv` adds prompt snapshots, stdout, stderr, and error fields.

Task rows in the SQLite results DB store the final prompt snapshot, structured submission, and aggregated prompt / completion / total token usage for the full solve loop.

With the observability extra, Mellea metrics and tracing turn on via env:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export MELLEA_LOGS_OTLP=true
```

## Architecture

- `src/mcode/cli.py` — Typer entry point. Top-level commands: `doctor`, `watch`, `bench`, `launch`, `results`, `compare`, `report`, `merge-shards`, `export-csv`, `deps`.
- `src/mcode/ui/` — shared UI primitives. `console` singleton, error formatter (`MCodeError` → `✗ what / why / next / logs` on stderr), Rich/Plain/JSON `TaskReporter`, sharded-bench `Dashboard` (writer thread + monotonic seq).
- `src/mcode/launch/` — server lifecycle. `bluevela.launch()` decomposed into `_phase_submit`, `_phase_queued`, `_phase_starting`, `_phase_ready`. `sync.py` runs the safety-checked rsync. `formatting.py` renders status output.
- `src/mcode/bench/` — benchmark runners and result merging. `runstate.py` writes the `RunRecord` lifecycle so `bench list` / `bench cancel` / `watch` see live state.
- `src/mcode/util/retry.py` — `with_backoff` exponential-backoff helper.
- `src/mcode/doctor.py` — top-level `mcode doctor` system checks.
- `src/mcode/watch.py` — live `mcode watch` dashboard.
- `tests/` — 369+ unit + integration tests. Mocks every external call (no network, no SSH).

## Notes

- [`research/`](research/) is the canonical home for benchmark run notes and rendered HTML reports.
- [`docs/COMMANDS.md`](docs/COMMANDS.md) is the full command reference with every flag.
- [`docs/e2e-verification.md`](docs/e2e-verification.md) captures live-cluster verification of the launcher.
- [`docs/bluevela-probe-findings.md`](docs/bluevela-probe-findings.md) documents the Blue Vela cluster probe.
- The launch state file lives at `$MCODE_LAUNCH_STATE` (defaults to `~/.config/mcode/launch-state.json`). It is fcntl-locked so concurrent mcode invocations don't race.
- `MCODE_DEBUG=1` disables the formatted error layout and re-enables raw tracebacks.
- Set `NO_COLOR=1` (or `MCODE_NO_COLOR=1`) to disable ANSI colors for plain-text logs and CI.
