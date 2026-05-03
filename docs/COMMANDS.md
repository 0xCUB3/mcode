# Commands

Reference for every mcode command and flag. Run `mcode <command> --help` for the live source-of-truth.

## Global

```bash
mcode --version                    # print the installed version and exit
mcode --verbose | -v               # raise the Mellea + mcode logger to INFO (otherwise WARNING)
mcode --help                       # list top-level commands
NO_COLOR=1 mcode ...               # disable ANSI color (CI / piped logs)
MCODE_NO_COLOR=1 mcode ...         # mcode-specific no-color override
MCODE_DEBUG=1 mcode ...            # disable formatted error layout, surface raw tracebacks
MCODE_LAUNCH_STATE=/path/state.json mcode ...   # override the persistent state file path
```

Exit codes:

|Code|Meaning|
|-|-|
|0|Success|
|1|User-actionable failure (formatted as `✗ what / why / next / logs` on stderr)|
|2|Usage error or "not cancellable from another shell"|
|3|`mcode launch wait`: server id not found|
|86|Retryable infrastructure failure (sharded benchmark)|
|130|Interrupted (Ctrl+C)|

## Doctor — system + launch diagnostics

```bash
mcode doctor                       # all system + per-target checks
mcode doctor bluevela              # only Blue Vela checks (queues, group, ssh)
mcode doctor local-vllm
mcode doctor local-ollama
mcode doctor bluevela --init --login user@login-host    # bootstrap launch.toml
mcode doctor bluevela --deep       # extra probes (slow)
```

System checks (no target):

- results dir writable (`MCODE_RESULTS_DIR` or `experiments/results`)
- container runtime (podman or docker on PATH)
- mellea importable
- ruff present (PATH or via uv)

Per-target checks delegate to `src/mcode/launch/<target>.py:doctor()`.

`mcode launch doctor` is preserved as an alias for back-compat.

## Watch — live dashboard

```bash
mcode watch                        # refreshes every 2s; quits on Ctrl+C
```

Combines `launch status` + `bench list` into a single Rich Live view. Recovers automatically from transient state-file read failures (partial writes, lock contention) by rendering the last-good snapshot with a warning footer.

## Launch — server lifecycle

```bash
mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
mcode launch local-vllm --model Qwen/Qwen2.5-0.5B
mcode launch local-ollama --model granite4
mcode launch status [--json] [--raw]
mcode launch logs <id>
mcode launch wait <id> [--timeout 600] [--poll 2.0] [--json]
mcode launch stop <id>
mcode launch stop --all                    # only recorded servers; never `bkill 0`
mcode launch refresh                       # re-query each server/run, persist updated status
mcode launch sync bluevela [--dry-run] [--src DIR] [--bootstrap]
mcode launch doctor <target> [--init] [--login user@host] [--deep]
```

Flags:

- `--model / -m` — HF model id, e.g. `Qwen/Qwen3.6-35B-A3B`
- `--json` — endpoint, status, etc. as machine-readable JSON
- `--raw` (status) — include the raw LSF state in JSON
- `--timeout N` (wait) — seconds before exit code 2; default 600
- `--poll s` (wait) — seconds between polls; default 2.0
- `--dry-run / -n` (sync) — preview the rsync, no file transfer
- `--src DIR` (sync) — local source path; defaults to git rev-parse root
- `--bootstrap` (sync) — claim a populated remote dir (creates safety marker)

Wait exit codes: 0 healthy, 1 failed-or-stopped, 2 timeout (or transient state read failure past deadline), 3 no-such-id.

## Bench — run benchmarks

```bash
uv run mcode bench list [--json] [--benchmark NAME] [--status running|done|failed|stopped] [--artifacts] [--limit N]
uv run mcode bench cancel <run-id>
uv run mcode bench swebench-live   --model M  [flags]
uv run mcode bench swebench-lite   --model M  [flags]
uv run mcode bench aider-polyglot  --model M  [flags]
uv run mcode bench smoke           --model M  [flags]
uv run mcode bench suite           --model M  [flags]
```

The normal benchmark path is intentionally small: one solver loop, the built-in code tools, optional split phases, SQLite results, artifacts, replay, and suite runs. Treat the flags below as controls for measurement and operations first. New capability should show up in results before it becomes part of the default path.

### Common flags (`swebench-live` / `swebench-lite` / `aider-polyglot` / `smoke` / `suite`)

- `--model M` — Mellea model id (required)
- `--backend B` — Mellea backend: `ollama` (default for swebench), `openai` (default for smoke / aider)
- `--loop-budget N` — agent retry budget per task; default 15
- `--temperature F` — sampling temperature
- `--seed N` — random seed
- `--timeout N` — eval timeout per task in seconds
- `--limit N` — run the first N tasks
- `--task-ids X` — comma-separated ids or path to JSON / text file
- `--db PATH` — SQLite results DB path
- `--phase {run,generate,evaluate}` — `run` generates and evaluates in one pass; `generate` writes task artifacts only; `evaluate` loads those artifacts and records eval rows
- `--artifact-dir DIR` — directory for generated task artifacts; defaults next to `--db` as `<db-stem>-artifacts`
- `--shards N` — run N shard workers, merge per-shard DBs into `--db`
- `--on {local,bluevela}` — where to run (default `local`)
- `--fetch-db / --no-fetch-db` — rsync the DB back from Blue Vela
- `--json` — machine-readable event stream (one JSON object per line, monotonic `seq`)
- Advanced: `--shard-count C / --shard-index I` runs one manual shard. The auto `--shards` path uses this internally
- Advanced: `--diagnostic-traces / --no-diagnostic-traces` persists compact bench trace events for debugging

Re-running the same bench command against the same `--db` resumes the matching run. Completed task rows are skipped, retryable infra rows are retried, and sharded runs reuse stable shard DBs before merging whatever completed rows exist.

### Artifact-backed phases

Use `--phase generate` when you want to produce patch artifacts without running official evaluation yet. Reuse the same `--artifact-dir` with `--phase evaluate` to load the saved candidates and write results to the DB. The default `--phase run` keeps the old one-command path and still writes artifacts as it goes.

```bash
uv run mcode bench swebench-lite \
  --backend openai --model granite4 --limit 16 --shards 4 \
  --db experiments/results/lite-split.db \
  --artifact-dir experiments/results/lite-split-artifacts \
  --phase generate

uv run mcode bench swebench-lite \
  --backend openai --model granite4 --limit 16 --shards 4 \
  --db experiments/results/lite-split.db \
  --artifact-dir experiments/results/lite-split-artifacts \
  --phase evaluate
```

### `suite`

The mixed suite runs several benchmark slices through the same phase runner and writes all runs into one DB. Use it when you want a small, broader regression sweep instead of another SWE-only pass. For harness A/B work, run the bundled `src/mcode/bench/fixtures/aider-regression-suite.json` before and after a loop change, then gate with `mcode compare --max-lost 0`.

```bash
uv run mcode bench suite \
  --backend openai --model granite4 \
  --db experiments/results/mixed-suite.db \
  --phase run

uv run mcode bench suite \
  --backend openai --model granite4 \
  --db experiments/results/mixed-suite.db \
  --artifact-dir experiments/results/mixed-suite-artifacts \
  --phase generate

uv run mcode bench suite \
  --backend openai --model granite4 \
  --db experiments/results/mixed-suite-eval.db \
  --artifact-dir experiments/results/mixed-suite-artifacts \
  --phase evaluate
```

- `--suite-file PATH` — optional JSON manifest overriding the bundled suite
- `--shards N` — run N suite shard workers and merge all run DBs back into `--db`
- `--retry-loop-budget N` — Aider Polyglot retry budget inside the suite
- Advanced: `--shard-count C / --shard-index I` runs one manual shard for the whole suite. Each slice applies the same shard split

### Artifact inspection

Once a split-phase or suite run has written artifacts, inspect them directly from the DB without spelunking through directories by hand.

```bash
uv run mcode bench artifacts-list --db experiments/results/mixed-suite-evaluate.db
uv run mcode bench artifacts-list --db experiments/results/mixed-suite-evaluate.db --task-id python/affine-cipher --phase evaluate --json
uv run mcode bench artifacts-show python/affine-cipher --db experiments/results/mixed-suite-evaluate.db
uv run mcode bench artifacts-patch python/affine-cipher --db experiments/results/mixed-suite-evaluate.db --out candidate.patch
uv run mcode bench artifacts-replay python/affine-cipher --db experiments/results/mixed-suite-generate.db
uv run mcode bench artifacts-fetch bench-<run-id> --dest research/mixed-suite/artifacts
uv run mcode bench artifacts-fetch --db experiments/results/mixed-suite-generate.db --json
```

- `artifacts-list` shows task ids, phase, selected candidate index, whether that candidate verified, selected patch bytes, candidate count, evaluation count, and manifest path for one run. Add `--task-id`, `--phase`, or `--json` when you want a narrower machine-readable inventory
- `artifacts-show` prints the saved task manifest JSON for one task, or one candidate entry with `--candidate-index N`
- `artifacts-patch` prints the selected candidate diff, or writes it to a file with `--out PATH`
- `artifacts-replay` re-evaluates one saved candidate into a fresh DB, optionally with `--candidate-index N`, `--out-db PATH`, and `--benchmark-root PATH` for cross-machine polyglot artifacts
- `artifacts-fetch` downloads the saved remote artifact directory later, using either a recorded run id or the latest fetchable run for a local `--db` path. Add `--json` when another script needs the resolved local and remote paths
### `swebench-live` / `swebench-lite` extras

The SWE-bench extras below are mostly eval controls and ablation knobs. The default kernel does not need multiple samples or candidate selection to run; use those flags when measuring variance or isolating a change.

- `--split` — `test` / `lite` / `verified` / `full` / `dev`
- `--mem-limit` — eval container memory limit; default `4g`
- `--pids-limit` — eval container PID limit; default 512
- `--n-samples N` — outer attempts when `--sampling none`, sampling budget fallback otherwise
- `--sampling {none,multiturn}` — Mellea sampling strategy
- `--sampling-budget N` — sampling-loop budget override
- `--selection-attempts N` — independent full-budget trajectories; pick one patch before official eval

### `swebench-lite` only

- `--namespace` — empty string forces local image builds; `swebench` (default) pulls prebuilt
- `--arch` — image arch override (`auto` / `x86_64` / `arm64`)
- `--max-workers N` — local image build concurrency
- `--force-rebuild` — rebuild eval images even if cached
- `--dataset` — HF dataset id (defaults to `SWE-bench/SWE-bench_Lite`)

### `aider-polyglot`

- `--language X` — `all` (default) or one of `python`, `go`, `rust`, `js`, `cpp`, `java`
- `--exercise X` — single exercise (requires concrete `--language`)
- `--benchmark-root DIR` — override the polyglot checkout location
- `--no-retry` — disable the second-pass retry loop
- `--retry-loop-budget N` — retry-attempt loop budget

### `smoke`

A 16-task SWE-bench Verified diagnostic slice (astropy + 6 projects). It runs `swebench-lite` under the hood with a bundled task-id list and sensible defaults, so the common phase and artifact flags apply there too.

### JSON event stream

Every bench command supports `--json`. Events are line-delimited JSON with strictly monotonic `seq`. Set `MCODE_LIVE_TRACE=1` to include compact per-turn model/tool events while each task is running:

```jsonl
{"seq": 1, "ts": 1719445200.123, "kind": "run_start", "data": {"benchmark": "smoke", "model": "...", "shards": 4}}
{"seq": 2, "ts": 1719445200.456, "kind": "shard_start", "shard": 0, "data": {"db": "...", "log": "..."}}
{"seq": 3, "ts": 1719445201.012, "kind": "shard_stdout", "shard": 0, "data": {"line": "..."}}
{"seq": 4, "ts": 1719445230.789, "kind": "shard_done", "shard": 0}
{"seq": 5, "ts": 1719445999.000, "kind": "merged", "data": {"db": "..."}}
```

Kinds: `run_start`, `shard_start`, `shard_stdout`, `shard_done`, `shard_failed`, `shard_infra`, `infra_failure`, `merged`, `summary`, `remote_stdout`, `info`.
`bench list` is sorted newest-first after filtering. Use `--limit N` when the state file is noisy and you only care about the most recent runs. When a remote artifact directory exists, `--artifacts` filters to those runs, and the table marks whether the artifacts were already fetched locally.




### Cancel semantics

`mcode bench cancel <run-id>` dispatches by run shape:

|Shape|Action|
|-|-|
|`shard_pids` non-empty|local sharded → SIGTERM each pid, SIGKILL stragglers after 10s|
|`remote` dict non-empty|Blue Vela → SSH `kill -TERM -<pid>`, `kill -KILL -<pid>`, then `kill -0` to verify dead|
|neither|in-process single run → exit 2 with "not cancellable from another shell"|

State transitions to `RunStatus.STOPPED` with `metadata.cancel_reason = "user"`. If the remote process can't be confirmed dead (kill verification fails), the cancel is rejected with a `MCodeError` so the record stays `running` and the user can retry rather than silently leak the job.

## Results

```bash
uv run mcode results [--db PATH | --db-glob 'g' | --db-dir DIR] [--benchmark X] [--model M] [--backend B] [--suite S] [--suite-entry E] [--loop-budget N] [--timeout N] [--compare-configs] [--time] [--json]
uv run mcode compare --baseline-dir A --candidate-dir B [--benchmark X] [--suite S] [--suite-entry E] [--task-ids file] [--max-lost N] [--min-net N] [--min-candidate-pass-rate F] [--json]
mcode report [--db ... | --db-dir DIR] [--benchmark X] --out report.html
mcode merge-shards --shards-glob 'glob' --out merged.db
mcode export-csv -i DIR --out-dir DIR --prefix mcode [--include-logs]
```

`compare` accepts either DB files or directories on both sides. For generate-only runs, `--json` includes artifact summary fields like generated task count, evaluated task count, selected verified candidate count, and total selected patch bytes so you can compare two unevaluated scaffolds before paying for official eval.

`results` flags:

- `--db PATH` (repeatable) — explicit DB paths
- `--db-glob 'glob'` (repeatable) — glob (quote it!)
- `--db-dir DIR` (repeatable) — recursively scan for `*.db`
- `--benchmark X` — filter by benchmark name
- `--model M / --backend B / --loop-budget N / --timeout N` — config filters
- `--compare-configs` — group results by `(backend_name, timeout_s, loop_budget)`
- `--time` — include `sec/solve`, `solves/hour`, p95 metrics

`export-csv` always writes runs, task_results, and artifact CSVs. The run and task result exports now include `suite_name` and `suite_entry_name`, and the artifact exports include those suite columns too so mixed-suite analysis stays join-free.



`report` produces a Plotly HTML report comparing pass rate vs time-to-solve across the matched runs.

## Deps

```bash
mcode deps sync                                # default extras (dev)
mcode deps sync --extra swebench --extra datasets
mcode deps sync --extra swebench --extra datasets --extra observability
mcode deps sync --no-dev                       # skip dev extras
MCODE_MELLEA_PATH=/path/to/mellea-checkout mcode deps sync ...
```

`MCODE_MELLEA_PATH` overrides the pinned upstream Mellea with a local working copy.

## Environment variables

|Variable|Purpose|
|-|-|
|`MCODE_DEBUG`|Disable formatted error layout, surface raw tracebacks|
|`MCODE_LAUNCH_STATE`|Override the persistent state file path|
|`MCODE_LAUNCH_CONFIG`|Override `launch.toml` path|
|`MCODE_RESULTS_DIR`|Override the results dir doctor checks for writability|
|`MCODE_CACHE_DIR`|Bench cache dir (otherwise `XDG_CACHE_HOME/mcode` or `/tmp/mcode-cache`)|
|`MCODE_KEEP_IMAGES`|Skip post-task image cleanup|
|`MCODE_SKIP_IMAGE_PULL`|Skip Docker image pre-pull (use existing local images)|
|`MCODE_CONTEXT_WINDOW`|LLM context window override (int)|
|`MCODE_MAX_NEW_TOKENS`|LLM max output tokens|
|`MCODE_REACT_TIMEOUT`|ReACT loop timeout in seconds|
|`MCODE_AIDER_POLYGLOT_ROOT`|Aider polyglot benchmark root override|
|`MCODE_PODMAN_PULL_ATTEMPTS`|Podman pull retry count|
|`MCODE_PODMAN_PULL_RETRY_DELAY`|Seconds between podman pull retries|
|`MCODE_NO_TTY`|Force non-TTY mode for the dashboard|
|`MCODE_NO_COLOR` / `NO_COLOR`|Disable ANSI color|
|`MCODE_MELLEA_PATH`|Local Mellea checkout for `deps sync`|
|`OPENAI_BASE_URL`|Override the auto-resolved endpoint for `--backend openai`|
|`OPENAI_API_KEY`|API key for `--backend openai` (defaults to `dummy`)|
|`MELLEA_METRICS_ENABLED` / `MELLEA_METRICS_CONSOLE`|Mellea token metrics (observability extra)|
|`MELLEA_TRACE_APPLICATION` / `MELLEA_TRACE_BACKEND` / `MELLEA_LOGS_OTLP`|Mellea tracing (observability extra)|

## Examples

End-to-end Blue Vela run:

```bash
# bootstrap config and validate
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
uv run mcode doctor bluevela

# push the local repo
uv run mcode launch sync bluevela

# bring up a vLLM server
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json

# wait for it (block until ready or 20 min)
uv run mcode launch wait server-bv-abc123 --timeout 1200

# run the smoke slice with 4 shards and a JSON event stream
MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela --shards 4 --json

# inspect / cancel
uv run mcode bench list --benchmark suite --artifacts --limit 5
uv run mcode bench list --json | jq '.[] | select(.status == "running")'
uv run mcode bench cancel run-abc123
uv run mcode launch stop server-bv-abc123
```
`compare` accepts either DB files or directories on both sides. When a run has no evaluated task rows yet, the JSON output still includes artifact summary fields such as generated task count, evaluated task count, selected verified candidate count, and total selected patch bytes so you can compare generate-only experiments before official eval.


Local Ollama smoke:

```bash
uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <id> --timeout 120
uv run mcode bench swebench-lite --backend openai --model granite4 --limit 16 --shards 4
```
