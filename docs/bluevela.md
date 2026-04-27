# Blue Vela workflow

Run mCode against a vLLM server on the Blue Vela LSF cluster. Steps are in the order you run them. For local-machine workflow, see [`local.md`](local.md). For the full flag reference, see [`COMMANDS.md`](COMMANDS.md).

The cluster login host is `<user>@login3.bluevela.rmf.ibm.com`. You need IBM VPN access, an SSH key on the login node, and membership in `grp_runtime`.

## 1. Install + bootstrap config

```bash
uv sync --extra dev
uv run mcode --version

# system-level checks first
uv run mcode doctor

# probe the cluster, write ~/.config/mcode/launch.toml
uv run mcode doctor bluevela --init --login skula@login3.bluevela.rmf.ibm.com

# verify everything mcode needs is there
uv run mcode doctor bluevela
```

`doctor bluevela --init` SSHes to the login node, lists your bgroup membership and visible queues, and writes a config file with sensible defaults. After that, every subsequent run reads from `~/.config/mcode/launch.toml` (or `MCODE_LAUNCH_CONFIG`).

The full bench dependency set:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

## 2. Push the local repo to the cluster

mCode runs benchmarks from a checkout on the cluster's shared filesystem. Sync your working copy:

```bash
uv run mcode launch sync bluevela --dry-run    # preview rsync output
uv run mcode launch sync bluevela              # actually sync

# only on the very first sync into a populated remote dir
uv run mcode launch sync bluevela --bootstrap
```

`launch sync` refuses `rsync --delete` into a remote dir without our `.mcode-launch-workspace` marker. The marker is created automatically on a clean dir; `--bootstrap` claims a populated one (destructive — it will mirror what you have locally).

The sync excludes `.git/`, virtual envs, caches, and the launcher-owned remote dirs (`runs/`, `bench-runs/`, `benchmarks/`).

## 3. Launch a vLLM server

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json
```

This submits an LSF job under your `grp_runtime` group, picks the highest-priority queue from your config, uploads `vllm.sh` + `env.json`, and waits for the server to become healthy. The four phases are visible in the live progress UI:

1. **submit** — bsub accepts the job
2. **queued** — LSF transitions PEND → RUN
3. **starting** — container pull + model load + warmup (40 min hard deadline)
4. **ready** — `vllm_host.txt` appears and `/v1/models` returns 200

If you want to script the wait separately:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json &
uv run mcode launch wait server-bv-<id> --timeout 1800
```

Available profiles ship in `src/mcode/launch/profiles.py`: Qwen3.5 (27B / 35B-A3B), Qwen3.6-35B-A3B, Gemma-4-31B-it, Granite 4.0, MiniMax-M2.5. Add new ones there.

## 4. Run a benchmark on Blue Vela

`--on bluevela` runs the bench remotely (under `setsid` so cancel works), `--backend openai` auto-resolves the endpoint to the healthy server matching `--model`:

```bash
# 16-task smoke slice — good first run
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench smoke \
  --backend openai --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela --shards 4

# SWE-bench Verified full
MCODE_CONTEXT_WINDOW=262144 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 \
uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B --backend openai \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 20 --sampling multiturn --sampling-budget 2 --selection-attempts 3 \
  --timeout 300 --mem-limit 8g --pids-limit 512 \
  --on bluevela --shards 4 \
  --db research/$(date +%F)-swebench-verified/results.db

# Aider Polyglot, 225 tasks
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B --backend openai \
  --on bluevela --shards 4
```

Add `--json` to any bench command to get a strictly-monotonic event stream you can pipe to `jq`. Add `--fetch-db / --no-fetch-db` to control whether the DB is rsync'd back when the run ends.

`OPENAI_BASE_URL` and `OPENAI_API_KEY` (if set) take precedence over auto-resolution.

## 5. Watch / list / cancel running benches

The Rich Live dashboard fans out per shard while a `--shards N` run executes. From a separate shell:

```bash
uv run mcode bench list                 # all bench runs
uv run mcode bench list --json | jq '.[] | select(.status == "running")'
uv run mcode bench cancel <run-id>      # see "cancel semantics" below
uv run mcode launch status              # servers + runs
uv run mcode watch                      # combined live dashboard, refreshes every 2s
```

### Cancel semantics on Blue Vela

`bench cancel <run-id>` for a Blue Vela run SSHes to the login host and runs `kill -TERM -<pid>` (process group, works because the remote bench is started under `setsid`), then `kill -KILL -<pid>` after a 10s grace period, then verifies the process is actually gone with `kill -0`. If the process is still alive, the cancel is **rejected** and the record stays `running` so you can retry — the worst outcome is no silent leak.

If the kill misses subprocesses (rare), you can verify orphans manually:

```bash
ssh skula@login3.bluevela.rmf.ibm.com 'pgrep -af mcode|podman'
```

## 6. Stop the server and fetch results

```bash
uv run mcode launch stop <server-id>
# or
uv run mcode launch stop --all          # only your recorded servers; never bkill 0

uv run mcode launch refresh             # re-query state from LSF
```

Results DBs are rsync'd back to the local `--db` path automatically (unless you passed `--no-fetch-db`). To inspect:

```bash
uv run mcode results --db-dir research/<run> --benchmark swebench-live --time
uv run mcode report --db-dir research/<run> --benchmark swebench-live --out report.html
uv run mcode merge-shards --shards-glob 'research/<run>/results.db-shards/*/results-shard-*.db' --out merged.db
```

## Common environment variables

|Variable|Purpose|
|-|-|
|`MCODE_LAUNCH_CONFIG`|Override `launch.toml` path|
|`MCODE_LAUNCH_STATE`|Override the persistent state file path|
|`MCODE_CONTEXT_WINDOW`|LLM context window (forwarded to the remote bench)|
|`MCODE_MAX_NEW_TOKENS`|Max output tokens|
|`MCODE_REACT_TIMEOUT`|ReACT loop timeout in seconds|
|`MCODE_HARNESS_EXPERIMENTS`|Comma-separated experiment flags|
|`OPENAI_BASE_URL` / `OPENAI_API_KEY`|Override auto-resolved endpoint|

Full env-var list and every flag: [`COMMANDS.md`](COMMANDS.md).

## Troubleshooting

- `doctor bluevela` says "ssh reachable: Permission denied" — load your key with `ssh-add ~/.ssh/id_ed25519`. The Blue Vela host needs `IdentitiesOnly yes` in your `~/.ssh/config` to avoid "too many auth failures".
- "queue stayed in PEND past 3600s" — your queues are backlogged; edit `[bluevela].queue_order` in `launch.toml` or wait. `doctor bluevela --init` will rewrite queue_order from current cluster state.
- "vLLM did not become ready within 2400s" — usually a cold container pull; `tail -n 200 -f <log_path>` (the path is in `launch status`). For Qwen3.5/3.6, the per-job graphroot trades cross-run image caching for reliability.
- Cancel failed with "remote pid still alive" — VPN dropped mid-cancel, or the job already moved off the login node's view. Run the manual `ssh ... kill -KILL -<pid>` from the error message and retry `bench cancel`.
- The legacy `deploy/bluevela/*.sh` scripts still work as a fallback. See [`../deploy/bluevela/README.md`](../deploy/bluevela/README.md).
- `MCODE_DEBUG=1` disables the formatted error layout and dumps a raw traceback.

## Reference notes

- [`e2e-verification.md`](e2e-verification.md) — live-cluster verification of the launcher.
- [`bluevela-probe-findings.md`](bluevela-probe-findings.md) — Blue Vela cluster probe notes.
