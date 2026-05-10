# Running mCode on Blue Vela

Blue Vela is where mCode is meant to run the bigger local-model experiments. The model server runs as a vLLM job under LSF. The benchmark can run on the cluster too, then rsync its SQLite DB and artifacts back to your laptop when it finishes.

The workflow has a few moving parts, but the loop is simple once the config exists: check the cluster, sync the repo, launch vLLM, run the bench with `--on bluevela`, inspect the run, and stop the server.

The login host used in these examples is `<user>@login3.bluevela.rmf.ibm.com`. You need VPN access, SSH access to the login node, and membership in `grp_runtime`.

## Install locally first

Run these from your local checkout:

```bash
uv sync --extra dev
uv run mcode --version
uv run mcode deps sync --extra swebench --extra datasets
```

The commands that submit jobs and read state run locally. The benchmark code is copied to the cluster by `launch sync`, so keep your local checkout in the state you want to run before syncing.

## Bootstrap the cluster config

The first useful command is the Blue Vela doctor with `--init`:

```bash
uv run mcode doctor bluevela --init --login <user>@login3.bluevela.rmf.ibm.com
```

That command SSHes to the login node, checks your group membership and visible queues, and writes `~/.config/mcode/launch.toml` with the values mCode needs later. If you keep configs somewhere else, set `MCODE_LAUNCH_CONFIG`.

After bootstrap, run the doctor again without `--init`:

```bash
uv run mcode doctor bluevela
```

The doctor output is meant to be followed literally. Red rows include a `next:` line. If SSH fails with too many authentication attempts, add an entry like this to `~/.ssh/config`:

```sshconfig
Host login3.bluevela.rmf.ibm.com
  User <user>
  IdentitiesOnly yes
  IdentityFile ~/.ssh/id_ed25519
```

## Sync the repo

mCode runs from a checkout on the cluster shared filesystem. Push your local checkout with:

```bash
uv run mcode launch sync bluevela --dry-run
uv run mcode launch sync bluevela
```

The dry run is worth doing when you have uncommitted files or big local artifacts lying around. The sync respects `.gitignore` and excludes the usual junk: `.git/`, virtual environments, caches, and launcher-owned remote directories such as `runs/`, `bench-runs/`, and `benchmarks/`.

There is a safety marker in the remote workspace. mCode will not run `rsync --delete` into a non-empty unmarked directory. If you are deliberately claiming an existing remote directory for the first time, use:

```bash
uv run mcode launch sync bluevela --bootstrap
```

Treat `--bootstrap` with care. It tells mCode that the remote directory is the mirror target and that deletes are allowed on future syncs.

## Launch vLLM

Start a server job:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json
```

The command chooses a queue from your config, submits an LSF job under `grp_runtime`, uploads the vLLM script and environment file, and waits until the server is healthy. The launch UI moves through submit, queued, starting, and ready. The starting phase includes container setup, model load, and the first health check, so it can be quiet for a while on a cold queue.

You can split launch and wait if you want the shell back immediately:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B --json &
uv run mcode launch wait <server-id> --timeout 1800
```

The profile table lives in `src/mcode/launch/profiles.py`. The built-in profiles cover the Qwen, Gemma, Granite, and MiniMax models used by the research notes. If a profile default is close but not quite right, override tensor parallel or context length at launch time:

```bash
uv run mcode launch bluevela \
  --model Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel 4 \
  --max-model-len 262144
```

## Run a smoke bench on the cluster

Once a server is healthy, run the smoke slice remotely:

```bash
MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --shards 4 \
  --db research/$(date +%F)-bluevela-smoke/results.db
```

`--on bluevela` sends the benchmark to the remote workspace and runs it under `setsid`, which is what lets `bench cancel` kill the whole remote process group later. `--backend openai` asks mCode to find a healthy recorded server whose model matches `--model`. If you set `OPENAI_BASE_URL` yourself, that value wins.

The local command streams the remote output, records the run in launch state, and fetches the DB back to the local `--db` path by default when the run ends.

## Full SWE-bench Verified example

This is the shape of the Qwen3.6 Verified runs in the research notes. Adjust the DB path and maybe the shard count before copying it into a shell.

```bash
MCODE_CONTEXT_WINDOW=262144 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=2400 \
uv run mcode bench swebench-lite \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 20 \
  --sampling multiturn \
  --sampling-budget 2 \
  --selection-attempts 3 \
  --eval-repair-attempts 0 \
  --timeout 300 \
  --mem-limit 8g \
  --pids-limit 512 \
  --cpu-limit 32 \
  --on bluevela \
  --shards 4 \
  --db research/$(date +%F)-swebench-verified/results.db
```

The `--cpu-limit` flag matters on shared login-style machines where runaway eval containers can get killed by administrators. It caps each SWE-bench eval container at a fixed number of cores. If your queue policy changes, revisit that value.

## Aider Polyglot and the mixed suite

Aider Polyglot is shorter to launch:

```bash
uv run mcode bench aider-polyglot \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --shards 4 \
  --db research/$(date +%F)-aider-polyglot/results.db
```

The mixed suite is useful for harness changes because it exercises more than one adapter through the same runner. I often run generation first, then evaluate the saved artifacts:

```bash
uv run mcode bench suite \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --shards 4 \
  --phase generate \
  --artifact-dir research/mixed-suite/artifacts \
  --db research/mixed-suite/generate.db

uv run mcode bench suite \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --shards 4 \
  --phase evaluate \
  --artifact-dir research/mixed-suite/artifacts \
  --db research/mixed-suite/evaluate.db
```

Add `--fetch-artifacts` if you want the artifact directory copied back at the end of the remote run. If you skipped it, you can fetch later from the run record.

```bash
uv run mcode bench artifacts fetch <run-id> --dest research/mixed-suite/artifacts
uv run mcode bench artifacts fetch --db research/mixed-suite/generate.db
```

## Remote resume and fetch behavior

Remote bench directories are stable for the same model, local DB path, bench arguments, and forwarded `MCODE_*` context variables. If the SSH stream dies after the remote process finished, rerun the same command. mCode should find the existing remote run metadata and fetch the DB instead of starting over.

The default is `--fetch-db`, which rsyncs the SQLite DB back when the run ends. Use `--no-fetch-db` only if you have a reason to leave the DB on the cluster. Artifact fetch is off by default because artifacts can be much larger. Pass `--fetch-artifacts` when you know you want local replay.

For very long SWE-bench jobs, `--chunk-size N` runs remote chunks sequentially, writes chunk DBs, and merges them. If you also pass `--relaunch-vllm`, mCode can launch a fresh vLLM server between chunks when no healthy server exists. That path is meant for long cluster runs where model servers may time out before the full bench is finished.

## Watching, listing, and showing runs

From another shell:

```bash
uv run mcode bench list
uv run mcode bench list --wide
uv run mcode bench list --json | jq '.[] | select(.status == "running")'
uv run mcode bench show --latest
uv run mcode launch status
uv run mcode watch
```

`bench list` is intentionally compact by default. Use `--wide` when you need remote paths, DB paths, artifact status, and target columns. `bench show` is the detailed view. It shows DB summaries when the DB has been fetched, failed tasks, artifact fetch commands, rerun metadata, and remote process information.

If the state file gets noisy after experiments, prune it locally:

```bash
uv run mcode bench prune --status failed --older-than 7d
uv run mcode bench prune --status failed --older-than 7d --yes
```

That first command is a dry run. I always run the dry run first.

## Cancelling a remote bench

Cancel by run id:

```bash
uv run mcode bench cancel <run-id>
```

For Blue Vela runs, mCode SSHes to the login host, sends `TERM` to the remote process group, waits briefly, sends `KILL` if needed, and then checks that the process is gone. If that final check fails, the run record stays `running` and the command returns an error. That is intentional. A false cancelled state is worse than an annoying retry.

If you suspect a leaked process, check manually:

```bash
ssh <user>@login3.bluevela.rmf.ibm.com 'pgrep -af "mcode|podman|vllm"'
```

## Stopping vLLM and refreshing state

Stop the server when you are done:

```bash
uv run mcode launch stop <server-id>
```

Or stop every server recorded in your local launch state:

```bash
uv run mcode launch stop --all
```

`stop --all` does not issue a blanket `bkill` for your user. It only stops jobs that mCode recorded. If the state looks stale, refresh it from LSF:

```bash
uv run mcode launch refresh
```

## Inspecting results

The DB should be local after a normal remote run. Query it directly:

```bash
uv run mcode results --db research/<run>/results.db --time
uv run mcode results --db-dir research/<run> --benchmark swebench-lite --time
```

If you need to merge shard DBs by hand, use the bench subcommand:

```bash
uv run mcode bench merge-shards \
  --out research/<run>/merged.db \
  research/<run>/results.db-shards/*/results-shard-*.db
```

For analysis outside the CLI:

```bash
uv run mcode export-csv \
  -i research/<run> \
  --out-dir research/<run> \
  --prefix mcode
```

## Environment variables I actually use

|Variable|Why it matters on Blue Vela|
|-|-|
|`MCODE_LAUNCH_CONFIG`|Use a launch config outside `~/.config/mcode/launch.toml`|
|`MCODE_LAUNCH_STATE`|Use a separate state file for a risky experiment|
|`MCODE_CONTEXT_WINDOW`|Forward the model context size into the remote bench|
|`MCODE_MAX_NEW_TOKENS`|Cap model output tokens|
|`MCODE_REACT_TIMEOUT`|Give long ReACT loops enough time before timeout|
|`OPENAI_BASE_URL`|Bypass launch-state endpoint discovery|
|`OPENAI_API_KEY`|Set an API token for an external OpenAI-compatible server|
|`MCODE_DEBUG`|Show raw tracebacks instead of the formatted error page|

## Common Blue Vela problems

If `doctor bluevela` reports `Permission denied`, load the SSH key you expect with `ssh-add` and check `IdentitiesOnly yes` for the Blue Vela host. The login node can reject you after too many offered keys.

If the queue stays in `PEND` for a long time, the queues in `launch.toml` may be backlogged. You can wait, edit `[bluevela].queue_order`, or rerun `doctor bluevela --init` to rewrite the queue order from what the cluster currently exposes.

If vLLM does not become ready before the launch timeout, check the log path from `mcode launch status`. Cold image pulls and model loads can look dead until the server writes `vllm_host.txt` and answers `/v1/models`.

If cancel fails with a message about the remote pid still being alive, do not mark the run stopped by hand. Retry once after checking VPN, then use the manual SSH command printed in the error if the process really needs to be killed.

If a DB fetch fails after the remote run completed, rerun the same bench command or use `mcode bench show <run-id>` to find the remote DB path. Do not start a new run until you have checked whether the old one already finished.
