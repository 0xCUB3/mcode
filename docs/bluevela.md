# Blue Vela workflow

Blue Vela is where mCode is meant to run the bigger local-model experiments. The model server runs as a vLLM job under LSF. The benchmark can run on the cluster too, then rsync its SQLite DB and artifacts back to your laptop when it finishes.

The workflow has a few moving parts, but the loop is simple once the config exists: check the cluster, sync the repo, launch vLLM, run the bench with `--on bluevela`, inspect the run, and stop the server.

The login host used in these examples is `<user>@login3.bluevela.rmf.ibm.com`. You need VPN access, SSH access to the login node, and membership in `grp_runtime`.

## Local installation

Run these from your local checkout:

```bash
uv sync --extra dev
uv run mcode --version
uv run mcode deps sync --extra swebench --extra datasets
```

The commands that submit jobs and read state run locally. The benchmark code is copied to the cluster by `launch sync`, so keep your local checkout in the state you want to run before syncing.

## Cluster configuration

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


## Launch config schema

`doctor bluevela --init` writes the config for you, but it helps to know what the file means. The default path is `~/.config/mcode/launch.toml`.

```toml
[bluevela]
login = "<user>@login3.bluevela.rmf.ibm.com"
workspace_root = "$HOME/mcode-launch"
shared_root = "$HOME/mcode-shared"
queue_order = ["normal"]
group = "grp_runtime"
gpu_mode = "exclusive_process"
hf_env = "$HOME/.config/mcode/hf-env.sh"

[bluevela.podman]
# graphroot_base = "$HOME/.local/share/mcode-podman-graphroot"
# runroot_base = "$HOME/.local/share/mcode-podman-runroot"

[local_vllm]
port = 8000

[local_ollama]
host = "127.0.0.1"
port = 11434
```

`workspace_root` is the synced checkout. `shared_root` is runtime storage for launcher-owned files, including server run directories, bench run directories, logs, and fetched metadata. Do not put hand-maintained files inside launcher-owned run directories.

`queue_order` is tried in order. `doctor bluevela --init` fills it from the queues visible to your account, and rerunning init can refresh it when the cluster changes. `group` becomes the `bsub -G` value. `gpu_mode` is usually `exclusive_process` unless the cluster policy says otherwise.

`hf_env` points to a shell file sourced by the remote launch script. Put Hugging Face exports there for gated models:

```bash
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN=...
```

Keep that file on the cluster and protect it like any other token file.

## Repository sync

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

## vLLM launch

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


## Logs and launch status

`launch status` is the fastest way to find a server id, endpoint, job id, status, and log path:

```bash
uv run mcode launch status
uv run mcode launch status --json --raw | jq '.'
```

`--raw` includes raw LSF state in JSON. When a launch is stuck or a health check times out, use the id from status with `launch logs`:

```bash
uv run mcode launch logs <server-id>
```

For cluster-side checks, the normal LSF tools are still useful:

```bash
ssh <user>@login3.bluevela.rmf.ibm.com 'bjobs -u $USER'
ssh <user>@login3.bluevela.rmf.ibm.com 'bqueues'
```

If mCode says your group or queue is invalid, fix `launch.toml` or rerun `doctor bluevela --init` so it can probe the current cluster view.

## Smoke benchmark

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

## Aider Polyglot and mixed suite

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


## Chunked long runs

For runs that may outlive one vLLM server allocation, use chunks. Each chunk runs a slice, writes a chunk DB, and merges into the requested DB. With `--relaunch-vllm`, mCode can start a fresh server when no healthy matching server exists between chunks.

```bash
MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench swebench-lite \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset princeton-nlp/SWE-bench_Verified \
  --on bluevela \
  --chunk-size 25 \
  --relaunch-vllm \
  --vllm-tensor-parallel 4 \
  --vllm-max-model-len 262144 \
  --db research/verified-chunked/results.db
```

Chunking is not needed for short smoke runs. It is an operational tool for full sweeps on a busy cluster.

## Remote resume and fetch behavior

Remote bench directories are stable for the same model, local DB path, bench arguments, and forwarded `MCODE_*` context variables. If the SSH stream dies after the remote process finished, rerun the same command. mCode should find the existing remote run metadata and fetch the DB instead of starting over.

The default is `--fetch-db`, which rsyncs the SQLite DB back when the run ends. Use `--no-fetch-db` only if you have a reason to leave the DB on the cluster. Artifact fetch is off by default because artifacts can be much larger. Pass `--fetch-artifacts` when you know you want local replay.

For very long SWE-bench jobs, `--chunk-size N` runs remote chunks sequentially, writes chunk DBs, and merges them. If you also pass `--relaunch-vllm`, mCode can launch a fresh vLLM server between chunks when no healthy server exists. That path is meant for long cluster runs where model servers may time out before the full bench is finished.


## Remote runtime details

The remote bench command runs in the synced workspace but writes runtime files under the configured shared root. The plan records the remote DB path, artifact path, log path, process id, and fetch destinations in launch state. `bench show <run-id>` is the easiest way to see those paths after the fact.

For SWE-bench evaluation on Blue Vela, the remote script points `DOCKER_HOST` at a per-run Podman socket. It also sets `MCODE_PODMAN_LOCK_DIR` so image operations can coordinate without trampling one another. If your account needs custom Podman storage, set `[bluevela.podman].graphroot_base` or `runroot_base` in `launch.toml` instead of editing the generated remote scripts.

A stable remote bench directory is derived from the model, local DB path, bench arguments, and forwarded context env. That is why rerunning the same command can fetch an already-finished DB instead of starting new work. Changing the DB path or important arguments means you are asking for a different remote run.

## Run monitoring

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

## Remote cancellation

Cancel by run id:

```bash
uv run mcode bench cancel <run-id>
```

For Blue Vela runs, mCode SSHes to the login host, sends `TERM` to the remote process group, waits briefly, sends `KILL` if needed, and then checks that the process is gone. If that final check fails, the run record stays `running` and the command returns an error. That is intentional. A false cancelled state is worse than an annoying retry.

If you suspect a leaked process, check manually:

```bash
ssh <user>@login3.bluevela.rmf.ibm.com 'pgrep -af "mcode|podman|vllm"'
```


## Artifact fetch failures

Artifact fetch can fail even when the benchmark succeeded. The usual causes are a dropped SSH session, a missing remote artifact directory, a full local disk, or a destination path you cannot write. Start with `bench show <run-id>` and check the remote artifact path it prints.

Then fetch explicitly:

```bash
uv run mcode bench artifacts fetch <run-id> --dest research/<run>/artifacts
```

If you do not want to look up the run id, resolve from the DB:

```bash
uv run mcode bench artifacts fetch --db research/<run>/results.db
```

Use `--json` when a script needs the exact source and destination paths.

## Server shutdown and state refresh

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

## Results inspection

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

## Environment variables

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


## Safe cleanup

Use the mCode commands first. `bench cancel` is for a running benchmark. `launch stop` is for the vLLM server. `launch refresh` asks LSF for current status and rewrites local state. Manual SSH checks are for suspected leaks, not the normal path.

```bash
uv run mcode bench cancel <run-id>
uv run mcode launch stop <server-id>
uv run mcode launch refresh
ssh <user>@login3.bluevela.rmf.ibm.com 'pgrep -af "mcode|podman|vllm"'
```

Avoid deleting the synced workspace or shared run directories while a run is still recorded as running. If you clean remote files by hand, expect `bench show`, artifact fetch, and resume to lose information.

## Troubleshooting

If `doctor bluevela` reports `Permission denied`, load the SSH key you expect with `ssh-add` and check `IdentitiesOnly yes` for the Blue Vela host. The login node can reject you after too many offered keys.

If the queue stays in `PEND` for a long time, the queues in `launch.toml` may be backlogged. You can wait, edit `[bluevela].queue_order`, or rerun `doctor bluevela --init` to rewrite the queue order from what the cluster currently exposes.

If vLLM does not become ready before the launch timeout, check the log path from `mcode launch status`. Cold image pulls and model loads can look dead until the server writes `vllm_host.txt` and answers `/v1/models`.

If cancel fails with a message about the remote pid still being alive, do not mark the run stopped by hand. Retry once after checking VPN, then use the manual SSH command printed in the error if the process really needs to be killed.

If a DB fetch fails after the remote run completed, rerun the same bench command or use `mcode bench show <run-id>` to find the remote DB path. Do not start a new run until you have checked whether the old one already finished.
