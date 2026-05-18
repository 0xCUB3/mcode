# Terminal-Bench 2.0

mCode runs Terminal-Bench 2.0 through Harbor, the official TB2 harness. Harbor owns task download, Docker/cloud environments, verifier injection, reward parsing, and trial logs. mCode owns the CLI, launch state, SQLite results, artifact import, and the optional mCode terminal agent.

## Setup

Install Harbor as a tool so it can keep its own Python/dependency set:

```bash
uv tool install harbor
uv run mcode doctor terminal-bench
```

Run the deep doctor when you want to prove Docker and the TB2 dataset work end to end:

```bash
uv run mcode doctor terminal-bench --deep
```

The deep check runs one Harbor oracle task and can take several minutes the first time because Harbor may download task metadata and pull/build a container image.

## Oracle smoke

Use Harbor's oracle agent first. This verifies the dataset, Docker, and result import path without spending model tokens.

```bash
uv run mcode bench terminal-bench \
  --agent oracle \
  --model unused \
  --limit 1 \
  --db experiments/results/terminal-bench-oracle.db
```

mCode imports Harbor's job output into the DB and stores per-task artifact manifests that point back to the Harbor trial directory.

## mCode agent run

The default agent is `mcode`. It is a custom Harbor external agent that runs mCode's terminal-mode ReACT loop against Harbor's task container.

```bash
uv run mcode bench terminal-bench \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --limit 5 \
  --n-concurrent 2 \
  --loop-budget 25 \
  --db experiments/results/terminal-bench.db
```

The mCode terminal agent uses tools for shell execution, listing, reading, writing, and string replacement. It is intentionally separate from the SWE-bench patch agent because Terminal-Bench tasks often require creating files or changing container state rather than producing a git diff.

## Built-in Harbor agents

You can still run Harbor's installed agents through the mCode command and import their results:

```bash
uv run mcode bench terminal-bench \
  --agent claude-code \
  --model anthropic/claude-opus-4-1 \
  --limit 5
```

Useful flags:

|Flag|Use|
|-|-|
|`--agent`|`mcode`, `oracle`, or any Harbor-supported agent such as `claude-code` or `codex`|
|`--dataset`|Harbor dataset id. Defaults to `terminal-bench/terminal-bench-2`|
|`--limit`|Run the first N selected tasks|
|`--task-ids`|Comma-separated TB2 ids such as `log-summary-date-ranges`|
|`--n-concurrent`|Harbor trial concurrency|
|`--env`|Harbor environment provider, usually `docker` locally|
|`--timeout-multiplier`|Scale task agent/verifier timeouts|
|`--jobs-dir`|Where Harbor writes job directories|
|`--artifact-dir`|Where mCode writes imported artifact manifests|
|`--harbor-arg`|Append a raw argument to `harbor run` for newer Harbor flags|

## Results and artifacts

Harbor writes a job like:

```text
jobs/<job-name>/
  result.json
  config.json
  <trial>/
    result.json
    trial.log
    agent/
    verifier/
      test-stdout.txt
      test-stderr.txt
      reward.txt or reward.json
    artifacts/
```

mCode imports each trial as a `terminal-bench` task result. The task `submission_json` includes Harbor metadata such as trial name, trial path, task checksum, and reward. The artifact manifest stores evaluation metadata and paths to the Harbor trial, verifier logs, and reward file.

## Notes

- Harbor currently requires Python 3.12+. Keeping it as a `uv tool` avoids dependency conflicts with mCode's optional dataset stack.
- Normal mCode patch replay does not apply to Terminal-Bench because the benchmark scores final container state, not a git patch.
- Full TB2 runs are expensive. Many tasks have 15-60 minute timeouts, and a few are longer. Start with `--agent oracle --limit 1`, then a small model slice.
