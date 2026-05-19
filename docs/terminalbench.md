# Terminal-Bench 2.0

> **Experimental:** this integration has local Harbor smoke coverage only. Full Terminal-Bench runs, the mCode terminal agent's benchmark quality, and Blue Vela execution are not validated yet.

mCode runs Terminal-Bench 2.0 through Harbor, the official TB2 harness. Harbor owns task download, environments, verifier injection, rewards, concurrency, and trial logs. mCode owns CLI UX, SQLite import, artifact manifests, and the optional mCode terminal agent.

## Setup and smoke tests

Install Harbor outside the project dependency set:

```bash
uv tool install harbor
uv run mcode doctor terminal-bench
uv run mcode doctor terminal-bench --deep
```

The deep doctor runs one Harbor oracle task and can take several minutes while Harbor downloads metadata and pulls/builds an image.

Start with the oracle agent to validate Harbor, Docker, and mCode result import without spending model tokens:

```bash
uv run mcode bench terminal-bench \
  --agent oracle \
  --model unused \
  --limit 1 \
  --db experiments/results/terminal-bench-oracle.db
```

## mCode agent run

The default `mcode` agent is a custom Harbor external agent that runs mCode's terminal-mode ReACT loop against Harbor's task container:

```bash
uv run mcode bench terminal-bench \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --limit 5 \
  --n-concurrent 2 \
  --loop-budget 25 \
  --db experiments/results/terminal-bench.db
```

Use `--harbor-executable "uv run --with harbor harbor"` when Harbor must run from a project-aware environment, especially for the custom mCode agent.

The terminal agent is intentionally separate from the SWE-bench patch agent because TB2 tasks mutate container state rather than submit git diffs.

## Other Harbor agents and flags

Built-in Harbor agents still work through mCode and are imported into the mCode DB:

```bash
uv run mcode bench terminal-bench --agent claude-code --model anthropic/claude-opus-4-1 --limit 5
```

Useful flags:

|Flag|Use|
|-|-|
|`--agent`|`mcode`, `oracle`, or any Harbor-supported agent|
|`--dataset`|Harbor dataset id; defaults to `terminal-bench/terminal-bench-2`|
|`--limit`|Run the first N selected tasks|
|`--task-ids`|Comma-separated TB2 ids, e.g. `log-summary-date-ranges`|
|`--n-concurrent`|Harbor trial concurrency|
|`--env`|Harbor environment provider, usually `docker` locally|
|`--timeout-multiplier`|Scale task agent/verifier timeouts|
|`--jobs-dir`|Where Harbor writes job directories|
|`--artifact-dir`|Where mCode writes imported artifact manifests|
|`--harbor-arg`|Append a raw argument to `harbor run`|

## Results and artifacts

Harbor writes `jobs/<job-name>/result.json` plus one trial directory per task, each with its own `result.json`, logs, verifier output, and rewards. mCode imports every trial as a `terminal-bench` task result. The artifact manifest records evaluation metadata and paths back to the Harbor trial, verifier logs, and reward file.

## Notes

- Harbor currently requires Python 3.12+. Keeping it as a `uv tool` avoids dependency conflicts with mCode's optional dataset stack.
- Normal mCode patch replay does not apply because TB2 scores final container state.
- Full TB2 runs are expensive. Start with `--agent oracle --limit 1`, then a small model slice.
- Blue Vela support is not validated yet; treat it as incomplete until Harbor execution is tested end to end there.
