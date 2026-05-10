# Local workflow

This is the path I use when I want a fast answer from a model I can run on my own machine. Ollama is the easiest route because it already exposes an OpenAI-compatible endpoint. Local vLLM is better when you want to match the server stack used on Blue Vela or tune vLLM itself.

The examples below assume you are in the repo and using `uv run`. If you installed the package another way, drop the `uv run` prefix.

## Prerequisites

You need Python 3.11 or newer, `uv`, a container runtime for SWE-bench evaluation, and either Ollama or vLLM. Docker Desktop is fine for local SWE-bench. Podman can work too, but most of the day-to-day local testing here has been with Docker.

Start with the lightweight install:

```bash
uv sync --extra dev
uv run mcode --version
```

Then run the doctor for the target you actually care about. I usually do not run the targetless doctor on an Ollama-only laptop because it also checks Blue Vela and local vLLM.

```bash
uv run mcode doctor local-ollama
# or
uv run mcode doctor local-vllm
```

If you plan to run SWE-bench or anything that reads Hugging Face datasets, install those extras now:

```bash
uv run mcode deps sync --extra swebench --extra datasets
```

For Aider Polyglot, check the language runtimes before you launch a long job. This catches missing Go, Rust, Java, C++, and Node toolchains before the agent wastes task budget on `command not found`.

```bash
uv run mcode deps toolchains --benchmark aider-polyglot
```

## Model server setup

### Ollama

Make sure the Ollama daemon is running. On macOS this may already be handled by the app. From a shell, this is enough:

```bash
ollama serve
```

Launch the model through mCode so the server is recorded in launch state:

```bash
uv run mcode launch local-ollama --model granite4
uv run mcode launch wait <server-id> --timeout 120
```

Use the exact model name from `ollama list`. If the model is listed as `qwen3.6:35b-a3b`, use that string everywhere:

```bash
uv run mcode launch local-ollama --model qwen3.6:35b-a3b
uv run mcode launch wait <server-id> --timeout 120
```

`launch local-ollama` records the endpoint and model name so `--backend openai --model qwen3.6:35b-a3b` can find it later. You can still bypass that discovery by setting `OPENAI_BASE_URL` yourself.

### Local vLLM

Local vLLM is the same idea, but mCode starts the vLLM process for you.

```bash
uv run mcode launch local-vllm --model Qwen/Qwen2.5-0.5B
uv run mcode launch wait <server-id> --timeout 600
```

Use this when you want vLLM behavior locally, or when you are debugging a model profile before trying the same model on Blue Vela.

## Smoke test

Run the smoke slice first. It is a small SWE-bench Verified slice that exercises the agent loop, patch handling, container evaluation, result DB writes, sharding, and run state.

```bash
uv run mcode bench smoke --backend openai --model granite4 --shards 4
```

For a larger local model, I usually set the context and output limits explicitly so a later shell history search tells me what I meant to test:

```bash
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=420 \
uv run mcode bench smoke \
  --backend openai --model qwen3.6:35b-a3b \
  --shards 1 \
  --db /tmp/mcode-local-smoke.db
```

The human output prints a run plan before work begins, then live task progress while the model is turning over a problem. SWE-bench tasks also show the boring but useful phases: preparing repo, generating patch, running official evaluation, and official evaluation ok or fail. At the end, mCode prints a short footer with pass count, task time, and the DB path.

If you prefer logs that another program can read, add `--json`:

```bash
uv run mcode bench smoke --backend openai --model granite4 --shards 4 --json | jq -c '.'
```

The JSON stream stays compact by default. If you want live model/tool trace events in JSON too, set `MCODE_LIVE_TRACE=1`. If the human trace is too noisy, set `MCODE_LIVE_TRACE=0`.

## Benchmark examples

SWE-bench Lite with the first 16 tasks:

```bash
uv run mcode bench swebench-lite \
  --backend openai --model granite4 \
  --limit 16 \
  --shards 4 \
  --db experiments/results/lite-16.db
```

Aider Polyglot:

```bash
uv run mcode bench aider-polyglot \
  --backend openai --model granite4 \
  --db experiments/results/polyglot.db
```

A single Aider Polyglot task is useful when you are debugging the control loop:

```bash
uv run mcode bench aider-polyglot \
  --backend openai --model granite4 \
  --language python \
  --exercise proverb \
  --loop-budget 4
```

The mixed suite runs several small slices through the same runner. I use it for harness changes because it is less SWE-only than the smoke command.

```bash
uv run mcode bench suite \
  --backend openai --model granite4 \
  --db experiments/results/mixed-suite.db
```

## Resume and sharding behavior

A benchmark resumes when you rerun the same command against the same DB. Finished task rows are skipped, retryable infrastructure failures are retried, and sharded runs reuse their shard DBs under `<db-stem>-shards/` before merging completed rows back into the main DB.

`--shards N` starts N local worker processes. That is usually what you want for SWE-bench smoke and small Lite runs. A single non-sharded run executes in the current process, so it cannot be cancelled from another shell. Use Ctrl+C in the terminal that started it.

If a `--task-ids` filter matches nothing, mCode now stops before launching the run and tells you exactly which filter failed. That is much better than creating an empty DB and pretending everything was fine.

## Split generation and evaluation

Sometimes you want the model to generate patches now and run official evaluation later. Use `--phase generate` with an explicit artifact directory, then run `--phase evaluate` against the same artifacts.

```bash
uv run mcode bench swebench-lite \
  --backend openai --model granite4 \
  --limit 16 \
  --db experiments/results/lite-split-generate.db \
  --artifact-dir experiments/results/lite-split-artifacts \
  --phase generate

uv run mcode bench swebench-lite \
  --backend openai --model granite4 \
  --limit 16 \
  --db experiments/results/lite-split-evaluate.db \
  --artifact-dir experiments/results/lite-split-artifacts \
  --phase evaluate
```

The default `--phase run` does both steps in one command and still writes artifacts as it goes.

## Run inspection

`bench list` reads the persistent launch state file. The compact id in the first column is accepted by `bench show` as long as it matches one run.

```bash
uv run mcode bench list
uv run mcode bench list --wide
uv run mcode bench show --latest
uv run mcode bench show <run-id>
```

`bench show` is the command I use most after a run. It prints the run metadata, the resolved DB path, a DB summary when the DB exists, failed task rows, remote paths if the run came from Blue Vela, and follow-up commands when they make sense.

If your state file has old failed experiments in it, prune it safely:

```bash
uv run mcode bench prune --status failed --older-than 7d
uv run mcode bench prune --status failed --older-than 7d --yes
```

The prune command is a dry run unless you pass `--yes`. By default it only removes records whose DB path is missing. Add `--any-db` only when you really mean to prune records even though the DB still exists.

## Artifacts

Generated artifacts let you inspect and replay what the model produced without rerunning generation. The most useful commands are:

```bash
uv run mcode bench artifacts list --db experiments/results/lite-split-evaluate.db
uv run mcode bench artifacts show <task-id> --db experiments/results/lite-split-evaluate.db
uv run mcode bench artifacts patch <task-id> --db experiments/results/lite-split-evaluate.db --out candidate.patch
uv run mcode bench artifacts replay <task-id> --db experiments/results/lite-split-generate.db
```

`artifacts list` gives you the inventory. `artifacts show` dumps the saved manifest for a task, or one candidate if you pass `--candidate-index`. `artifacts patch` is the quick way to pull out the selected diff. `artifacts replay` takes a saved candidate and runs evaluation again into a fresh DB.

## Server shutdown

Stop one server:

```bash
uv run mcode launch stop <server-id>
```

Stop every server mCode has recorded for you:

```bash
uv run mcode launch stop --all
```

`stop --all` only acts on recorded servers. It does not run a blanket kill command against the machine.

## Results and comparisons

For a quick pass-rate view:

```bash
uv run mcode results --db experiments/results/lite-16.db
uv run mcode results --db-dir experiments/results --benchmark swebench-lite --time
```

For a regression gate:

```bash
uv run mcode compare \
  --baseline-dir experiments/results/baseline.db \
  --candidate-dir experiments/results/candidate.db \
  --max-lost 0
```

CSV export is still handy for notebooks and spreadsheets:

```bash
uv run mcode export-csv \
  -i experiments/results \
  --out-dir experiments/results \
  --prefix mcode
```

## Troubleshooting

If Docker is not running, SWE-bench will fail before official evaluation. Start Docker Desktop or set `DOCKER_HOST` for your daemon and rerun the task. The common error mentions the Docker socket path and says the daemon is not reachable.

If the first SWE-bench run appears to sit at image pulling, give it a few minutes. The first pull can be slow. If you already have the images locally, `MCODE_SKIP_IMAGE_PULL=1` skips the pre-pull step.

If mCode says there is no healthy server for a model, run `mcode launch status`. Either the model was launched under a different name, the server failed, or you meant to use an endpoint outside the launch registry. In that last case, set `OPENAI_BASE_URL` and `OPENAI_API_KEY` directly.

If you need a clean retry, stop recorded servers and start fresh:

```bash
uv run mcode launch stop --all
uv run mcode launch local-ollama --model <model>
```

If the formatted error page hides something you need, rerun with `MCODE_DEBUG=1` to get the raw traceback.
