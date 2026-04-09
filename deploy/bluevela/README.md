# Blue Vela Deployment

The preferred path is now the unified launcher:

```bash
uv run mcode launch
```

For the full operational command set, including `status`, `attach`, `fetch`, and `stop`, use [`docs/COMMANDS.md`](/Users/skula/Documents/mcode/docs/COMMANDS.md).

Recommended first checks:

```bash
uv run mcode launch doctor --target bluevela
uv run mcode launch sync --target bluevela --check
```

The shell scripts in this directory are still supported during the transition. They are for debugging and legacy flows. New run docs should prefer `mcode launch` commands over these scripts.

## Prerequisites

- SSH access to `login3.bluevela.rmf.ibm.com`
- Connected to IBM VPN

## One-time setup

Preferred:

```bash
uv run mcode deps sync --no-dev --extra swebench --extra datasets
```

Legacy:

```bash
./setup.sh
```

## Running SWE-bench Live

Recommended:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-live \
  --split verified \
  --parallelism 4 \
  --yes
```

The launcher handles:

- config defaults
- code sync
- vLLM server reuse or startup
- benchmark launch
- state tracking for attach, fetch, and stop

You can override defaults in:

```bash
~/.config/mcode/launch.toml
```

Example:

```toml
[bluevela]
login = "your-user@login3.bluevela.rmf.ibm.com"
workspace_root = "/u/your-user/mcode-launch"
shared_root = "/proj/dmfexp/your-user"
hf_env = "/u/your-user/.config/mcode/hf-env.sh"
```

Legacy scripted flow:

### 1. Configure

Edit `env.sh` to set your run parameters. For Hugging Face auth and shared cache, the launchers automatically source the path configured in `~/.config/mcode/launch.toml`, which defaults to `/u/$USER/.config/mcode/hf-env.sh` when it exists.

```bash
MODEL=meta-llama/Llama-3.1-70B-Instruct
SHARD_COUNT=4
SWB_SPLIT=verified
```

### 2. Start vLLM server

```bash
./start-vllm.sh
```

Preferred replacement:

```bash
uv run mcode launch --target bluevela --model Qwen/Qwen3.5-27B --benchmark swebench-live --parallelism 4 --yes
```

### 3. Run the benchmark

```bash
./run-swebench-live.sh
```

Preferred replacement:

```bash
uv run mcode launch --target bluevela --model Qwen/Qwen3.5-27B --benchmark swebench-live --parallelism 4 --yes
```

### 4. Monitor

```bash
uv run mcode launch status --json
```

### 5. Stop vLLM when done

```bash
./stop-vllm.sh
```

Preferred replacement:

```bash
uv run mcode launch stop server-12345678
```

## Fetching results

Run locally (not on the cluster):

```bash
./fetch-results.sh
```

Preferred replacement:

```bash
uv run mcode launch fetch run-12345678 --destination results
```

## Common LSF commands

| Command | Description |
|-|-|
| `bjobs` | List running/pending jobs |
| `bjobs -l <jobid>` | Detailed job info |
| `bkill <jobid>` | Kill a job |
| `bkill 0` | Kill all your jobs |
| `bpeek <jobid>` | View stdout/stderr of a running job |
| `bhist <jobid>` | Job history and resource usage |

## env.sh configuration reference

| Variable | Description | Example |
|-|-|-|
| `MODEL` | HuggingFace model ID | `meta-llama/Llama-3.1-70B-Instruct` |
| `SHARD_COUNT` | Number of parallel shards | `4` |
| `SWB_SPLIT` | SWE-bench Live split | `verified` |
| `SWB_TIMEOUT` | Timeout per task | `1800` |
| `BV_RESULTS_DIR` | Remote results directory | `/u/$USER/mcode/results` |
| `VLLM_PORT` | Port for vLLM server | `8321` |
| `VLLM_GPU_COUNT` | GPUs allocated to vLLM | `1` |
| `VLLM_MAX_MODEL_LEN` | Max sequence length | `32768` |


The benchmark and vLLM launchers will reuse the shared Hugging Face cache under `/proj/dmfexp/$USER/hf-cache` by default. If you need to override the secret/bootstrap path, set `BV_HF_ENV` before invoking the scripts.
