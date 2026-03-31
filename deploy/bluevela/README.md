# Blue Vela Deployment

## Prerequisites

- SSH access to `login3.bluevela.rmf.ibm.com`
- Connected to IBM VPN

## One-time setup

SSH into the cluster and run:

```bash
./setup.sh
```

This installs dependencies and configures the environment.

## Running SWE-bench Live

### 1. Configure

Edit `env.sh` to set your run parameters. For Hugging Face auth and shared cache, the launchers now automatically source `/u/skula/.config/mcode/hf-env.sh` when it exists.

```bash
MODEL=meta-llama/Llama-3.1-70B-Instruct
SHARD_COUNT=4
SWB_SPLIT=verified
```

### 2. Start vLLM server

```bash
./start-vllm.sh
```

Submits the vLLM serving job to LSF. Wait for it to report ready before proceeding.

### 3. Run the benchmark

```bash
./run-swebench-live.sh
```

Submits a sharded SWE-bench Live array job.

### 4. Monitor

```bash
bjobs
```

### 5. Stop vLLM when done

```bash
./stop-vllm.sh
```

## Fetching results

Run locally (not on the cluster):

```bash
./fetch-results.sh
```

Uses rsync to pull results from the cluster to your local machine.

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
| `BV_RESULTS_DIR` | Remote results directory | `/u/skula/mcode/results` |
| `VLLM_PORT` | Port for vLLM server | `8321` |
| `VLLM_GPU_COUNT` | GPUs allocated to vLLM | `1` |
| `VLLM_MAX_MODEL_LEN` | Max sequence length | `32768` |


The benchmark and vLLM launchers will reuse the shared Hugging Face cache under `/proj/dmfexp/skula/hf-cache`. If you need to override the secret/bootstrap path, set `BV_HF_ENV` before invoking the scripts.