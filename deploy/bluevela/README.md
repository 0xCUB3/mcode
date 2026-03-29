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

Edit `env.sh` to set your run parameters:

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
| `VLLM_PORT` | Port for vLLM server | `8321` |
| `VLLM_GPU_COUNT` | GPUs allocated to vLLM | `1` |
| `VLLM_MAX_MODEL_LEN` | Max sequence length | `32768` |
