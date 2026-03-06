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

## Running a benchmark

### 1. Configure

Edit `env.sh` to set your run parameters:

```bash
MODEL=meta-llama/Llama-3.1-70B-Instruct
BENCHMARK=humaneval
SHARD_COUNT=4
```

### 2. Start vLLM server

```bash
./start-vllm.sh
```

Submits the vLLM serving job to LSF. Wait for it to report ready before proceeding.

### 3. Run the benchmark

```bash
./run-bench.sh
```

Submits a sharded benchmark array job (one task per shard).

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
| `BENCHMARK` | Benchmark name | `humaneval` |
| `SHARD_COUNT` | Number of parallel shards | `4` |
| `VLLM_PORT` | Port for vLLM server | `8000` |
| `VLLM_GPUS` | GPUs allocated to vLLM | `8` |
| `MAX_MODEL_LEN` | Max sequence length | `4096` |
| `RESULTS_DIR` | Where results are written | `./results` |
