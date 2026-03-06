# Blue Vela Deployment Design

## Goal

Run mcode benchmarks on IBM Blue Vela (768x H100 GPU cluster, LSF scheduler) starting with Qwen3.5-35B-A3B on HumanEval/MBPP, scaling to full suite.

## Architecture

Two-job pattern:

1. **vLLM server** - LSF job running `vllm/vllm-openai` in podman on 1 H100. Serves OpenAI-compatible API.
2. **Benchmark array job** - LSF array job (`bsub -J "bench[0-N]"`), each task runs mcode in a virtualenv with ProcessSandbox. Calls vLLM over the network. No GPU needed.
3. **Local fetch script** - rsync results from cluster to local Mac.

## File Layout

```
deploy/bluevela/
  env.sh              # shared config (model, queue, shard count)
  setup.sh            # one-time: clone repo, create venv, install deps
  start-vllm.sh       # bsub vLLM server job
  run-bench.sh        # bsub benchmark array job
  stop-vllm.sh        # bkill vLLM job
  fetch-results.sh    # local script to rsync results
  README.md           # usage
```

## Cluster Environment

- Login: `ssh skula@login3.bluevela.rmf.ibm.com`
- Home: `/u/skula/` (100TB)
- GPFS: `/gpfs/ess6000-1/` (5.7PB shared)
- Python 3.9 system, no conda
- Podman 5.2.2, no Docker
- LSF queues: normal, interactive, priority, preemptable
- 8x H100 80GB per node, 96 CPU cores, ~2TB RAM

## Inference

- vLLM in podman container with GPU passthrough
- Model: Qwen/Qwen3.5-35B-A3B (MoE, 3B active)
- HuggingFace cache: `/u/skula/.cache/huggingface/`

## Benchmark Execution

- mcode installed in virtualenv at `/u/skula/mcode/venv/`
- ProcessSandbox (no Docker needed)
- Sharded via LSF array jobs, each shard writes separate .db
- Results: `/u/skula/mcode/results/`

## Result Transfer

- `fetch-results.sh` runs locally, rsync over SSH to `results/`

## Cleanup

- Delete `deploy/k8s/` (OpenShift no longer used)
