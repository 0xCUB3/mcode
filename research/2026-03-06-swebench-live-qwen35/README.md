# SWE-bench Live Lite: Qwen3.5-35B-A3B

## Setup

- **Model:** Qwen/Qwen3.5-35B-A3B (MoE, 3B active params)
- **Serving:** vLLM with `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`
- **Hardware:** IBM Blue Vela cluster, 2x H100 80GB (tensor-parallel)
- **Agent:** mcode ReACT loop, 3 tool-call budget per task
- **Strategy:** repair (search_code, read_file, apply_edit, final_answer)
- **Split:** SWE-bench Live Lite (300 instances)
- **Evaluation:** Docker containers via podman Docker-compat API

## Results

| Metric | Value |
|-|-|
| Total tasks | 300 |
| Resolved | 6 (2.0%) |
| Not resolved | 191 |
| Infrastructure errors | 109 |

### Resolved tasks

| Task | Time |
|-|-|
| pvlib__pvlib-python-2249 | 48s |
| pvlib__pvlib-python-2393 | 27s |
| pypa__twine-1225 | 55s |
| python-control__python-control-1111 | 38s |
| matplotlib__matplotlib-29721 | 115s |
| run-llama__llama_deploy-397 | 61s |

### Infrastructure errors (not agent failures)

| Error | Count | Cause |
|-|-|-|
| ImageNotFound | 46 | Docker Hub image missing for instance |
| KeyError | 42 | mellea upstream bug (intermittent) |
| JSONDecodeError | 13 | Malformed vLLM response |
| ReadTimeout | 7 | Podman API timeout (>600s) |
| APIError | 1 | vLLM server error |

Excluding infrastructure errors (109 tasks), the effective resolve rate on evaluable tasks is 6/191 = 3.1%.

## Files

- `results.json` — summary with resolved task IDs and error breakdown
- `results.csv` — per-task results (task_id, passed, time_ms, error)
- `swebench-live-shard-{0,1,2,3}.db` — raw SQLite databases from each shard
