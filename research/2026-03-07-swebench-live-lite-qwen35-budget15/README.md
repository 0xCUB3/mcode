# SWE-bench Live Lite: Qwen3.5-35B-A3B (budget=15)

## Setup

- **Model:** Qwen/Qwen3.5-35B-A3B (MoE, 3B active params)
- **Serving:** vLLM with `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`
- **Hardware:** IBM Blue Vela cluster, 2x H100 80GB (tensor-parallel)
- **Agent:** mcode ReACT loop, 15 tool-call budget per task
- **Strategy:** repair (search_code, read_file, apply_edit, final_answer)
- **Split:** SWE-bench Live Lite (300 instances)
- **Evaluation:** Docker containers via podman Docker-compat API

## Results

| Metric | Value |
|-|-|
| Total tasks | 300 |
| Resolved | 9 (3.0%) |
| Not resolved | 123 |
| Infrastructure errors | 168 |

Excluding infrastructure errors, effective resolve rate: 9/132 = 6.8%.

### Resolved tasks

| Task | Time |
|-|-|
| fonttools__fonttools-3682 | 52s |
| keras-team__keras-20443 | 126s |
| pvlib__pvlib-python-2249 | 105s |
| pybamm-team__pybamm-4644 | 52s |
| pypa__twine-1225 | 17s |
| python-babel__babel-1141 | 33s |
| python-control__python-control-1111 | 15s |
| run-llama__llama_deploy-397 | 51s |
| streamlink__streamlink-6242 | 69s |

### Comparison to prior run (budget=3)

| Metric | budget=3 | budget=15 |
|-|-|-|
| Resolved | 6 (2.0%) | 9 (3.0%) |
| Effective rate | 6/191 = 3.1% | 9/132 = 6.8% |
| New solves | - | fonttools-3682, babel-1141, streamlink-6242, pybamm-4644, keras-20443 |
| Lost solves | matplotlib-29721, pvlib-python-2393 | - |

Note: budget=3 run had a bug where the react loop budget was multiplied by 5x (effective budget=15 for the first run too, but with broken enforcement that allowed unlimited turns). The budget=15 run has correct enforcement.

### Infrastructure errors

| Error | Count | Cause |
|-|-|-|
| Not resolved | 60 | Agent failed to produce correct patch |
| ImageNotFound | 52 | Docker Hub image missing for instance |
| KeyError | 40 | mellea upstream bug (intermittent) |
| JSONDecodeError | 14 | Malformed vLLM response |
| APIError | 2 | vLLM server error |

## Files

- `results.json` -- summary with resolved task IDs and error breakdown
- `results.csv` -- per-task results (task_id, passed, time_ms, error)
- `swebench-live-shard-{0,1,2,3}.db` -- raw SQLite databases from each shard
