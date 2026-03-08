# SWE-bench Live Lite: Qwen3.5-35B-A3B

## Setup

- **Model:** Qwen/Qwen3.5-35B-A3B (MoE, 3B active params)
- **Serving:** vLLM with `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`
- **Hardware:** IBM Blue Vela cluster, 2x H100 80GB (tensor-parallel)
- **Agent:** mcode ReACT loop
- **Strategy:** repair (search_code, read_file, apply_edit, final_answer)
- **Split:** SWE-bench Live Lite (300 instances)
- **Evaluation:** Docker containers via podman Docker-compat API

## Commands

Start vLLM server:

```bash
cd /u/skula/mcode
bash deploy/bluevela/start-vllm.sh
```

Run 1 (budget=3):

```bash
SWB_SPLIT=lite LOOP_BUDGET=3 bash deploy/bluevela/run-swebench-live.sh
```

Run 2 (budget=15):

```bash
SWB_SPLIT=lite LOOP_BUDGET=15 bash deploy/bluevela/run-swebench-live.sh
```

## Results

| Metric | Run 1 (budget=3) | Run 2 (budget=15) |
|-|-|-|
| Total tasks | 300 | 300 |
| Resolved | 6 (2.0%) | 9 (3.0%) |
| Not resolved | 185 | 123 |
| Infrastructure errors | 109 | 168 |
| Effective rate (excl infra) | 6/191 = 3.1% | 9/132 = 6.8% |

### Resolved tasks

| Task | Run 1 | Run 2 |
|-|-|-|
| fonttools__fonttools-3682 | | 52s |
| keras-team__keras-20443 | | 126s |
| matplotlib__matplotlib-29721 | 115s | |
| pvlib__pvlib-python-2249 | 48s | 105s |
| pvlib__pvlib-python-2393 | 27s | |
| pypa__twine-1225 | 55s | 17s |
| pybamm-team__pybamm-4644 | | 52s |
| python-babel__babel-1141 | | 33s |
| python-control__python-control-1111 | 38s | 15s |
| run-llama__llama_deploy-397 | 61s | 51s |
| streamlink__streamlink-6242 | | 69s |

Union of resolved tasks across both runs: 11.

### Infrastructure errors

| Error | Run 1 | Run 2 |
|-|-|-|
| ImageNotFound | 46 | 52 |
| KeyError (mellea upstream) | 42 | 40 |
| JSONDecodeError (vLLM) | 13 | 14 |
| ReadTimeout (podman) | 7 | 0 |
| APIError (vLLM) | 1 | 2 |
| Not resolved (error field) | 0 | 60 |

### Notes

- Run 1 had a bug: the react loop budget was multiplied by 5x internally, so the effective budget was 15 but with broken enforcement that allowed unlimited turns (some tasks ran 60+ turns).
- Run 2 fixed the multiplier bug. Budget of 15 is correctly enforced.
- ReadTimeout errors in run 1 were fixed by increasing the Docker SDK timeout from 60s to 600s.
- "Not resolved" in the error column of run 2 means the agent produced a patch but it didn't pass the evaluation tests (distinct from empty error = agent's patch was wrong but evaluation ran successfully).

## Files

- `swebench-live-report.html` -- interactive Plotly report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-03-06-swebench-live-qwen35/swebench-live-report.html))
- `run1-budget3/` -- first run (budget=3, broken 5x multiplier)
  - `results.json`, `results.csv`, `swebench-live-shard-{0,1,2,3}.db`
- `run2-budget15/` -- second run (budget=15, fixed)
  - `results.json`, `results.csv`, `swebench-live-shard-{0,1,2,3}.db`
