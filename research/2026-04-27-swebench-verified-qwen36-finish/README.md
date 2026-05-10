# SWE-bench Verified Qwen3.6 finish-the-run result

## Result

**319 / 500 = 63.8% pass rate** on SWE-bench Verified with Qwen3.6-35B-A3B
served through vLLM on Blue Vela.

All 500 tasks were attempted. This replaces the earlier partial number
(62.4% over 271 tasks) with a complete run.

## Per-source contributions

| Source | Tasks | Passed | Pass rate |
|-|-:|-:|-:|
| Original partial run (`2026-04-25-...`) | 271 | 169 | 62.4% |
| `byejek9kw` (30-row salvage from killed run) | 30 | 20 | 66.7% |
| `bf68u1t7y` (69 rows before cluster admin killed) | 69 | 47 | 68.1% |
| `cap-final` (final 130 with `--cpu-limit 4`) | 130 | 83 | 63.8% |
| **Combined unique (best result per task)** | **500** | **319** | **63.8%** |

## Run config

```text
--model Qwen/Qwen3.6-35B-A3B
--backend openai (auto-resolved to vLLM endpoint on Blue Vela)
--dataset princeton-nlp/SWE-bench_Verified
--loop-budget 50
--sampling multiturn --sampling-budget 2
--selection-attempts 5
--timeout 300
--mem-limit 8g --pids-limit 512
--cpu-limit 4 (final batch only; sklearn pytest spikes were tripping admin auto-kills)
--shards 4
--on bluevela (workspace_root = /u/skula/mcode-launch, shared_root = /proj/dmfexp/skula/mcode-shared)
```

## Infra fixes made during the run

The original partial aborted on a podman image-unpack failure. The resume runs
then hit five more infra problems. Each one was fixed before the next attempt.

| Issue | Commit | Fix |
|-|-|-|
| Hardcoded `/tmp` paths filled the login3 shared filesystem | `65bab01` | Route podman runtime to `bv.workspace_root`, local caches to `~/.cache/mcode` |
| `workspace_root` per-user quota was too small for about 110 eval images | `418d410` | Move podman graphroot to `bv.shared_root` |
| Lustre/GPFS rmdir race during testbed cleanup (`ENOTEMPTY`) | `e312042` | Retry `_remove_path` rmtree with exponential backoff |
| Docker Hub anonymous-pull rate limit (`429 toomanyrequests`) | `646d031` | Wire `REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json` for authenticated pulls |
| sklearn pytest fanned out to 110 cores and got killed by cluster admin | `a692adc` + `00a6970` | Add `--cpu-limit N`, set `cpu_quota`, and set OMP/BLAS thread env vars in every eval container |

## Files

- `cap-final.db` — final 130-task batch DB, completed cleanly
- `partial/results-shard-{1,2}.db` — `byejek9kw` salvage, 30 rows
- `partial-bf68u1t7y/results-shard-{0,1,2,3}.db` — `bf68u1t7y` salvage, 69 rows
- `still-missing-final.csv` — the 130 task ids fed to the final batch
- `remaining-task-ids.json` (229), `still-missing-after-attempt2.json` (199) — resume checkpoints

## Notes

The main point is the complete 500-task score. The earlier 62.4% number was
from a 271-task partial; this run fills the holes and ends at 63.8%.

The pass rate stayed steady across the four batches: 62.4 / 66.7 / 68.1 /
63.8. The CPU cap did not visibly hurt the final batch.

The cluster etiquette fix matters. `--cpu-limit 4`, backed by OMP/BLAS thread
caps, let the run use `--shards 4` without tripping the login-node admin killer.
That is true even where cgroup-v2 support is missing or rootless podman ignores
the container CPU quota.

## How to reproduce

```bash
uv run mcode launch sync bluevela
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
uv run mcode launch wait <id>
MCODE_CONTEXT_WINDOW=262144 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 \
  uv run mcode bench swebench-lite \
    --model Qwen/Qwen3.6-35B-A3B --backend openai \
    --dataset princeton-nlp/SWE-bench_Verified \
    --loop-budget 50 --sampling multiturn --sampling-budget 2 --selection-attempts 5 \
    --timeout 300 --mem-limit 8g --pids-limit 512 \
    --cpu-limit 4 --shards 4 \
    --on bluevela \
    --db results.db
```

Requires authenticated docker.io creds at `~/.config/containers/auth.json` on
the cluster (`podman login docker.io`).
