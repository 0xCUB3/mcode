# SWE-bench Verified Qwen3.6 — finish-the-run (final)

## Result

**319 / 500 = 63.8% pass rate** on SWE-bench Verified with Qwen3.6-35B-A3B served via vLLM on Blue Vela.

All 500 tasks attempted. Beats the original aborted partial (62.4% over 271) and the published baselines.

## Per-source contributions

|Source|Tasks|Passed|Pass rate|
|-|-|-|-|
|Original partial run (`2026-04-25-...`)|271|169|62.4%|
|`byejek9kw` (30-row salvage from killed run)|30|20|66.7%|
|`bf68u1t7y` (69 rows before cluster admin killed)|69|47|68.1%|
|`cap-final` (final 130 with `--cpu-limit 4`)|130|83|63.8%|
|**Combined unique (best result per task)**|**500**|**319**|**63.8%**|

## Run config

```
--model Qwen/Qwen3.6-35B-A3B
--backend openai (auto-resolved to vLLM endpoint on Blue Vela)
--dataset princeton-nlp/SWE-bench_Verified
--loop-budget 50
--sampling multiturn --sampling-budget 2
--selection-attempts 5
--timeout 300
--mem-limit 8g --pids-limit 512
--cpu-limit 4 (final batch only — sklearn pytest spikes were tripping login-node admin auto-killers)
--shards 4
--on bluevela (workspace_root = /u/skula/mcode-launch, shared_root = /proj/dmfexp/skula/mcode-shared)
```

## Infra fixes that landed during this run

The original partial aborted on a podman image-unpack failure. The resume run hit five additional infra issues, each fixed before the next attempt:

|Issue|Commit|Fix|
|-|-|-|
|Hardcoded `/tmp` paths filling login3 shared filesystem|`65bab01`|Route podman runtime to `bv.workspace_root`, local caches to `~/.cache/mcode`|
|`workspace_root` per-user quota too small for ~110 eval images|`418d410`|Move podman graphroot to `bv.shared_root` (multi-TB)|
|Lustre/GPFS rmdir race during testbed cleanup (`ENOTEMPTY`)|`e312042`|Retry `_remove_path` rmtree with exponential backoff|
|Docker Hub anonymous-pull rate limit (`429 toomanyrequests`)|`646d031`|Wire `REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json` so authenticated pulls (~10× rate limit)|
|sklearn pytest fanning out to 110 cores → cluster admin auto-kill|`a692adc` + `00a6970`|Add `--cpu-limit N` flag; set `cpu_quota` (cgroup-v1 silently no-ops in rootless podman) AND `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `BLIS_NUM_THREADS` env vars on every eval container — library-level cap that always works|

## Files

- `cap-final.db` — final 130-task batch DB (the one that completed cleanly)
- `partial/results-shard-{1,2}.db` — `byejek9kw` salvage (30 rows)
- `partial-bf68u1t7y/results-shard-{0,1,2,3}.db` — `bf68u1t7y` salvage (69 rows)
- `still-missing-final.csv` — the 130 task ids fed to the final batch
- `remaining-task-ids.json` (229), `still-missing-after-attempt2.json` (199) — checkpoints during the resume

## What's notable

- **63.8% over 500 with no holes.** Earlier we reported 62.4% but only over the 271-task partial; this is the full 500.
- **Pass-rate is consistent across the 4 batches** (62.4 / 66.7 / 68.1 / 63.8). No regression from the cpu-limit cap on the final batch.
- **Cluster etiquette resolved.** The `--cpu-limit 4` flag (with the OMP env var fallback) lets us run --shards 4 against this dataset without ever tripping the login-node admin auto-killer, regardless of cgroup-v2 support.

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

Requires authenticated docker.io creds at `~/.config/containers/auth.json` on the cluster (`podman login docker.io`).
