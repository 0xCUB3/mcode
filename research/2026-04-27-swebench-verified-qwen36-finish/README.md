# SWE-bench Verified Qwen3.6 — finish-the-run attempt

## Goal

Resume the partial SWE-bench Verified run from `2026-04-25-swebench-verified-qwen36-research/full-qwen36-selection5-loop50-sb2-aborted-infra-partial.db` (271/500 tasks done, 169 passed, 62.4%) by running the remaining 229 tasks against the same Qwen3.6-35B-A3B vLLM server on Blue Vela. End state: 301/500 covered, 189 passed = 37.8% over 500, 62.8% over attempted. 199 tasks still need running.

## Combined results so far

|Source|Tasks|Passed|Pass rate|
|-|-|-|-|
|Original partial (`2026-04-25-...`)|271|169|62.4%|
|Attempt 1 (`byejek9kw` shards 1+2)|30|20|66.7%|
|**Combined**|**301**|**189**|**62.8%** over attempted, **37.8%** over 500|
|Still missing|199|—|—|

## Timeline

|Time|Event|
|-|-|
|21:16 UTC|LSF job 56569 transitioned PEND → RUN on `p4-r23-n4`|
|~21:25|`server-bv-33e2468b` healthy at `http://p4-r23-n4...:8321/v1`|
|21:25|Attempt 1 (`byejek9kw`) submitted: 229 missing ids, `--shards 4`|
|21:25-21:46|Shard 3 hit "no space left on device"; bash-wrapper auto-reset runtime; attempt 2 spawned|
|21:46-23:50|Shards 0+3 died on Docker Hub HTTP 504s (pre-fix retry list); shards 1+2 ran 30 tasks (20 passed)|
|23:50|Pushed retry-list fix (commit `9995c95`): 502/503/504 + reading-blob now retryable|
|00:04|Cancelled `byejek9kw` via `mcode bench cancel`; scp'd shards 1+2 partial DBs locally|
|00:07|Submitted `bm2nxxn17`: 199 missing ids, `--shards 4`, patched code|
|00:07-00:31|All 4 shards stalled on `_podman_image_pull_lock` flock contention + persistent Docker Hub flakiness; bash wrapper exhausted retry budget; bench exited rc=1 with 0 rows|
|00:32|Stopped, cleaned `/tmp` to 122G free|

## Why the resume didn't finish tonight

Two compounding infra problems:

1. **Docker Hub flakiness:** persistent 504 / 503 / `reading blob: HTTP status` errors when pulling SWE-bench eval images. The retry-list patch landed mid-session (commit `9995c95`) but the per-shard `with_backoff` budget (3 attempts) × bash-wrapper `max_infra_retries=1` budget = 4 total attempts per shard. 5+ pulls in a row failed during the fresh attempt.
2. **Global podman pull lock:** `_podman_image_pull_lock()` in `src/mcode/execution/swebench.py` serializes ALL podman image pulls cluster-wide via `flock`. With `--shards 4`, half the shards were blocked on the lock for 25+ minutes while the other half monopolized it. Effective parallelism = 1 even with 4 shards.

## Files

- `partial/results-shard-1.db`, `partial/results-shard-2.db` — `byejek9kw` shards 1+2 final state (30 rows, 20 passed)
- `remaining-task-ids.json` (229) — initial deficit at run start
- `still-missing-after-attempt2.json` (199) — tasks still needing a result row
- `followup-shards-0and3.csv` (115) — the shard-0/3 allocation that bm2nxxn17 attempted (subset of still-missing)
- `still-missing.csv` (199) — comma-separated form, fits inline in `--task-ids`

## What needs to happen for the next attempt

The single fix that unblocks the rest:

- **Per-image pull lock** (instead of one global lock). Allow shards to pull different images concurrently as long as no two shards try to pull the same blob.

Without that change, `--shards N` for `swebench-lite` against fresh eval images is bottlenecked by sequential pulls regardless of N. A `--shards 1` run is what we actually have; the parallelism is illusory.

Until that lands, the workable path is:

1. Wait for Docker Hub to be less flaky (no specific timing), or
2. Pre-pull the missing 199 eval images out-of-band on the cluster via `podman pull` directly with a longer retry budget, then re-run the bench with `MCODE_SKIP_IMAGE_PULL=1`, or
3. Run with `--shards 1` and accept the longer wall-clock (~16 hours sequential pulls + ~6 min/task for ~199 tasks = ~36 hours).

## Server cleanup

`server-bv-33e2468b` (LSF job 56569) is **still running** on `p4-r23-n4` (healthy, idle). Stop with:

```bash
uv run mcode launch stop server-bv-33e2468b
```

if not needed for the next attempt.
