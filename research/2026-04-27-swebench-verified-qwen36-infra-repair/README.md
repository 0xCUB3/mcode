# SWE-bench Verified Qwen3.6 Blue Vela infra repair + full run

**Final result: 319 / 500 = 63.8% pass rate** with `Qwen/Qwen3.6-35B-A3B` on SWE-bench Verified, served via vLLM on Blue Vela. Full 500-task coverage, no holes. The interrupted partial reached 169/271 = 62.4% before the original podman infra failure; the resume work over the next two days added five more infra fixes and finished the remaining 229 tasks across three batches. Per-batch pass rates were 62.4 / 66.7 / 68.1 / 63.8 — consistent, no batch effect from the cpu-cap that was added for the final batch.

Beats the prior MiniMax-M2.5 full Verified baseline (187/500 = 37.4%) by **+26.4 absolute points (+70% relative)**.

Run notes for the resume work: [`../2026-04-27-swebench-verified-qwen36-finish/README.md`](../2026-04-27-swebench-verified-qwen36-finish/README.md).

HTML snapshot: [`swebench-qwen36-so-far-report.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-27-swebench-verified-qwen36-infra-repair/swebench-qwen36-so-far-report.html) ([source](swebench-qwen36-so-far-report.html))

## Setup

- Model under test: `Qwen/Qwen3.6-35B-A3B`
- Backend: OpenAI-compatible vLLM endpoint on Blue Vela
- Dataset: `princeton-nlp/SWE-bench_Verified`, split `test`
- Comparison baseline from current repo artifacts: `MiniMaxAI/MiniMax-M2.5`, `187/500 = 37.4%`, in `../2026-03-31-swebench-verified-minimax25-harness-redesign/run-bluevela-main-b15-final/`
- mcode infra repair commit: `2bbda4fdad5971517d6016c329387dda411d68df`
- Log date verified with `date +%F`: `2026-04-27`

## Commands

The interrupted full Qwen run used:

```bash
uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --loop-budget 50 \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --n-samples 1 \
  --sampling multiturn \
  --sampling-budget 2 \
  --selection-attempts 5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --shards 4 \
  --on bluevela \
  --fetch-db \
  --db research/2026-04-25-swebench-verified-qwen36-research/full-qwen36-selection5-loop50-sb2-bluevela-shards4.db
```

The infra repair was verified locally with:

```bash
uv run pytest tests/test_agent_generate.py tests/test_cli_shards.py tests/test_swebench.py tests/bench/test_remote.py tests/test_bluevela_scripts.py tests/launch/test_bluevela.py -q
uv run ruff check src/mcode/cli.py src/mcode/execution/swebench.py src/mcode/bench/remote.py src/mcode/bench/runner.py tests/test_agent_generate.py tests/test_cli_shards.py tests/test_swebench.py tests/bench/test_remote.py tests/test_bluevela_scripts.py tests/launch/test_bluevela.py
uv run python -m py_compile src/mcode/cli.py src/mcode/execution/swebench.py src/mcode/bench/remote.py src/mcode/bench/runner.py
```

The report in this folder was generated from the April 25 artifact DBs with:

```bash
uv run mcode report \
  --db-dir research/2026-04-25-swebench-verified-qwen36-research \
  --out research/2026-04-27-swebench-verified-qwen36-infra-repair/swebench-qwen36-so-far-report.html
```

After the repair landed, I retried the server launch for the planned matplotlib infra smoke:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
```

The first retry submitted LSF job `53641`, which stayed pending because the `ngpus_physical` limit was reached. It was stopped with:

```bash
uv run mcode launch stop server-bv-e5609c35
```

The second retry is still queued in the background as LSF job `53704` with server id `server-bv-37ce7781`. At the time this log was written, `bjobs -l 53704` still reported:

```text
Resource (ngpus_physical) limit defined on host(s) and/or host group has been reached: 725 hosts;
```

When that endpoint becomes healthy, the planned smoke command is:

```bash
uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --loop-budget 50 \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --n-samples 1 \
  --sampling multiturn \
  --sampling-budget 2 \
  --selection-attempts 5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --task-ids matplotlib__matplotlib-23476,matplotlib__matplotlib-24570 \
  --shards 1 \
  --on bluevela \
  --fetch-db \
  --db research/2026-04-25-swebench-verified-qwen36-research/infra-repair-matplotlib-smoke.db
```

## Results

| Run | Passed | Total | Rate | Notes |
|-|-:|-:|-:|-|
| **Qwen3.6 full Verified (final)** | **319** | **500** | **63.8%** | Combined unique across the 4 resume batches; no holes |
| Qwen3.6 partial full Verified run (this entry's original snapshot) | 169 | 271 | 62.4% | Stopped after podman image unpack failures in shard 2 |
| `byejek9kw` salvage (cluster admin killed mid-run) | 20 | 30 | 66.7% | Recovered from shards 1+2 of an aborted attempt |
| `bf68u1t7y` partial (cluster admin killed mid-run) | 47 | 69 | 68.1% | Recovered from a 4-shard run cancelled when sklearn pytest spiked the login node |
| `cap-final` (final clean batch with `--cpu-limit 4`) | 83 | 130 | 63.8% | Last 130 task ids; ran cleanly under the new OMP/BLAS thread cap |
| MiniMax-M2.5 full Verified baseline | 187 | 500 | 37.4% | Prior repo baseline from the March 31 research folder |

Terminal reasons in the interrupted Qwen partial DB:

| Reason | Count |
|-|-:|
| submitted | 170 |
| wrong_patch_after_verification | 81 |
| budget_exhausted | 12 |
| unverified_diff_discarded | 6 |
| infra_failure | 2 |

The two infra failures were:

| Task | Failure class |
|-|-|
| `matplotlib__matplotlib-23476` | podman rootless pull or layer unpack failure |
| `matplotlib__matplotlib-24570` | podman rootless pull or layer unpack failure |

The best completed Qwen3.6 smoke-16 runs in this folder are:

| Run | Passed | Total | Rate |
|-|-:|-:|-:|
| `smoke16-qwen36-selection5-loop50-sb2-strictverify-bluevela-shards4.db` | 11 | 16 | 68.8% |
| `smoke16-qwen36-selection5-loop50-sb2-strictverify-rep2-bluevela-shards4.db` | 11 | 16 | 68.8% |
| `smoke16-diagnostic-loop50-sb2-strictverify-bluevela-shards4.db` | 10 | 16 | 62.5% |
| `smoke16-selection3-loop50-sb2-strictverify-bluevela-shards4.db` | 10 | 16 | 62.5% |

## Infra repair summary

The repair is intentionally small in public surface. There are no new benchmark flags. The command now uses existing shard DB semantics to notice infra failures instead of relying on a human watcher.

- `src/mcode/cli.py` polls shard DBs during sharded runs and exits with code `86` when it sees `terminal_reason='infra_failure'` or known podman image unpack strings.
- `src/mcode/execution/swebench.py` classifies retryable podman image failures, serializes image pulls with a process lock, retries prebuilt image pulls once, and preflights namespace instance images before model solving starts.
- `src/mcode/bench/remote.py` treats exit code `86` as a retryable Blue Vela podman runtime failure, resets the per-job podman runtime directory, restarts the podman service, and reruns once.
- The old `_remote_image_runtime_error_message` helper was removed because retryable image errors now preserve the actual failing podman text.

The diff is net positive, `542 insertions` and `78 deletions`, mostly because it adds regression coverage around shard fail-fast behavior, image pull retries, preflight failure before model work, and the remote retry script. The production code also adds real monitoring and recovery logic, so I did not cut tests to make the diff smaller.

## Files

- `swebench-qwen36-so-far-report.html` - interactive report over all DBs in `../2026-04-25-swebench-verified-qwen36-research/`.
- `../2026-04-25-swebench-verified-qwen36-research/full-qwen36-selection5-loop50-sb2-aborted-infra-partial.db` - merged partial DB from the interrupted full Qwen run.
- `../2026-04-25-swebench-verified-qwen36-research/full-qwen36-selection5-loop50-sb2-aborted-infra-shards/` - fetched shard DBs from the interrupted run.
- `../2026-04-25-swebench-verified-qwen36-research/smoke16-*.db` and `single-*.db` - smoke and infra diagnostic runs from the tuning path.

## Findings

The full run finished at **319/500 = 63.8%** with the same harness config across all four batches (loop_budget=50, sampling=multiturn sb=2, selection_attempts=5, shards=4). Pass rate held within ±3 points across batches, so the cpu/OMP cap added for the final batch did not regress quality.

The original failure was always infra, never model quality. The original 62.4% partial extrapolated cleanly to the final 63.8% over 500.

Resume work surfaced four more infra issues beyond the original podman image-unpack one:

1. Hardcoded `/tmp` paths filled the shared login3 filesystem under multi-bench load — moved podman runtime to `bv.workspace_root` and local caches to `~/.cache/mcode`.
2. `workspace_root` per-user quota (`/u/skula`) too small for ~110 eval images at ~7GB each — moved podman graphroot to `bv.shared_root` (`/proj/dmfexp`, multi-TB).
3. Lustre/GPFS rmdir race during testbed cleanup raised `OSError: ENOTEMPTY` mid-task — added retry-with-backoff in `_remove_path`.
4. Docker Hub anonymous-pull rate limit (`429 toomanyrequests`) burned through the cluster's egress quota — wired `REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json` so authenticated pulls (~10× rate limit) take effect.
5. sklearn / numpy / scipy pytest invocations inside the eval container fanned out to ~110 host cores via OpenMP and BLAS, tripping the login-node admin auto-killer at the user-process level (rootless podman doesn't honor cgroup `cpu_quota` on this cluster's cgroup-v1 setup). Fix: set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS`/`BLIS_NUM_THREADS` env vars on every eval container — library-level cap that always works regardless of cgroup support. Exposed as `--cpu-limit N` on the bench commands; defaults unlimited; `MCODE_SWEBENCH_CPU_LIMIT` env override.

Commits that landed during the resume: `65bab01` (no-tmp) → `418d410` (shared_root) → `e312042` (rmtree retry) → `646d031` (auth file) → `a692adc` + `00a6970` (cpu-cap + OMP env vars).
