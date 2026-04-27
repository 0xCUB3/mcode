# SWE-bench Verified Qwen3.6 Blue Vela infra repair log

This log captures the Qwen3.6 SWE-bench Verified work so far. It is not a final benchmark result. The full run was interrupted after podman rootless image unpack failures, then mcode was patched so sharded Blue Vela SWE-bench runs fail fast on infra rows, preflight and serialize prebuilt image pulls, and retry once from a fresh podman runtime on the special infra exit code.

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

## Results so far

The full Qwen run is not complete, so the Qwen vs MiniMax comparison is not final. The partial full-run rows are still useful as a signal and as the repro for the infra class.

| Run | Passed | Total rows | Rate | Notes |
|-|-:|-:|-:|-|
| Qwen3.6 partial full Verified run before infra repair | 169 | 271 | 62.4% | Stopped after podman image unpack failures in shard 2 |
| MiniMax-M2.5 full Verified baseline | 187 | 500 | 37.4% | Current repo baseline from the March 31 research folder |

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
The partial full run was too strong to ignore but not valid as a final score. It reached 169 passed rows out of 271 completed rows before the infra stop, while the current full MiniMax baseline is 187/500. That does not prove Qwen will finish above MiniMax because the remaining tasks are unknown, but it does justify spending cluster time to finish the repaired full run.

The failure was infra, not model quality. Both full-run failures hit podman rootless image pull or layer unpack behavior, with text such as `writing blob`, `adding layer`, `unpacking failed`, and `Chown error detected`. That matches Blue Vela's rootless podman constraints and the lack of broad subuid/subgid mappings.

The right repair boundary is the command layer plus the image pull path. Preflighting and serializing pulls should make most failures happen before model tokens are spent. If the rootless store is already poisoned, the Blue Vela wrapper now has a bounded runtime reset instead of making the operator discover the issue manually after hundreds of rows.

The next required result is the two-task matplotlib smoke after LSF starts the Qwen server. If that passes without infra failures, rerun the full Verified command with `--shards 4`; if podman pressure is still visible, fall back to `--shards 2` before spending another full run.
