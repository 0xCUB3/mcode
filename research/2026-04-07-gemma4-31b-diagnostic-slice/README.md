# Gemma 4 31B on the 16-task diagnostic slice

## Goal / scope

This was a Blue Vela check of `google/gemma-4-31B-it` on the same 16-task SWE-bench Verified diagnostic slice used for the April 3 harness work. I used vLLM rather than Ollama because vLLM already has official Gemma 4 parser support and a clean OpenAI-compatible endpoint shape.

The original plan was to push parallelism hard. That exposed launcher and rootless podman problems first, so the only publishable result from this session is the fixed-service single-host rerun on `p1-r04-n1`.

HTML snapshot: [`diagnostic-swebench-report-gemma4-31b-single.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-07-gemma4-31b-diagnostic-slice/diagnostic-swebench-report-gemma4-31b-single.html) ([source](diagnostic-swebench-report-gemma4-31b-single.html))

## Environment + commands

vLLM server:

- model: `google/gemma-4-31B-it`
- server job: `822802`
- host: `p2-r28-n2.bluevela.rmf.ibm.com`
- endpoint: `http://p2-r28-n2.bluevela.rmf.ibm.com:8331/v1`
- tensor parallel: `2`
- vLLM parsers: `--tool-call-parser gemma4 --reasoning-parser gemma4`

Shared benchmark settings:

```bash
export OPENAI_BASE_URL=http://p2-r28-n2.bluevela.rmf.ibm.com:8331/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh
```

Final valid benchmark command:

```bash
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman/podman.sock"

uv run mcode bench swebench-lite \
  --backend openai \
  --model google/gemma-4-31B-it \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --n-samples 1 \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --db research/2026-04-07-gemma4-31b-diagnostic-slice/run-bluevela-gemma4-31b-diagnostic-b15-single-fixedservice-p1-20260407/diagnostic.db
```

The worker-side launcher used the same podman service shape that had already worked in the earlier 8-shard script:

```bash
podman \
  --cgroup-manager=cgroupfs \
  --storage-driver=overlay \
  --root="$GRAPHROOT" \
  --runroot="$RUNROOT" \
  system service --time=0 "unix://$SOCK"
```

Blue Vela attempts from this session:

- `822818`, 8 shards, invalid
- `822848`, `822943`, 4 shards, invalid
- `823114`, single pinned host `p1-r04-n1`, valid

## Key results

| Attempt | Job(s) | Shape | Result | Notes |
|-|-|-|-|-|
| Initial parallel run | `822818` | 8 shards | invalid | mixed infra failures, not publishable |
| Parallel retries | `822848`, `822943` | 4 shards | invalid | podman socket and shared-state failures, not publishable |
| Fixed-service rerun | `823114` | single host `p1-r04-n1` | `5/16`, `31.25%` | first valid Gemma 4 run from this session |

Final fixed-service metrics:

| Metric | Value |
|-|-|
| Total tasks | 16 |
| Passed | 5 |
| Pass rate | 31.25% |
| Terminal reason `submitted` | 5 |
| Terminal reason `unverified_diff_discarded` | 7 |
| Terminal reason `wrong_patch_after_verification` | 4 |
| Verification-succeeded tasks | 9 |
| Zero-edit tasks | 0 |
| Zero-verification tasks | 6 |
| Malformed tool-call recoveries | 59 |
| Blocked verification commands | 0 |
| Avg turns to first edit | 5.88 |
| Avg turns to first verification | 6.70 |

Passed tasks:

- `astropy__astropy-12907`
- `astropy__astropy-13453`
- `astropy__astropy-14309`
- `scikit-learn__scikit-learn-13328`
- `sympy__sympy-13877`

## Findings

- vLLM was the right serving path. Once the launcher reused the known-good podman service shape, Gemma ran cleanly through all 16 tasks with no infrastructure failures.
- The early sharded runs were not model evidence. They were invalid because the later launcher variants drifted away from the working podman setup and shared rootless state too aggressively.
- The final valid Gemma result was `5/16`, which is below the current MiniMax control run at `8/16` on the same slice.
- The failure mix is mostly semantic, not transport or tool-boundary churn. There were `7` `unverified_diff_discarded` outcomes and `4` `wrong_patch_after_verification` outcomes.
- `blocked_verification_commands` stayed at `0`, so Gemma already fits the current verification boundary reasonably well. The bigger problem is patch quality after it gets through the tool interface.

## Additional files

- `diagnostic-swebench-report-gemma4-31b-single.html` - interactive report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-07-gemma4-31b-diagnostic-slice/diagnostic-swebench-report-gemma4-31b-single.html))
- `diagnostic-results-summary-gemma4-31b-single.txt` - plain-text summary snapshot
- `run-bluevela-gemma4-31b-diagnostic-b15-single-fixedservice-p1-20260407/diagnostic.db` - final results DB
- `run-bluevela-gemma4-31b-diagnostic-b15-single-fixedservice-p1-20260407/run_gemma4_diag_single.sh` - exact Blue Vela launcher used for the valid rerun
