# SWE-bench Verified on MiniMax-M2.5 after the harness redesign

This was the first full 500-task Blue Vela run after the harness redesign landed in both `mellea` and `mcode`. The point was not just to get another score on the board. I wanted to see whether the redesign actually fixed the control-loop problems that had been showing up in the earlier MiniMax runs: too much wandering, too much soft prompting, too much late verification, and too many alternate paths in the harness.

The run used the new state-aware tool gating, the tighter code tool surface, the single main text-react solving path, and the Blue Vela HF auth and shared-cache cleanup. The result was good enough to keep as the new baseline.

HTML snapshot: [`swebench-verified-report.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-03-31-swebench-verified-minimax25-harness-redesign/swebench-verified-report.html) ([source](swebench-verified-report.html))

## Setup

The setup was straightforward once the harness and cluster plumbing stopped fighting us. We ran `MiniMaxAI/MiniMax-M2.5` behind an OpenAI-compatible vLLM server on Blue Vela, used the `swebench-lite` CLI path against `princeton-nlp/SWE-bench_Verified`, and split the full 500-task run across 7 shards. The actual benchmark run used `mcode` commit `90c941100580d0286eb4deffbf90d5cefc74cab4` together with `mellea` commit `76303c91b0ef29ae80945c44ec0e589e9bbbd154`. After the run was already in flight, I made one follow-up infra change in `mcode` commit `9b8be793a853f8bb7ea82b1b330c62ba72b09ec2` so the checked-in Blue Vela launchers would permanently source the HF env helper instead of needing an ad hoc script.
## Commands

Remote checkout and deps:

```bash
RUN_DIR=/u/skula/mcode-main-20260331-harness
git clone https://github.com/0xCUB3/mcode "$RUN_DIR"
cd "$RUN_DIR"
git checkout 90c941100580d0286eb4deffbf90d5cefc74cab4
uv sync --extra dev --extra swebench --extra datasets
```

Remote HF bootstrap:

```bash
source /u/skula/.config/mcode/hf-env.sh
uv run python - <<'PY'
from datasets import load_dataset
load_dataset("princeton-nlp/SWE-bench_Verified", split="test[:1]")
print("dataset cache warm")
PY
```

Remote benchmark job payload, submitted as a 7-shard LSF array while reusing the existing vLLM server on `http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1`:

```bash
export OPENAI_BASE_URL=http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh

uv run mcode bench swebench-lite \
  --backend openai \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --shard-count 7 \
  --shard-index <0..6> \
  --n-samples 1 \
  --db /u/skula/mcode/results/live-m25-verified-full-main-20260331-b15-hf-shard-<idx>.db
```

Local report generation:

```bash
uv run mcode report \
  --db-dir research/2026-03-31-swebench-verified-minimax25-harness-redesign/run-bluevela-main-b15-final \
  --benchmark swebench-lite \
  --out research/2026-03-31-swebench-verified-minimax25-harness-redesign/swebench-verified-report.html

uv run mcode results \
  --db-dir research/2026-03-31-swebench-verified-minimax25-harness-redesign/run-bluevela-main-b15-final \
  --benchmark swebench-lite \
  --compare-configs \
  --time \
  > research/2026-03-31-swebench-verified-minimax25-harness-redesign/results-summary.txt
```

## Key results

| Metric | Value |
|-|-:|
| Total tasks | 500 |
| Passed | 187 |
| Pass rate | 37.4% |
| Shards | 7 |
| vLLM job | 733630 |
| Benchmark job array | 762048 |

Per-shard totals:

| Shard | Completed | Passed | Pass rate |
|-|-:|-:|-:|
| 0 | 72 | 20 | 27.8% |
| 1 | 72 | 23 | 31.9% |
| 2 | 72 | 33 | 45.8% |
| 3 | 71 | 27 | 38.0% |
| 4 | 71 | 30 | 42.3% |
| 5 | 71 | 26 | 36.6% |
| 6 | 71 | 28 | 39.4% |

## What changed in the architecture

The useful part of this redesign was mostly subtraction. The old harness had multiple solving paths, a lot of prompt-time nudges, and too many ways for benchmark-specific behavior to leak into the wrong layer. The result was a loop that could look busy for a long time without actually converging.

In `mellea`, I tightened the tool surface around bounded read, search, find, list, structured edit, and `run_tests`. `bash` is still there, but now it is an escape hatch instead of the default way to do everything. I also added state-aware tool gating, better text tool-call repair, and a generic `tool_gate` hook so the caller can block bad actions based on actual runtime state instead of just hoping the prompt will be persuasive enough.

In `mcode`, the benchmark agent now goes through one main text-react path. The old alternate strategy plumbing is gone from the active harness design. Verification is part of the loop now, steering the run while it is still salvageable, instead of showing up mostly as a late rejection step. The Blue Vela scripts were also cleaned up so shared HF auth and cache settings are part of the normal launcher path.

## Findings

The redesign is meaningfully better than the old budget-nudge loop. Finishing at 187/500 on the full Verified slice gave us the first MiniMax result from this architecture that feels like a real baseline instead of an experiment artifact.

The HF and cache cleanup mattered more than I expected. Early in the session, anonymous Hugging Face requests were still getting rate-limited. Once the shared cache and token-backed env were in place, startup stopped being the bottleneck and the run could settle into normal task execution.

The harness still loses work in two familiar ways. Some tasks die with budget exhaustion and an unsubmitted diff because verification never got far enough to unlock submission. Others produce a patch that applies cleanly and even passes a narrow local check, but still misses the real benchmark requirement. That is a scaffold quality problem, not an infrastructure problem.

The logs are cleaner than the old runs. The runtime now blocks test execution through `bash`, pushes that work through `run_tests`, and recovers malformed text tool calls without wasting half the trajectory. You can see the model getting redirected into the structured path instead of wandering off into shell churn.

Blue Vela cleanup is still ugly. The benchmark itself finished, all shard DBs were written, and every shard printed its done marker, but the array job still had to be killed because the wrapper cleanup never exited cleanly. That looks like podman or shell teardown lag, not unfinished benchmark work.

Docker Hub pull limits also showed up late in the run while shards were moving between eval images. They did not erase the final score, but they are still an annoying infra tax and worth removing if this becomes a routine benchmark path.

## Files

- `swebench-verified-report.html` - interactive report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-03-31-swebench-verified-minimax25-harness-redesign/swebench-verified-report.html))
- `results-summary.txt` - CLI summary snapshot
- `final-summary.json` - exact totals used above
- `run-bluevela-main-b15-final/` - final shard DBs and logs for this run
