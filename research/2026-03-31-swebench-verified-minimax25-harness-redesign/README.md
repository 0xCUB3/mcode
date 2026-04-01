# SWE-bench Verified on MiniMax-M2.5 after the harness redesign

Goal: measure the first full 500-task Blue Vela run after the harness redesign landed in mellea and mcode. This run used the new state-aware tool gating, the sharper code tool surface, the single main text-react solving path, and the Blue Vela HF auth/shared-cache cleanup.

HTML snapshot: [`swebench-verified-report.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-03-31-swebench-verified-minimax25-harness-redesign/swebench-verified-report.html) ([source](swebench-verified-report.html))

## Setup

- Model: `MiniMaxAI/MiniMax-M2.5`
- Backend: OpenAI-compatible vLLM on Blue Vela
- Benchmark: `swebench-lite` CLI path with dataset `princeton-nlp/SWE-bench_Verified`
- Split size: 500 tasks total across 7 shards
- Loop budget: 15
- Timeout: 300 seconds per task
- Run checkout: `mcode` `90c941100580d0286eb4deffbf90d5cefc74cab4`
- Agent substrate: `mellea` `76303c91b0ef29ae80945c44ec0e589e9bbbd154`
- Follow-up infra commit: `mcode` `9b8be793a853f8bb7ea82b1b330c62ba72b09ec2` made the Blue Vela HF env wiring permanent in the checked-in launchers after this run was already launched.

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

The useful part of this redesign was subtraction. The old harness had multiple solving paths, prompt-time budget nudges, and strategy surface that let benchmark policy leak into too many places. This run used the new layout instead.

In `mellea`, the code tool surface was tightened around bounded read/search/find/list, structured edit, and `run_tests`, with `bash` demoted to an escape hatch instead of the main workflow. The runtime now has state-aware tool gating, stronger text tool-call repair, and a generic `tool_gate` hook so callers can block bad actions based on actual execution state rather than prompt reminders.

In `mcode`, the benchmark agent now runs through one main text-react path. The old alternate strategy plumbing was removed from the active harness design, verification moved into the loop as steering instead of mostly end-of-run rejection, and the Blue Vela scripts were cleaned up to use shared HF auth and cache settings.

## Findings

- The redesign is materially better than the old budget-nudge loop. The run reached 187/500 on the full Verified slice, which is the first result from this new architecture worth keeping as a real baseline.
- The HF/auth/cache fix mattered operationally. Anonymous Hugging Face startup requests were rate-limited early in the session. After adding the shared HF cache and token-backed env, dataset startup stopped being the bottleneck.
- The harness still loses work in two ways. Some tasks end with budget exhaustion and an unsubmitted diff because verification never crossed the submission gate. Others produce a cleanly applied patch that passes a narrow local check but still does not resolve the benchmark task.
- Tool discipline is visibly better in the logs. The new runtime blocks test execution through `bash`, pushes those calls through `run_tests`, and recovers malformed text tool calls without derailing the run.
- Cleanup is still rough on Blue Vela. The benchmark itself finished, but the array job had to be killed after every shard had already written its final summary. That looks like podman or wrapper cleanup lag rather than unfinished benchmark work.
- Docker Hub image pull limits showed up late in the run while shards were moving between eval images. That did not erase the result, but it is still an infra paper cut worth removing if these runs become routine.

## Files

- `swebench-verified-report.html` - interactive report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-03-31-swebench-verified-minimax25-harness-redesign/swebench-verified-report.html))
- `results-summary.txt` - CLI summary snapshot
- `final-summary.json` - exact totals used above
- `run-bluevela-main-b15-final/` - final shard DBs and logs for this run
