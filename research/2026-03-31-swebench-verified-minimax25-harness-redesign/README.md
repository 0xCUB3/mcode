# SWE-bench Verified on MiniMax-M2.5 after the harness redesign

This was the first full 500-task Blue Vela run after the harness redesign landed
in both `mellea` and `mcode`. I wanted a clean answer to one question: did the
new loop fix the problems we kept seeing in the earlier MiniMax runs? Those runs
spent too much time wandering, leaned too hard on prompt nudges, verified too
late, and had too many harness paths that behaved differently.

This run used state-aware tool gating, a tighter coding tool surface, one main
text-react solving path, and the cleaned-up Blue Vela HF/cache setup. The score
was good enough to keep as the MiniMax baseline.

## Setup

The setup was simple once the harness and cluster plumbing stopped fighting us.
`MiniMaxAI/MiniMax-M2.5` ran behind an OpenAI-compatible vLLM server on Blue
Vela. The benchmark used the `swebench-lite` CLI path against
`princeton-nlp/SWE-bench_Verified`, split across 7 shards.

The benchmark itself used `mcode` commit
`90c941100580d0286eb4deffbf90d5cefc74cab4` and `mellea` commit
`76303c91b0ef29ae80945c44ec0e589e9bbbd154`. While the run was already in
flight, I made one infra follow-up in `mcode` commit
`9b8be793a853f8bb7ea82b1b330c62ba72b09ec2` so the checked-in Blue Vela launcher
would source the HF env helper instead of relying on an ad hoc script.

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

Remote benchmark job payload, submitted as a 7-shard LSF array while reusing the
existing vLLM server on `http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1`:

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

Local result snapshot:

```bash
uv run mcode results \
  --db-dir research/2026-03-31-swebench-verified-minimax25-harness-redesign/run-bluevela-main-b15-final \
  --benchmark swebench-lite \
  --compare-configs \
  --time \
  > research/2026-03-31-swebench-verified-minimax25-harness-redesign/results-summary.txt
```

## Result

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

## What changed in the harness

Most of the improvement came from taking things away. The old harness had
multiple solving paths, a lot of prompt-time nudges, and too many places where
benchmark-specific behavior could leak into the wrong layer. It could look busy
for a long time without getting closer to a patch.

In `mellea`, the coding loop now leans on bounded read, search, find, list,
structured edit, and `run_tests`. `bash` still exists, but it is an escape hatch
rather than the default tool for everything. I also added state-aware tool
gating, better text tool-call repair, and a generic `tool_gate` hook so callers
can block bad actions from runtime state instead of hoping the prompt says it
strongly enough.

In `mcode`, the benchmark agent goes through one main text-react path. The old
alternate strategy plumbing is gone from the active harness. Verification now
happens during the loop, when it can still change the trajectory, instead of
mostly showing up as a late rejection. The Blue Vela scripts also picked up the
shared HF auth and cache setup as part of the normal launcher path.

## Notes from the run

The new loop was clearly better than the old budget-nudge setup. A full
187/500 on Verified gave us the first MiniMax number from this harness that felt
like a real baseline instead of a one-off experiment.

The HF/cache cleanup mattered more than I expected. Early on, anonymous Hugging
Face requests were still getting rate-limited. Once the shared cache and token
backed env were in place, startup stopped dominating the run.

The harness still lost work in two familiar ways. Some tasks ended with budget
exhaustion and an unsubmitted diff because verification never got far enough to
unlock submission. Others produced a patch that applied cleanly, and sometimes
passed a narrow local check, but still missed the real benchmark requirement.
That is patch quality, not cluster plumbing.

The logs were much cleaner than the older runs. The runtime blocked test runs
through `bash`, pushed them through `run_tests`, and repaired malformed text tool
calls without burning half the trajectory. You can see the model being steered
back onto the structured path instead of drifting into shell churn.

Blue Vela cleanup was still rough. The benchmark finished, all shard DBs were
written, and every shard printed its done marker, but the array job still had to
be killed because wrapper cleanup never exited cleanly. That looked like podman
or shell teardown lag, not unfinished benchmark work.

Docker Hub pull limits also appeared late in the run while shards were moving
between eval images. They did not change the final score, but they are still an
annoying infra tax.

## Files

- `results-summary.txt` - CLI summary snapshot
- `final-summary.json` - exact totals used above
- `run-bluevela-main-b15-final/` - final shard DBs for this run
