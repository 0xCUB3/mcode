# Mellea-first harness experiments on Qwen3.5 smoke

## Goal / scope

This batch was the next pass after the final-answer selection fix. The idea was to move the harness closer to a Mellea-first shape and see whether that actually helps the Blue Vela smoke slice on `Qwen/Qwen3.5-35B-A3B`.

I focused on three kinds of changes. First, use more of the forked Mellea surface directly, especially the agent tool builder. Second, reuse Mellea loop control ideas like repeated-call detection instead of adding more prompt nudges in mcode. Third, test whether malformed required-arg tool calls can be turned into retries instead of wasted turns.

Rendered HTML report:

- `research/2026-04-21-mellea-first-harness-ab/mellea-first-harness-ab-report.html`
- https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-21-mellea-first-harness-ab/mellea-first-harness-ab-report.html

## Environment + commands

Shared setup:

- model: `Qwen/Qwen3.5-35B-A3B`
- backend: `openai`
- target: Blue Vela
- smoke budget: `15`
- timeout: `300`
- repo synced to Blue Vela with `uv run mcode launch sync bluevela`
- remote env refreshed with `uv sync --all-extras`

This pass also aligned `pyproject.toml` with the git-backed Mellea install we were already using locally, instead of the old plain `mellea==0.4.2` pin. That pulled the Blue Vela runtime onto the same forked Mellea line as the local dev env.

Commands used:

```bash
uv run mcode launch sync bluevela
ssh skula@login3.bluevela.rmf.ibm.com 'bash -lc "cd /u/skula/mcode-launch && uv sync --all-extras"'

uv run mcode bench smoke \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --on bluevela \
  --db experiments/results/ab-qwen35-control-full-mellea-20260421.db

MCODE_HARNESS_EXPERIMENTS=mellea_loop_detect_v1 \
uv run mcode bench smoke \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --on bluevela \
  --db experiments/results/ab-qwen35-loop-detect-full-20260421.db

MCODE_HARNESS_EXPERIMENTS=mellea_toolkit_v1 \
uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --on bluevela \
  --db experiments/results/ab-qwen35-toolkit-first4-20260421.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling repair \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-repair-first4-20260421.db

uv run mcode bench swebench-lite \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend openai \
  --timeout 300 \
  --split test \
  --arch auto \
  --namespace swebench \
  --max-workers 4 \
  --mem-limit 8g \
  --pids-limit 512 \
  --limit 4 \
  --task-ids src/mcode/bench/fixtures/smoke-16.txt \
  --dataset princeton-nlp/SWE-bench_Verified \
  --sampling rejection \
  --sampling-budget 2 \
  --on bluevela \
  --db experiments/results/ab-qwen35-rejection-first4-20260421.db
```

I also ran first-four diagnostics for the default Mellea-first path and for loop detection. Two newer experiments, `required_arg_repair_v1` and `finalizer_success_guard_v1`, both hit infra-heavy reruns and are recorded as invalid below.

## Results

Valid runs:

| Run | Scope | Passed | Zero edit | Zero verification | Submitted | Budget exhausted | Unverified diff discarded | Wrong patch | Infra failure |
|-|-|-|-|-|-|-|-|-|-|
| Control | first 4 | 0/4 | 1 | 2 | 0 | 1 | 2 | 1 | 0 |
| `mellea_loop_detect_v1` | first 4 | 1/4 | 1 | 1 | 1 | 1 | 2 | 0 | 0 |
| `mellea_toolkit_v1` | first 4 | 0/4 | 1 | 4 | 0 | 1 | 3 | 0 | 0 |
| `--sampling repair --sampling-budget 2` | first 4 | 0/4 | 4 | 4 | 0 | 0 | 0 | 0 | 4 |
| Control | full smoke | 3/16 | 6 | 9 | 3 | 8 | 5 | 0 | 0 |
| `mellea_loop_detect_v1` | full smoke | 2/16 | 8 | 11 | 2 | 2 | 5 | 1 | 6 |

Invalid or infra-dominated reruns:

| Run | Scope | Result |
|-|-|-|
| `required_arg_repair_v1` | first 4 | `0/4`, all `infra_failure` |
| `required_arg_repair_v1` rerun | first 4 | `0/4`, all `infra_failure` |
| `finalizer_success_guard_v1` | first 4 | `0/4`, all `infra_failure` |
| `--sampling rejection --sampling-budget 2` | first 4 | `0/4`, all `infra_failure` |

## Findings

The big result here is not that a new experiment won. It is that moving onto the Mellea-first dependency path exposed a worse default baseline than the earlier mcode-specific control. The valid full-control run on this branch landed at `3/16`, far below the earlier `7/16` smoke control from the final-answer fix entry.

The logs make the failure mode pretty obvious. Qwen is still emitting a lot of malformed tool calls with missing required args, especially empty `final_answer`, empty `read_file`, and empty `run_tests` calls. That churn is generic and model-agnostic in the sense that it is a runtime integration problem, not a benchmark-specific patch heuristic. But it means the next useful work is lower-level than more turn nudges.

`mellea_loop_detect_v1` helped a little on the first four tasks, `0/4` to `1/4`, but it did not survive the full smoke. The full run regressed to `2/16` and also picked up six infra failures, so there is no case for keeping it as-is.

`mellea_toolkit_v1` was clearly worse. In the first-four run the model lost `run_tests` entirely from the tool surface and slid into more `unverified_diff_discarded` outcomes. That tells me the current swap from mcode’s tool wrapper to Mellea’s stock tool builder is not drop-in compatible for this benchmark path yet.

The sampling-based checks were also a dead end here. Repair and rejection did not rescue the malformed-call problem. In practice they either stayed flat at zero or got buried under infra failures before they could tell us anything useful.

The concrete next step is to stay Mellea-first, but aim much narrower. The best target now is malformed required-arg handling around tool invocation, especially the finalizer path. That is a cleaner bet than more prompt nudges, more phase logic, or broad tool-surface swaps.

## Files

Included in this entry:

- `research/2026-04-21-mellea-first-harness-ab/README.md`
- `research/2026-04-21-mellea-first-harness-ab/mellea-first-harness-ab-report.html`
- `research/2026-04-21-mellea-first-harness-ab/control-first4/ab-qwen35-control-first4-mellea-20260420.db`
- `research/2026-04-21-mellea-first-harness-ab/control-full/ab-qwen35-control-full-mellea-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/loop-detect-first4/ab-qwen35-loop-detect-first4-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/loop-detect-full/ab-qwen35-loop-detect-full-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/toolkit-first4/ab-qwen35-toolkit-first4-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/repair-first4/ab-qwen35-repair-first4-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/rejection-first4/ab-qwen35-rejection-first4-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/required-arg-first4/ab-qwen35-required-arg-first4-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/required-arg-first4-rerun/ab-qwen35-required-arg-first4-rerun-20260421.db`
- `research/2026-04-21-mellea-first-harness-ab/finalizer-success-first4/ab-qwen35-finalizer-success-first4-20260421.db`
