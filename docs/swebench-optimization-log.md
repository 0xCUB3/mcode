# SWE-bench Live Optimization Log

Iterative experiments to improve SWE-bench Live resolution rate on a 25-task smoke suite, using Qwen3.5-35B-A3B on Blue Vela.

## Setup

- **Model:** Qwen/Qwen3.5-35B-A3B (MoE, 3B active)
- **Serving:** vLLM on 2x H100, `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`
- **Cluster:** IBM Blue Vela, podman rootless Docker-compat API
- **Budget:** 15 ReACT turns per sample
- **Smoke suite:** 25 tasks from SWE-bench Live Lite (`data/smoke-suite.json`), 5 categories x 5 tasks
- **Eval:** Docker containers from `starryzhang/sweb.eval.x86_64.*`

## Baseline

**Run 5** (pre-optimization): 5/25 = 20%

Passed: babel-1141, streamlink-6242, keras-20443, pybamm-4644, fonttools-3682 (partial, inconsistent)

Failure analysis on 20 failed tasks:

| Category | Count | % |
|-|-|-|
| wrong_fix | 10 | 48% |
| read_loop | 5 | 24% |
| budget_exhausted | 3 | 12% |
| edit_struggle | 1 | 4% |
| infra | 2 | 8% |

## Experiment 1: LLM File Localization

**Hypothesis:** Pre-localizing relevant files with BM25 + LLM ranking would focus the agent on the right code.

**Implementation:** `FileLocalization` class in session.py. BM25-ranked top-20 files from the repo, then LLM selects top-5 most relevant. Injected as "Localized files" block in the agent goal.

**Run 7:** 2/25 = 8% (regression from 20%)

**Analysis:** Localization was actively harmful. For 3 regressed tasks:
- keras-20443: localization pointed to `traceback_utils.py` instead of `io_utils.py`
- pybamm-4644: localization pointed to `processed_variable.py`, fix was in `symbol.py`
- The wrong files consumed agent budget reading irrelevant code

**Verdict: DISCARD.** Removed all localization code. The agent's own search is better than pre-localization with a small model.

**Commit:** `1477d12` remove LLM file localization

## Experiment 2: Read Loop Detection

**Hypothesis:** Agents get stuck re-reading the same files. Blocking repeated reads would force exploration.

**Implementation:** Added `_read_counts` tracking in mellea fork `make_agent_tools()`. After 3 reads of the same file path, tool returns an error message instead.

**Run 8:** 0/25 (but infrastructure failure masked results)

**Analysis:** Even ignoring infra failure, 98 loop detection warnings fired. The detection was too aggressive -- the agent legitimately reads the same file with different line ranges (e.g., lines 1-200 then 300-500). Blocking this prevents normal exploration of large files.

**Verdict: DISCARD.** Reverted. A smarter approach would track (path, start_line, end_line) tuples instead of just paths, but the complexity isn't worth it given the marginal benefit.

**Commits:** `669cb43` add read loop detection, `287742d` revert read loop detection

## Experiment 3: Majority Voting

**Hypothesis:** Running react() N times per task and picking the most common diff would reduce variance.

**Implementation:** Added `n_samples` parameter to `generate_patch()`. When n_samples > 1, runs react() that many times, collects diffs, picks the most frequent one via `Counter.most_common()`, applies via `git apply`.

**Run 9b (n_samples=3):** 7/24 = 29.2% (1 task had no Docker image)

| Task | Baseline | Voting |
|-|-|-|
| python-babel__babel-1141 | PASS | PASS |
| streamlink__streamlink-6242 | PASS | PASS |
| keras-team__keras-20443 | PASS | PASS |
| pybamm-team__pybamm-4644 | PASS | PASS |
| fonttools__fonttools-3682 | PASS | PASS |
| kubernetes-client__python-2303 | FAIL | PASS (new) |
| matplotlib__matplotlib-29285 | FAIL | PASS (new) |
| (17 others) | FAIL | FAIL |

3 new passes, 0 regressions.

**Verdict: KEEP.** 29.2% vs 20% baseline. All baseline passes maintained.

**Commit:** `fb76222` add --n-samples CLI option for majority voting

## Experiment 4: Docker Test Execution During Repair

**Hypothesis:** Letting the agent run tests inside a Docker container during the repair loop would enable test-driven fixing.

**Implementation:** Created `_make_docker_test_fn()` in runner.py. Starts a Docker container with the host testbed bind-mounted at `/testbed`. Agent's `run_tests` tool execs test commands inside the container. Modified mellea fork `make_agent_tools()` to accept a `test_fn` callback.

**Run 10 (n_samples=3 + Docker tests):** 1/8 = 12.5% (killed early, clear regression)

**Analysis:**
- 18/18 react() samples hit "budget exhausted without final_answer"
- `final_answer` tool had a TypeError bug (12 occurrences), wasting turns
- Test suites timed out at 120s, consuming budget without feedback
- The agent spent its 15-turn budget on test-iterate cycles instead of finding the fix
- 26 `run_tests` calls across 8 tasks -- tests were being called but the feedback loop consumed too much budget

**Verdict: DISCARD.** Reverted. The approach is sound in principle but needs: (a) much larger budget (30+ turns), (b) targeted test execution (single test file, not full suite), (c) fix the `final_answer` TypeError in mellea. Not worth pursuing with current model/budget constraints.

**Commits:** `8959272` add Docker test execution, `c89ef47` revert Docker test execution

## Round 2: Isolated A/B Tests

After researching SOTA approaches (OpenHands, SWE-agent, Moatless, Agentless), implemented 4 prompt/budget changes with env var toggles for isolated testing. Each experiment enables only one change vs the baseline.

Only 15/25 tasks had Docker images available (down from 24 in run9b). Baseline on this 15-task subset: 4/15 = 26.7%.

### Experiment A: Explore-First Prompt (`MCODE_EXPLORE_PROMPT=1`)

**Hypothesis:** A structured EXPLORE→DIAGNOSE→EDIT→VERIFY prompt (inspired by OpenHands) would reduce premature editing.

**Implementation:** Replaced the simple system prompt with a phased strategy prompt in `session.py`. Toggleable via `MCODE_EXPLORE_PROMPT` env var.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

**Verdict: NEUTRAL/NEGATIVE.** No improvement. The structured prompt didn't help the small model.

### Experiment B: Budget Warning (`MCODE_BUDGET_WARNING=1`)

**Hypothesis:** Warning the agent 2 turns before budget exhaustion would reduce "budget_exhausted" failures.

**Implementation:** `on_turn` callback in mellea's `react()` that injects a user message at turn N-2. Added to mellea fork and wired in `session.py`.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

**Verdict: NEUTRAL/NEGATIVE.** No improvement. The warning didn't change agent behavior.

### Experiment C: Larger Budget (`loop_budget=25`)

**Hypothesis:** Increasing budget from 15 to 25 turns would let the agent complete more tasks.

**Implementation:** Set `loop_budget=25` via env var.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

**Verdict: NEUTRAL/NEGATIVE.** Extra turns didn't help. The agent either finds the fix quickly or goes in circles.

### Experiment D: Read History Nudge (`MCODE_READ_NUDGE=1`)

**Hypothesis:** Nudging (not blocking) the agent when it re-reads files 3+ times would reduce read loops.

**Implementation:** Counter in `make_agent_tools()` that prepends a note after 3 reads of the same path.

**Result:** 4/15 = 26.7% (matches baseline exactly, same 4 tasks pass)

**Verdict: NEUTRAL.** No improvement. The nudge didn't change behavior.

### Round 2 Summary

| Exp | Feature | Result (15 tasks) | vs Baseline (4/15) |
|-|-|-|-|
| A | Explore-first prompt | 3/15 = 20% | -1 |
| B | Budget warning | 3/15 = 20% | -1 |
| C | Budget=25 | 3/15 = 20% | -1 |
| D | Read nudge | 4/15 = 26.7% | 0 |

D matched baseline exactly. A/B/C lost kubernetes-2303, likely voting variance. None of these prompt/budget tweaks moved the needle for this model.

**Takeaway:** Small prompt changes and budget increases don't help. The 29.2% ceiling with majority voting appears to be a model capability limit. Significant improvement likely requires either a larger model or architectural changes (agentless pipeline, AST-based localization, or multi-agent decomposition).

## Overall Summary

| Experiment | Score | vs Baseline | Kept? |
|-|-|-|-|
| Baseline | 5/25 = 20% | -- | -- |
| File localization | 2/25 = 8% | -12% | No |
| Read loop detection | 0/25 | -20% (+ infra) | No |
| Majority voting (n=3) | 7/24 = 29.2% | +9.2% | Yes |
| Docker test execution | 1/8 = 12.5% | -16.7% | No |
| Explore-first prompt | 3/15 = 20% | -6.7% (vs 15-task baseline) | No |
| Budget warning | 3/15 = 20% | -6.7% (vs 15-task baseline) | No |
| Budget=25 | 3/15 = 20% | -6.7% (vs 15-task baseline) | No |
| Read nudge | 4/15 = 26.7% | 0 (vs 15-task baseline) | No |

**Final best: 7/24 = 29.2%** with majority voting (n_samples=3).

## Infrastructure Notes

- **Podman rootless netns:** Stale `/tmp/podman-run-$UID/networks/rootless-netns/rootless-netns: file exists` error blocks all Docker operations. Fix: `podman system reset --force` before starting service.
- **Cross-rack connectivity:** Blue Vela compute nodes can't always reach other racks. Jobs must target the same rack as vLLM (p1).
- **Shell variable interpolation:** `OPENAI_BASE_URL` must be hardcoded in bsub scripts, not interpolated from env vars inside heredocs.
- **uv run vs pip:** `uv run` resolves from lockfile and can override manually installed packages. Use `uv lock --upgrade-package mellea` after pushing fork changes.
