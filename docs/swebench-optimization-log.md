# SWE-bench Optimization Log

We started with Qwen3.5-35B-A3B on a 25-task smoke suite, tried a bunch of things, then scaled up to Qwen3.5-27B on the full 300-task benchmarks.

## Initial setup

- **Model:** Qwen/Qwen3.5-35B-A3B (MoE, 3B active)
- **Serving:** vLLM on 2x H100, `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`
- **Cluster:** IBM Blue Vela, podman rootless Docker-compat API
- **Budget:** 15 ReACT turns per sample
- **Smoke suite:** 25 tasks from SWE-bench Live Lite (`data/smoke-suite.json`), 5 categories x 5 tasks
- **Eval:** Docker containers from `starryzhang/sweb.eval.x86_64.*`

## Baseline

**Run 5** (pre-optimization): 5/25 = 20%

Passed: babel-1141, streamlink-6242, keras-20443, pybamm-4644, fonttools-3682 (partial, inconsistent)

Looking at the 20 failures:

| Category | Count | % |
|-|-|-|
| wrong_fix | 10 | 48% |
| read_loop | 5 | 24% |
| budget_exhausted | 3 | 12% |
| edit_struggle | 1 | 4% |
| infra | 2 | 8% |

## Experiment 1: LLM file localization

Idea: pre-localize relevant files with BM25 + LLM ranking to focus the agent on the right code.

Added a `FileLocalization` class in session.py. BM25-ranked top-20 files from the repo, then LLM picks top-5 most relevant. Injected as "Localized files" block in the agent goal.

**Run 7:** 2/25 = 8% (regression from 20%)

Localization was actively harmful. For the regressed tasks:
- keras-20443: localization pointed to `traceback_utils.py` instead of `io_utils.py`
- pybamm-4644: localization pointed to `processed_variable.py`, fix was in `symbol.py`
- The wrong files ate agent budget reading irrelevant code

**Discarded.** The agent's own search is better than pre-localization with a small model.

**Commit:** `1477d12`

## Experiment 2: Read loop detection

Idea: agents get stuck re-reading the same files. Block repeated reads to force exploration.

Added `_read_counts` tracking in the mellea fork's `make_agent_tools()`. After 3 reads of the same file path, the tool returns an error instead.

**Run 8:** 0/25 (infra failure masked results, but still bad)

98 loop detection warnings fired. Way too aggressive -- the agent legitimately reads the same file with different line ranges (e.g., lines 1-200 then 300-500). Blocking this prevents normal exploration of large files.

**Discarded.** A smarter approach would track (path, start_line, end_line) tuples instead of just paths, but not worth the complexity.

**Commits:** `669cb43`, `287742d` (revert)

## Experiment 3: Majority voting

Idea: run react() N times per task and pick the most common diff to reduce variance.

Added `n_samples` parameter to `generate_patch()`. When n_samples > 1, runs react() that many times, collects diffs, picks the most frequent one via `Counter.most_common()`.

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

2 new passes, 0 regressions. Nice.

**Kept.** 29.2% vs 20% baseline.

**Commit:** `fb76222`

## Experiment 4: Docker test execution during repair

Idea: let the agent run tests inside a Docker container during the repair loop for test-driven fixing.

Created `_make_docker_test_fn()` in runner.py. Starts a container with the host testbed bind-mounted at `/testbed`. Agent's `run_tests` tool execs commands inside the container.

**Run 10 (n_samples=3 + Docker tests):** 1/8 = 12.5% (killed early, obvious regression)

What went wrong:
- 18/18 react() samples hit "budget exhausted without final_answer"
- `final_answer` tool had a TypeError bug (12 occurrences), wasting turns
- Test suites timed out at 120s, eating budget without useful feedback
- Agent spent all 15 turns on test-iterate cycles instead of finding the fix

**Discarded.** The idea is sound but needs much larger budget (30+ turns), targeted test execution (single test file, not full suite), and a fix for the `final_answer` TypeError. Not worth it with current constraints.

**Commits:** `8959272`, `c89ef47` (revert)

## Round 2: isolated A/B tests

After reading up on SOTA approaches (OpenHands, SWE-agent, Moatless, Agentless), implemented 4 prompt/budget changes with env var toggles for isolated testing. Each experiment enables only one change vs baseline.

Only 15/25 tasks had Docker images available (down from 24 in run9b). Baseline on this subset: 4/15 = 26.7%.

### A: Explore-first prompt (`MCODE_EXPLORE_PROMPT=1`)

Structured EXPLORE/DIAGNOSE/EDIT/VERIFY prompt inspired by OpenHands.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

Didn't help. The structured prompt just constrained the model.

### B: Budget warning (`MCODE_BUDGET_WARNING=1`)

`on_turn` callback in mellea's `react()` that injects a warning at turn N-2.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

No improvement at this scale.

### C: Larger budget (`loop_budget=25`)

Just gave it more turns.

**Result:** 3/15 = 20% (-1 vs baseline, lost kubernetes-2303)

Extra turns didn't help. The agent either finds the fix early or goes in circles.

### D: Read history nudge (`MCODE_READ_NUDGE=1`)

Gentle nudge (not a block) when the agent re-reads files 3+ times.

**Result:** 4/15 = 26.7% (matches baseline exactly)

Neutral. Didn't change behavior.

### Round 2 summary

| Exp | Feature | Result (15 tasks) | vs Baseline (4/15) |
|-|-|-|-|
| A | Explore-first prompt | 3/15 = 20% | -1 |
| B | Budget warning | 3/15 = 20% | -1 |
| C | Budget=25 | 3/15 = 20% | -1 |
| D | Read nudge | 4/15 = 26.7% | 0 |

A/B/C all lost kubernetes-2303, probably voting variance. None of these moved the needle.

**Bottom line:** Small prompt changes and budget increases don't help at this model size. The 29.2% ceiling with majority voting seems to be a capability limit. Need a bigger model or architectural changes to go further.

---

## Phase 2: Full SWE-bench Lite (300 tasks) with Qwen3.5-27B

Scaled up from the 25-task smoke suite to the full 300-task benchmark, and switched from Qwen3.5-35B-A3B (MoE, 3B active) to Qwen3.5-27B (dense, all 27B active).

### Setup

- **Model:** Qwen/Qwen3.5-27B (dense 27B, 1x H100 80GB)
- **Serving:** vLLM v0.17.0 on 1x H100, `--enforce-eager --tool-call-parser qwen3_coder --reasoning-parser deepseek_r1`
- **Cluster:** IBM Blue Vela (p2-r24-n2), podman rootless
- **Budget:** 15 ReACT turns (25 for C-budget25)
- **Eval:** Full SWE-bench Lite (300 tasks), prebuilt `swebench` images
- **Parallelism:** 7 shards per experiment, 5 experiments x 7 = 35 parallel processes
- **Timeout:** 450s per task

### Results

First run had 24 tasks fail with /tmp disk quota errors (podman graphroot filled 49G). Moved graphroot to NFS (`/proj/dmfexp/skula/podman/graphroot`) and reran the failures. Combined results:

| Exp | Feature | Resolved | Rate | vs Baseline |
|-|-|-|-|-|
| baseline | all toggles off, budget=15 | 81/300 | 27.0% | -- |
| A-prompt | explore-first prompt | 64/300 | 21.3% | -5.7pp |
| B-warning | budget warning at turn N-2 | 84/300 | 28.0% | +1.0pp |
| C-budget25 | budget=25 turns | 79/300 | 26.3% | -0.7pp |
| D-nudge | read history nudge | 83/300 | 27.7% | +0.7pp |

4 tasks across experiments hit the 32k context window limit.

### What we learned

**Explore-first prompt is harmful at scale too.** -5.7pp. The structured prompt constrains the model's natural problem-solving. Confirmed bad across both model sizes and task counts now.

**Budget warning is the best single tweak.** +1.0pp, 3 more tasks resolved. The warning at turn N-2 gives the model a signal to wrap up with a patch instead of continuing to explore.

**Read nudge helps a little.** +0.7pp, 2 more tasks. Gentle reminders nudge toward new exploration.

**More turns still don't help.** -0.7pp with budget=25. The model either finds the fix early or loops. Same story as Round 2.

**Model upgrade matters most.** 27.0% baseline with dense 27B vs ~20% with MoE 3B-active. All 27B parameters active per token vs only 3B makes a real difference.

### Infrastructure pain

- Podman rootless + Docker SDK needed a bunch of workarounds: `_fq_image()` for fully-qualified names, `_copy_to_container_safe()` with uid=0 tars to dodge lchown failures, `_ensure_image()` using `client.api.pull()` to bypass podman name normalization
- Container name collisions with parallel experiments, fixed with UUID suffix
- Docker Hub rate limits, switched to anonymous pulls
- Node pinning (`-m <hostname>`) to keep shards on the same node (cached images are node-local)
- /tmp filled up with 35 parallel containers, moved graphroot to NFS
- Jobs getting stuck on `wait` because podman container cleanup hangs (SIGTERM timeouts, pasta process errors). Had to kill and merge manually.

## Phase 3: SWE-bench Live Lite (300 tasks)

Ran the positive-EV experiments (baseline, B-warning, D-nudge) on SWE-bench Live Lite. These are newer GitHub issues, generally harder than the curated Lite set.

Same model/serving as Phase 2. 3 experiments x 7 shards = 21 parallel processes.

| Experiment | Score | Rate | vs Baseline |
|-|-|-|-|
| Baseline | 19/300 | 6.3% | -- |
| Budget warning | 15/300 | 5.0% | -1.3pp |
| Read nudge | 19/300 | 6.3% | 0pp |

14 tasks per experiment hit the 32k context window limit. No infrastructure failures (all 300 ran).

Live Lite is way harder than regular Lite: 6.3% vs 27.0% baseline. The budget warning that helped on Lite (+1.0pp) was actually negative on Live (-1.3pp), so it doesn't generalize to harder tasks. Read nudge was neutral on both. The model is roughly 4x worse on real-world recent issues vs the curated benchmark set.

## Phase 4: Raw model comparison (no agent)

This was the fun one. We ran Qwen3.5-27B in single-shot mode on both benchmarks -- no mellea agent, no tools, no ReAct loop. Just give the model the problem statement and a repo map and ask it to produce a unified diff. One shot, no file access, no iteration.

The question: how much does the agent framework actually matter?

| Benchmark | Agent (baseline) | Raw (no agent) | Agent advantage |
|-|-|-|-|
| SWE-bench Lite (300) | 81/300 = 27.0% | 1/300 = 0.3% | +26.7pp |
| SWE-bench Live Lite (300) | 19/300 = 6.3% | 0/300 = 0.0% | +6.3pp |

Yeah. Nearly every raw diff failed `git apply` -- corrupt formatting, wrong line numbers, hallucinated file content. The model just can't produce valid patches without actually reading the files. The one lucky pass on Lite was `django__django-14580`, presumably a simple enough change that the model guessed the right diff format.

The agent framework (tool use, ReAct loop, iterative editing) is responsible for essentially all of the benchmark performance. Without file access, you get nothing.

## Overall summary

### Phase 1: Qwen3.5-35B-A3B (MoE, 3B active), 25-task smoke suite

| Experiment | Score | vs Baseline | Kept? |
|-|-|-|-|
| Baseline | 5/25 = 20% | -- | -- |
| File localization | 2/25 = 8% | -12% | No |
| Read loop detection | 0/25 | -20% (+ infra) | No |
| Majority voting (n=3) | 7/24 = 29.2% | +9.2% | Yes |
| Docker test execution | 1/8 = 12.5% | -16.7% | No |
| Explore-first prompt | 3/15 = 20% | -6.7% (vs 15-task) | No |
| Budget warning | 3/15 = 20% | -6.7% (vs 15-task) | No |
| Budget=25 | 3/15 = 20% | -6.7% (vs 15-task) | No |
| Read nudge | 4/15 = 26.7% | 0 (vs 15-task) | No |

### Phase 2: Qwen3.5-27B (dense 27B), full SWE-bench Lite (300 tasks)

| Experiment | Score | vs Baseline | Kept? |
|-|-|-|-|
| Baseline | 81/300 = 27.0% | -- | -- |
| Explore-first prompt | 64/300 = 21.3% | -5.7pp | No |
| Budget warning | 84/300 = 28.0% | +1.0pp | Maybe |
| Budget=25 | 79/300 = 26.3% | -0.7pp | No |
| Read nudge | 83/300 = 27.7% | +0.7pp | Maybe |

### Phase 3: Qwen3.5-27B, SWE-bench Live Lite (300 tasks)

| Experiment | Score | vs Baseline |
|-|-|-|
| Baseline | 19/300 = 6.3% | -- |
| Budget warning | 15/300 = 5.0% | -1.3pp |
| Read nudge | 19/300 = 6.3% | 0pp |

### Phase 4: Raw model (no agent), both benchmarks

| Benchmark | Agent | Raw | Delta |
|-|-|-|-|
| SWE-bench Lite | 81/300 = 27.0% | 1/300 = 0.3% | +26.7pp |
| Live Lite | 19/300 = 6.3% | 0/300 = 0.0% | +6.3pp |

**Best result: 84/300 = 28.0%** on SWE-bench Lite with budget warning.

**Key takeaways:**
- Model size matters most. Dense 27B >> MoE 3B-active.
- The agent framework is everything. Without tools and iteration, the model gets 0%.
- Prompt/budget tweaks are marginal (+/-1pp). The explore-first prompt is consistently harmful.
- Live tasks are ~4x harder than curated Lite tasks.
- Further gains probably need bigger models or architectural changes (multi-agent, agentless pipeline, AST localization).

## Infrastructure notes

- **Podman rootless netns:** Stale `/tmp/podman-run-$UID/networks/rootless-netns/rootless-netns: file exists` blocks everything. Fix: `podman system reset --force`.
- **Cross-rack connectivity:** Compute nodes can't always reach other racks. Pin jobs to the same rack as vLLM.
- **Shell variable interpolation:** `OPENAI_BASE_URL` must be hardcoded in bsub scripts, not interpolated from env vars inside heredocs.
- **uv run vs pip:** `uv run` resolves from lockfile and can override manual installs. Use `uv lock --upgrade-package mellea` after pushing fork changes.
- **Podman `ignore_chown_errors`:** Need `--storage-opt ignore_chown_errors=true` on podman service to avoid lchown failures.
- **Docker SDK `.pull()` vs `.api.pull()`:** High-level `client.images.pull()` fails on podman because post-pull `images.get()` can't find images under normalized names. Use `client.api.pull()` instead.
- **Container cleanup hangs:** Podman SIGTERM/SIGKILL escalation + pasta process errors keep `wait` stuck. Kill and merge manually.
- **Docker Hub rate limits:** Use anonymous pulls (100/6hr per-IP) instead of authenticated (200/6hr per-user, shared across jobs).
