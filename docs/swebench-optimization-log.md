# SWE-bench Optimization Log

We started with Qwen3.5-35B-A3B on a 25-task smoke suite, tried a bunch of things, then scaled up to Qwen3.5-27B on the full 300-task benchmarks.

## Initial setup

Qwen/Qwen3.5-35B-A3B (MoE, 3B active), served via vLLM on 2x H100 with `--tool-call-parser qwen3_xml --reasoning-parser deepseek_r1`. Running on IBM Blue Vela with podman rootless Docker-compat API. 15 ReACT turns per sample. Eval on a 25-task smoke suite from SWE-bench Live Lite (`data/smoke-suite.json`, 5 categories x 5 tasks), using Docker containers from `starryzhang/sweb.eval.x86_64.*`.

## Baseline

Run 5 (pre-optimization): 5/25 = 20%

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

Run 7: 2/25 = 8% (regression from 20%)

Localization was actively harmful. keras-20443 got pointed to `traceback_utils.py` instead of `io_utils.py`. pybamm-4644 got pointed to `processed_variable.py` when the fix was in `symbol.py`. The wrong files ate agent budget reading irrelevant code.

Discarded. The agent's own search is better than pre-localization with a small model. Commit `1477d12`.

## Experiment 2: Read loop detection

Idea: agents get stuck re-reading the same files. Block repeated reads to force exploration.

Added `_read_counts` tracking in the mellea fork's `make_agent_tools()`. After 3 reads of the same file path, the tool returns an error instead.

Run 8: 0/25 (infra failure masked results, but still bad)

98 loop detection warnings fired. Way too aggressive. The agent legitimately reads the same file with different line ranges (e.g., lines 1-200 then 300-500). Blocking this prevents normal exploration of large files.

Discarded. A smarter approach would track (path, start_line, end_line) tuples instead of just paths, but not worth the complexity. Commits `669cb43`, `287742d` (revert).

## Experiment 3: Majority voting

Idea: run react() N times per task and pick the most common diff to reduce variance.

Added `n_samples` parameter to `generate_patch()`. When n_samples > 1, runs react() that many times, collects diffs, picks the most frequent one via `Counter.most_common()`.

Run 9b (n_samples=3): 7/24 = 29.2% (1 task had no Docker image)

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

2 new passes, 0 regressions. Kept. Commit `fb76222`.

## Experiment 4: Docker test execution during repair

Idea: let the agent run tests inside a Docker container during the repair loop for test-driven fixing.

Created `_make_docker_test_fn()` in runner.py. Starts a container with the host testbed bind-mounted at `/testbed`. Agent's `run_tests` tool execs commands inside the container.

Run 10 (n_samples=3 + Docker tests): 1/8 = 12.5% (killed early, obvious regression)

18/18 react() samples hit "budget exhausted without final_answer". The `final_answer` tool had a TypeError bug (12 occurrences) wasting turns. Test suites timed out at 120s eating budget without useful feedback. Agent spent all 15 turns on test-iterate cycles instead of finding the fix.

Discarded. The idea is sound but needs much larger budget (30+ turns), targeted test execution (single test file, not full suite), and a fix for the `final_answer` TypeError. Commits `8959272`, `c89ef47` (revert).

## Round 2: isolated A/B tests

After reading up on SOTA approaches (OpenHands, SWE-agent, Moatless, Agentless), implemented 4 prompt/budget changes with env var toggles for isolated testing. Each experiment enables only one change vs baseline.

Only 15/25 tasks had Docker images available (down from 24 in run9b). Baseline on this subset: 4/15 = 26.7%.

A: Explore-first prompt (`MCODE_EXPLORE_PROMPT=1`). Structured EXPLORE/DIAGNOSE/EDIT/VERIFY prompt inspired by OpenHands. 3/15 = 20%, lost kubernetes-2303. The structured prompt just constrained the model.

B: Budget warning (`MCODE_BUDGET_WARNING=1`). Injects a warning at turn N-2 via `on_turn` callback. 3/15 = 20%. No improvement at this scale.

C: Larger budget (`loop_budget=25`). 3/15 = 20%. The agent either finds the fix early or goes in circles.

D: Read history nudge (`MCODE_READ_NUDGE=1`). Gentle nudge when the agent re-reads files 3+ times. 4/15 = 26.7%, matches baseline exactly.

| Exp | Feature | Result (15 tasks) | vs Baseline (4/15) |
|-|-|-|-|
| A | Explore-first prompt | 3/15 = 20% | -1 |
| B | Budget warning | 3/15 = 20% | -1 |
| C | Budget=25 | 3/15 = 20% | -1 |
| D | Read nudge | 4/15 = 26.7% | 0 |

A/B/C all lost kubernetes-2303, probably voting variance. None of these moved the needle. The 29.2% ceiling with majority voting seems to be a capability limit at this model size.

---

## Phase 2: Full SWE-bench Lite (300 tasks) with Qwen3.5-27B

Scaled up from the 25-task smoke suite to the full 300-task benchmark, and switched from Qwen3.5-35B-A3B (MoE, 3B active) to Qwen3.5-27B (dense, all 27B active).

Qwen/Qwen3.5-27B on 1x H100 80GB, vLLM v0.17.0 with `--enforce-eager --tool-call-parser qwen3_coder --reasoning-parser deepseek_r1`. 15 ReACT turns (25 for C-budget25), 450s timeout. Full SWE-bench Lite (300 tasks) with prebuilt `swebench` images. 7 shards per experiment, 5 experiments = 35 parallel processes.

First run had 24 tasks fail with /tmp disk quota errors (podman graphroot filled 49G). Moved graphroot to NFS (`/proj/dmfexp/skula/podman/graphroot`) and reran the failures. Combined results:

| Exp | Feature | Resolved | Rate | vs Baseline |
|-|-|-|-|-|
| baseline | all toggles off, budget=15 | 81/300 | 27.0% | -- |
| A-prompt | explore-first prompt | 64/300 | 21.3% | -5.7pp |
| B-warning | budget warning at turn N-2 | 84/300 | 28.0% | +1.0pp |
| C-budget25 | budget=25 turns | 79/300 | 26.3% | -0.7pp |
| D-nudge | read history nudge | 83/300 | 27.7% | +0.7pp |

4 tasks across experiments hit the 32k context window limit.

Explore-first prompt confirmed harmful at scale too, dropping 5.7pp. Budget warning was the best single tweak at +1.0pp (3 more tasks resolved) by giving the model a signal to wrap up. Read nudge helped a little at +0.7pp. More turns still didn't help, same as Round 2.

The model upgrade mattered most. 27.0% baseline with dense 27B vs ~20% with MoE 3B-active. Having all 27B parameters active per token vs only 3B makes a real difference.

Infrastructure was painful. Podman rootless + Docker SDK needed workarounds for fully-qualified image names, lchown failures, name normalization. Container name collisions with parallel experiments (fixed with UUID suffix). Docker Hub rate limits. Node pinning to keep cached images accessible. /tmp filling up with 35 containers (moved graphroot to NFS). Jobs stuck on `wait` from podman cleanup hangs.

## Phase 3: SWE-bench Live Lite (300 tasks)

Ran the positive-EV experiments (baseline, B-warning, D-nudge) on SWE-bench Live Lite. These are newer GitHub issues, generally harder than the curated Lite set. Same model/serving as Phase 2. 3 experiments x 7 shards = 21 parallel processes.

| Experiment | Score | Rate | vs Baseline |
|-|-|-|-|
| Baseline | 19/300 | 6.3% | -- |
| Budget warning | 15/300 | 5.0% | -1.3pp |
| Read nudge | 19/300 | 6.3% | 0pp |

14 tasks per experiment hit the 32k context window limit. No infrastructure failures.

Live Lite is way harder than regular Lite: 6.3% vs 27.0% baseline. The budget warning that helped on Lite (+1.0pp) was actually negative on Live (-1.3pp), so it doesn't generalize to harder tasks. Read nudge was neutral on both. The model is roughly 4x worse on real-world recent issues vs the curated set.

## Phase 4: Raw model comparison (no agent)

We ran Qwen3.5-27B in single-shot mode on both benchmarks, no mellea agent, no tools, no ReAct loop. Just the problem statement and a repo map, ask for a unified diff, one shot. How much does the agent framework actually matter?

| Benchmark | Agent (baseline) | Raw (no agent) | Agent advantage |
|-|-|-|-|
| SWE-bench Lite (300) | 81/300 = 27.0% | 1/300 = 0.3% | +26.7pp |
| SWE-bench Live Lite (300) | 19/300 = 6.3% | 0/300 = 0.0% | +6.3pp |

Nearly every raw diff failed `git apply` with corrupt formatting, wrong line numbers, hallucinated file content. The model can't produce valid patches without actually reading the files. The one lucky pass on Lite was `django__django-14580`, presumably simple enough that the model guessed the right diff format.

The agent framework is responsible for essentially all of the benchmark performance. Without file access you get nothing.

## Phase 5: Autoresearch on SWE-bench Live Lite

Systematic iteration on Live Lite, testing one change at a time against the baseline. We dug into the failure modes first: 80% of no-patch failures (127 out of 159) made zero edit calls. The agent spent its entire budget reading and searching without ever committing to a change. That became the main target.

| Experiment | Score | vs Baseline |
|-|-|-|
| Baseline | 19/300 = 6.3% | -- |
| Mid-budget edit nudge (turn N/2) + budget warning | 30/300 = 10.0% | +3.7pp |
| + repo map 4096 tokens (was 2048) | 32/300 = 10.7% | +4.4pp |
| + "don't delete definitions" prompt guard | 2/300 = 0.7% | -5.6pp (DISCARD) |
| budget 15->20 | 30/300 = 10.0% | -0.7pp (DISCARD) |
| + read nudge | 28/300 = 9.3% | -1.4pp (DISCARD) |
| self-verification retry | 27/300 = 9.0% | +2.7pp (NEUTRAL) |
| control rerun (same as best config) | 26/300 = 8.7% | +2.4pp (variance) |

Run-to-run variance is about 4 tasks (26-32 across reruns of the same config). The true effect of mid-nudge + repo map 4096 lands somewhere in the 9-11% range on Live Lite.

The mid-budget nudge was by far the biggest win. At turn N/2, the agent gets a message saying "if you haven't edited yet, start editing NOW." This converted 68 no-patch tasks into actual patch attempts. 15 of those became new passes (with 4 regressions, so net +11).

Prompt constraints are catastrophic with this model. The "don't delete definitions" guard, meant to address 42 tasks where the agent broke test imports, dropped from 10.7% to 0.7%. The explore-first structured prompt dropped 5.7pp in Phase 2. Qwen3.5-27B does best with minimal instructions and freedom to act.

More budget doesn't help even with mid-nudge. Budget=20 with nudge at turn 10 scored the same as budget=15 with nudge at turn 7. Stacking nudges (read nudge + mid-nudge) scored worse than mid-nudge alone.

Self-verification retry (ask the LLM "does this patch fix the issue?" and retry on "no") was neutral. The model makes the same mistakes the second time around. The verification call itself fired on about 20/300 tasks, but the retries didn't produce better patches.

## Phase 6: Mellea framework changes

Tested three changes to the mellea agent framework (the fork), each independently on top of the best config.

| Experiment | Score | vs Best (~9-11%) |
|-|-|-|
| Context compression (summarize old tool outputs at turn N/2) | 10/300 = 3.3% | regression (DISCARD) |
| File position echo (show context around edit) | 31/300 = 10.3% | neutral |
| Fuzzy edit matching (whitespace-normalized fallback) | 32/300 = 10.7% | neutral |

Context compression actively hurts. The model needs the full tool output history to reason about what it already tried and what it found. Compressing old results removes information it relies on for later turns.

File position echo (showing surrounding lines after each edit) was neutral at 31/300. The model doesn't benefit from seeing the edit context since it already knows what it changed.

Fuzzy edit matching (try whitespace-normalized substring match when exact match fails) was neutral at 32/300. The edit tool's existing exact-match + error message is already good enough since the model can usually fix its whitespace on retry.

None of the framework changes improved over the base config. The mellea tools are already well-designed for this model size.

## Where things stand

Best result: 84/300 = 28.0% on SWE-bench Lite, roughly 28-32/300 (9-11%) on Live Lite.

The biggest factor is model capability. Dense 27B performed way better than MoE 3B-active, and the agent framework accounts for essentially all performance over raw generation. The mid-budget nudge is the most effective prompt intervention we found (+3.7pp on Live), while the explore-first prompt is consistently harmful. Live tasks are about 4x harder than the curated Lite tasks. Further gains at this model size seem unlikely from prompt or tool tweaks alone. The ceiling is the model itself.

## Infrastructure notes

Stale `/tmp/podman-run-$UID/networks/rootless-netns/rootless-netns: file exists` blocks all podman operations. Fix with `podman system reset --force`. Compute nodes can't always reach other racks, so pin jobs to the same rack as vLLM. `OPENAI_BASE_URL` must be hardcoded in bsub scripts, not interpolated inside heredocs. `uv run` resolves from lockfile and can override manual installs; use `uv lock --upgrade-package mellea` after pushing fork changes. Need `--storage-opt ignore_chown_errors=true` on podman service for lchown. High-level `client.images.pull()` fails on podman; use `client.api.pull()` instead. Podman SIGTERM/SIGKILL escalation + pasta process errors keep `wait` stuck; kill and merge manually. Use anonymous Docker Hub pulls (100/6hr per-IP) instead of authenticated (200/6hr per-user, shared across jobs).
