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

## Phase 7: Larger models (MiniMax M2.5, Devstral 2)

Tried scaling up from Qwen3.5-27B to larger models on 4x H100 GPUs. Both MiniMax M2.5 (230B MoE, 10B active, 80.2% Verified) and Devstral 2 (123B dense, 72.2% Verified) had broken tool calling through vLLM. Over 50% of tool calls lost their arguments entirely, resulting in the agent wasting half its turns on failed calls.

| Model | Verified score | Our Live Lite score | Tool call failure rate |
|-|-|-|-|
| MiniMax M2.5 (vLLM v0.17.0) | 80.2% | 7.3% | 51% |
| MiniMax M2.5 (vLLM nightly) | 80.2% | 5.3% | 53% |
| Devstral 2 (vLLM v0.17.1) | 72.2% | 2.7% | 53% |

The root cause is vLLM's tool call parsers (minimax_m2, mistral). They extract the function name from the model's output but drop the argument JSON during streaming token accumulation. This is a known issue across multiple vLLM GitHub issues (#31501, #29192, #22975). The models themselves can generate code fine (MiniMax raw mode produced 14x more valid diffs than Qwen), but the tool calling pipeline breaks them.

The top SWE-bench frameworks (SWE-agent, smolagents, OpenHands) all handle tool calling at the application level rather than relying on vLLM's parsers. We built a text-based tool calling mode (`MELLEA_TEXT_TOOLS=1`) that embeds tool schemas in the prompt and parses tool calls from the model's text output using XML-delimited JSON blocks. This bypasses vLLM entirely and works with any model.

With text-based tool calling, MiniMax M2.5 tool call failures dropped from 51% to 0%. The model produced patches on 46% of tasks (comparable to Qwen's 47%), but the pass rate on generated patches was lower (9.4% vs Qwen's 14.9%). On Live Lite, increasing the turn budget from 15 to 100 barely helped (13/300 to 15/300), producing only 17 more patches but at roughly the same success rate. The 80.2% Verified score that MiniMax M2.5 achieves with OpenHands is clearly a scaffold gap, not a model gap.

On the full SWE-bench Verified benchmark (500 tasks), the corrected text-tool setup plus the sandboxed bash tool reached 165/500 = 33.0%. This supersedes the earlier 156/500 = 31.2% run after we fixed a shell-tool bug where piped commands could report false `PASSED` statuses without `pipefail`. The updated result puts us at about 41.1% of OpenHands' published 80.2%.

## Phase 8: Scaffold gap analysis

The gap between our results and published benchmarks is almost entirely about the agent scaffold, not model capability or tool calling.

We studied OpenHands' CodeAct agent (which achieves 80.2% with MiniMax M2.5) and found the following key differences from our setup:

OpenHands gives the model 100 iterations. More turns alone are not enough, but timeout and loop policy were real bottlenecks on our side. On the 10-task Verified smoke slice, the progression was 3/10 at 15 turns, 4/10 at 25 turns, 5/10 at 100 turns, then 6/10 once we kept 100 turns but raised the outer ReAct timeout from 450s to 1800s and fixed malformed text-tool-call recovery. The real differences are still architectural. Our current Verified setup now includes both a bash tool and `run_tests` inside the loop, but it still lacks several things OpenHands relies on: IPython for inline Python execution, a memory condenser that compresses conversation history when context gets long, and a much richer system prompt with structured workflow guidance covering exploration, analysis, testing, implementation, and verification phases. There's also a "think" tool for explicit reasoning steps.

The implication is clear: matching top-tier benchmark scores requires rebuilding the agent scaffold to be much closer to OpenHands' architecture. The specific gaps that matter most now are stronger in-loop execution and verification flow (`run_tests`, IPython, richer execution flow), context condensation (manages the 32k window across long trajectories), and the structured problem-solving workflow in the system prompt.

## Phase 9: Verified smoke reruns after runtime fixes

After the full 33.0% Verified run, we focused on runtime correctness and agent-loop behavior rather than immediately rerunning all 500 tasks.

The main fixes were:

- added real `run_tests` usage inside the text-tool loop and blocked `bash` from being used for raw pytest invocations
- fixed the visible working directory so the model sees `/testbed` instead of host temp paths
- moved SWE-bench command execution into the task container with the same `testbed` environment as evaluation
- added narrow repair for malformed `<tool_call>` JSON instead of silently discarding near-miss calls
- raised the outer ReAct timeout from 450s to 1800s so model latency would not dominate the result

On the 10-task Verified smoke slice (all Astropy tasks), the progression was:

| Setting | Result |
|-|-|
| budget 15 | 3/10 = 30% |
| budget 25 | 4/10 = 40% |
| budget 100 | 5/10 = 50% |
| budget 100 + parser repair + timeout 1800 | 6/10 = 60% |

The important conclusion is that `MCODE_REACT_TIMEOUT=450` was a real bottleneck. Once the model-serving path was stable and we raised the outer timeout, the same 10-task slice gained one more pass and reached 60%. That is still nowhere near the published 80.2% OpenHands figure, but it is materially better than the earlier smoke runs and gives us a more honest baseline for the next larger rerun.

We also reran that same 10-task, 100-turn Astropy smoke through the fully `uv`-managed server path after pushing the newer reasoning preservation, live condensation, repo customization, and line-range editing work. That `uv`-backed rerun matched the previous best at 6/10 = 60.0%, not better and not worse. The four remaining failures on that slice were `astropy__astropy-13033`, `astropy__astropy-13398`, `astropy__astropy-13977`, and `astropy__astropy-14182`. The postmortem on those four showed three deterministic scaffold failures, submitting the wrong verified target (`13033`), submitting an insufficiently verified patch (`13977`), and submitting a write-side-only fix without checking the task-default round-trip behavior (`14182`), plus one large-task trajectory failure with no patch produced at all (`13398`).

## Where things stand

Best result with Qwen3.5-27B: 84/300 = 28.0% on SWE-bench Lite, roughly 28-32/300 (9-11%) on Live Lite. Best result with MiniMax M2.5 on Live Lite (text tools, budget=100): 15/300 = 5.0%. Best result with MiniMax M2.5 on full SWE-bench Verified so far: 165/500 = 33.0%. Best post-fix Verified smoke result on the 10-task Astropy slice: 6/10 = 60.0% with budget=100 and `MCODE_REACT_TIMEOUT=1800`.

The text-based tool calling infrastructure works and is model-agnostic. The mid-budget nudge remains the most effective single intervention (+3.7pp on Live with Qwen), and the newer MiniMax smoke runs show that timeout and verification wiring were also real bottlenecks. But the gap between our 33.0% full Verified result, even our improved 60% smoke on a narrow slice, and OpenHands' 80.2% is still fundamentally about the scaffold architecture, not model capability or tool calling mechanics. Closing that gap requires giving the agent stronger in-loop verification, better context management, and richer task guidance.
