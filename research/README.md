# Research log

Short notes for benchmark runs and parameter sweeps. Each entry should include:

- goal (what are we optimizing for)
- exact command(s) used
- key results (tables + rendered HTML report link + source link)
- findings (plain bullets; no objective/subjective labels)

## Entries

- `2026-02-08-mbpp-oc-sweep-granite4`: MBPP OpenShift sweep (18 configs) for pass rate vs time-to-solve.
- `2026-02-08-mbpp-oc-focused-500-granite4`: MBPP OpenShift focused rerun (`samples=2,3`, `debug=0,1`, `timeout=60`, `limit=500`).
- `2026-02-09-oc-confirm-granite4`: MBPP repeated confirm runs + HumanEval spot-check on OpenShift.
- `2026-02-11-mbpp-grid-r2-granite4`: MBPP 18-config grid rerun (`samples=1,2,3`, `debug=0,1,2`, `timeout=60,90`, `limit=500`) with timeout tracking.
- `2026-02-15-mellea-first-verify-granite4`: MBPP verification of `mellea-first` refactor (`loop_budget=1,3,6`, `timeout=60`, `limit=500`) vs grid-r2 baseline.
- `2026-02-15-sofai-compare-granite4`: SOFAI (8B S1 + 32B S2) vs repair-only comparison on MBPP (`budget=3,5`, `timeout=90,120`, `limit=500`).
- `2026-02-20-benchmark-expansion-granite4`: EvalPlus, LiveCodeBench, BigCodeBench expansion sweeps (granite4, loop_budget=1,3,5, timeout=60,120).
- `2026-03-06-swebench-live-qwen35`: SWE-bench Live Lite (300 instances), Qwen3.5-35B-A3B on Blue Vela. Two runs: budget=3 (6/300, 2.0%) and budget=15 (9/300, 3.0%).
- `2026-03-31-swebench-optimization-log`: running SWE-bench optimization history, moved out of `docs/` so the long-form benchmark log lives under `research/`.
- `2026-03-31-swebench-verified-minimax25-harness-redesign`: SWE-bench Verified (500 instances), MiniMax-M2.5 on Blue Vela after the harness redesign. Final result: 187/500 (37.4%) with shared HF auth/cache and the new single-path text-react runtime.
- `2026-04-03-adapter-aware-orchestrator-contract`: design note for the next scaffold pass, plus a 16-task Blue Vela MiniMax-M2.5 diagnostic slice that validates the new scaffold metrics and terminal-reason buckets.
- `2026-04-07-gemma4-31b-diagnostic-slice`: SWE-bench Verified diagnostic slice on Blue Vela with `google/gemma-4-31B-it` over vLLM. Final valid control run: 5/16 (31.25%). An April 8 prompt-steer follow-up matched 5/16, improved tool-use metrics, worsened the safety mix, and was reverted.
- `2026-04-20-qwen36-smoke`: Qwen3.6-35B-A3B smoke on Blue Vela. Baseline and `15-2` multiturn both landed at 4/16, `15-3` improved to 5/16, and `20-2` jumped to 8/16, which is the current best observed Qwen3.6 run.
- `2026-04-18-verify-rerank-v1-ab`: Blue Vela smoke-16 A/B of controller-side reranking on Qwen and Gemma. Qwen was flat at 4/16, Gemma treatment stalled at 4/9, and the experiment was reverted.
- `2026-04-20-final-answer-selection-fix`: finalizer tool-selection fix in the Mellea ReAct driver, plus Blue Vela Qwen3.5 smoke A/Bs around generic control-loop nudges. The kept fix is small and correct, the best valid control hit 7/16, and every extra nudge regressed.
- `2026-04-21-mellea-first-harness-ab`: Mellea-first Qwen3.5 smoke A/Bs on Blue Vela. Reusing more of the forked Mellea surface exposed a weaker 3/16 control, loop detection did not recover it, the stock Mellea tool builder was not drop-in, and several malformed-call repair experiments were drowned by infra failures.
- `2026-04-21-upstream-clean-head-repair-ab`: upstream-compatible Blue Vela Qwen3.5 smoke sweep from a clean worktree. Repair sampling helped, but upstream `multiturn` was the bigger find. We now have valid multiturn full-smoke runs at 8/16, 7/16, and 5/16, plus a 15/20 by 2/3 matrix where `15-3` and `20-2` both landed at 7/16.
- `2026-04-20-little-coder-gap-analysis-untracked`: little-coder comparison plus Aider Polyglot Qwen3.6 harness run. Baseline full run was 103/225, Java 21 fixed Java from 0/47 to 30/47, C-family edit guard bypass moved C++ from 2/26 to 17/26, and the corrected score is 148/225 (65.8%). Report: `2026-04-20-little-coder-gap-analysis-report.html`.
- `2026-04-24-aider-polyglot-workspace-context-full`: full Aider Polyglot Qwen3.6 runs with generic workspace context discovery and verifier prompt cleanup. Best result so far: 159/225 (70.7%), still below little-coder's 177/225. Report: `2026-04-24-aider-polyglot-workspace-context-full/aider-polyglot-workspace-context-report.html`.



## Entry template

1. Goal / scope
2. Environment + commands
3. Key results
4. Findings

Rendered HTML link pattern (interactive + Plotly-friendly):

`https://raw.githack.com/<org>/<repo>/main/research/<entry>/<benchmark>-sweep.html`
