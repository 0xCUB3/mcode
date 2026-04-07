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
- `2026-04-07-gemma4-31b-diagnostic-slice`: SWE-bench Verified diagnostic slice on Blue Vela with `google/gemma-4-31B-it` over vLLM. Final valid run: 5/16 (31.25%) after fixing the podman service launcher.

## Entry template

1. Goal / scope
2. Environment + commands
3. Key results
4. Findings

Rendered HTML link pattern (interactive + Plotly-friendly):

`https://raw.githack.com/<org>/<repo>/main/research/<entry>/<benchmark>-sweep.html`
