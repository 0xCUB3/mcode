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

## Entry template

1. Goal / scope
2. Environment + commands
3. Key results
4. Findings

Rendered HTML link pattern (interactive + Plotly-friendly):

`https://raw.githack.com/<org>/<repo>/main/research/<entry>/<benchmark>-sweep.html`
