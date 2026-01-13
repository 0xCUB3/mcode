# Results DB (SQLite)

Default path: `experiments/results/results.db` (override with `mcode ... --db <path>`).

## Tables

- `runs`: one row per benchmark run (benchmark name + config + timestamp)
- `task_results`: one row per `(run_id, task_id)` with pass/fail + timing + captured stderr/stdout

See implementation in `src/mcode/bench/results.py`.

## Typical queries

- Per-run pass rate: `mcode results --benchmark humaneval`
- Grouped by samples (and config): `mcode results --benchmark humaneval --model <id> --compare-samples`
