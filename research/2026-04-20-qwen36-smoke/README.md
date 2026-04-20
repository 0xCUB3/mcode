# Qwen3.6-35B-A3B smoke on Blue Vela

## Setup

- model: `Qwen/Qwen3.6-35B-A3B`
- serving: vLLM `v0.19.0`
- parser flags: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
- tensor parallel: `2`
- max model len: `262144`
- benchmark: `uv run mcode bench smoke`

## Commands

Launch profile support:

```bash
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
```

Direct server fallback used for this run:

```bash
ssh skula@login3.bluevela.rmf.ibm.com \
  'lsrun -m p1-r15-n1 bash /proj/dmfexp/skula/mcode-shared/runs/bv-61efcae2/vllm.sh'
```

Smoke command:

```bash
OPENAI_BASE_URL=http://p1-r15-n1.bluevela.rmf.ibm.com:8321/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=262144 \
uv run mcode bench smoke \
  --backend openai \
  --model Qwen/Qwen3.6-35B-A3B \
  --on bluevela \
  --db experiments/results/smoke-bluevela-qwen36-20260420.db
```

## Results

Final DB:

- `experiments/results/smoke-bluevela-qwen36-20260420.db`

Final result:

- passed: `4/16`
- pass rate: `25.0%`
- terminal reasons:
  - `submitted`: `4`
  - `budget_exhausted`: `7`
  - `unverified_diff_discarded`: `4`
  - `wrong_patch_after_verification`: `1`

Submitted tasks:

- `astropy__astropy-12907`
- `astropy__astropy-14309`
- `scikit-learn__scikit-learn-13328`
- `sphinx-doc__sphinx-8120`

Comparison against earlier Qwen3.5 smoke baselines:

- best patched Qwen3.5 smoke: `6/16`
  - DB: `experiments/results/bluevela-smoke-16-qwen35-35b-a3b-20260419-finalizer-guard.db`
  - terminal reasons: `submitted=6`, `budget_exhausted=4`, `unverified_diff_discarded=2`, `wrong_patch_after_verification=4`
- bad April 20 control: `3/16`
  - DB: `experiments/results/qwen-control.db`
  - terminal reasons: `submitted=3`, `budget_exhausted=5`, `unverified_diff_discarded=5`, `wrong_patch_after_verification=3`

Relative to the best Qwen3.5 smoke, Qwen3.6 kept:

- `astropy__astropy-12907`
- `astropy__astropy-14309`
- `scikit-learn__scikit-learn-13328`

Qwen3.6 newly added:

- `sphinx-doc__sphinx-8120`

Qwen3.6 lost:

- `astropy__astropy-13453`
- `astropy__astropy-13579`
- `sympy__sympy-13877`

## Notes

- The first `mcode launch bluevela` attempts on `normal` and `preemptable_test1` never dispatched despite free-looking H100 hosts in `bhosts -gpu`.
- `lsrun` to `p1-r15-n1` was the first path that actually started the 3.6 server on cluster hardware.
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` override support in `--on bluevela` was required for the smoke, because the fallback `lsrun` server was not registered in launch state.
- The 3.6 run beat the bad April 20 Qwen3.5 control by one task, but it did not recover the best patched Qwen3.5 score. The main regression versus the best 3.5 run was more `budget_exhausted` and `unverified_diff_discarded`.
