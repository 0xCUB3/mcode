# Adapter-aware orchestrator contract

This note is the design center for the next scaffold pass. The benchmark is still useful, but it is no longer the product target. The target is a local coding agent with a small kernel that can sit on top of a base model and a handful of tool-family adapters.

The best documented full SWE-bench Verified run is 187/500, or 37.4%, in `research/2026-03-31-swebench-verified-minimax25-harness-redesign/README.md`. That run was good enough to keep, but the remaining failures are mostly scaffold failures such as budget exhaustion, late or missing verification, and patches that looked plausible without actually closing the task. That is exactly the kind of signal we want from the benchmark. It tells us where the orchestrator is weak.

I also do not want to build a benchmark-first maze of special cases. The reusable runtime belongs in `mellea`. The dataset loading, scoring, cluster launchers, and result reporting belong in `mcode`. What changes now is the design center inside the reusable scaffold.

## What the kernel owns

The kernel owns the state that has to stay true no matter which model or adapter happens to be underneath it.

At minimum, that state is:

- current task and goal text
- current phase in the solving loop
- event log for tool calls, tool results, and termination
- whether the repo has been edited
- whether verification has been attempted
- whether verification has succeeded
- submission eligibility and the reason it is blocked
- lightweight counters that describe loop quality, not just end results

This list might change, but those counters matter because pass rate alone hides the failure shape. If a run burned half its budget before the first edit, or reached the end with a dirty unverified diff, the orchestrator should say that directly.

## Canonical phase machine

The solving loop has one canonical path:

`diagnose -> edit -> verify -> submit`

`diagnose` means the agent is still narrowing the problem and has not committed to a code change. The only acceptable exit is a concrete edit.

`edit` means the agent has changed the repo and now owes the system evidence. More searching can still happen, but it is in service of fixing or finishing the patch, not reopening the task from scratch.

`verify` means the agent is running the cheapest checks that can falsify its current patch. Verification is the gate between editing and submission.

`submit` means the agent has either met the verification requirement or has no patch worth keeping. The runtime, not the prompt, decides whether submission is allowed.

There can be loops inside a phase. A failed test can push the run back toward edit. A bad early hypothesis can keep the run in diagnose longer. But the phase labels themselves should stay stable. The point is to make the state legible enough that both the model and the harness can act on it.

## Capability families

The runtime should stop thinking in terms of a flat bag of tools. It should instead think in terms of capability families.

For the next pass, the families are:

- repository exploration: `read`, `search`, `find`, optional repo summary
- editing: `edit`
- verification: `run_tests` and closely related task checks
- shell escape hatch: `bash`
- optional planning or summarization later, only if it earns a permanent spot

A family is a routing hint and a policy boundary. The orchestrator does not need to care whether repository exploration is implemented by one adapter or a future model-specific tool head. It only needs to know that the model is asking for repository exploration, and that this request is or is not appropriate in the current phase.

That matters for the likely long-term direction. If the local agent eventually gains adapter-backed inference for some families, the runtime already has the seam. If it does not, the same family request can fall back to the ordinary bundled tools without changing the loop contract.

## Verification rules

Verification is the strongest boundary in this design. A patch is not eligible for submission just because it exists. A patch is eligible only when one of these is true:

1. the task has no meaningful local verification path and the runtime records that explicitly, or
2. the agent has attempted the default verification path, or
3. the agent has attempted a narrower verification command that the runtime accepts as sufficient for the current task state

The runtime should also distinguish between:

- verification attempted
- verification succeeded
- verification failed
- verification skipped because no sane local check exists

That distinction is important because the benchmark failures are not all the same. An unverified diff discarded at budget is different from a verified but wrong patch. Both are failures, but they point at different scaffold problems.

## Capability routing contract

The routing contract should stay small.

A model turn may request a capability family. The orchestrator then decides how to satisfy that request:

1. direct family route, if an adapter-aware implementation exists
2. bundled tool route, if the family maps to ordinary tools in the current runtime
3. blocked route, if the request violates the current phase or safety rules

The model does not get to bypass the family boundary by prompt habit. If it asks for shell in a situation where verification should go through the verification family, the runtime should redirect or block it.

This is where the benchmark remains useful. If the model keeps trying to do verification through shell, or keeps wandering in repository exploration after a patch already exists, the runtime counters should make that visible.

## Fallback when no adapter-aware runtime exists

The first implementation should not wait for real adapter routing. If no adapter-aware backend exists, the runtime should:

- keep the same phase machine
- keep the same verification gate
- keep the same family labels in state and metrics
- satisfy family requests through the existing bundled tools
- record that the run used fallback routing

That gives us a clean seam without inventing a second architecture. The loop behavior stays the same; only the route behind a family changes.

## Kernel versus modular edges

The kernel should stay opinionated and small.

Kernel responsibilities:

- runtime state representation
- canonical phase transitions
- capability-family routing
- verification gating
- submission rules
- loop-quality counters and terminal reasons

Modular edges:

- benchmark dataset loaders and runners
- backend and model wiring
- repo customization
- telemetry and reports
- cluster scripts
- future adapter router implementations

This keeps the core small enough to reason about. It also avoids the trap of building a grand plugin framework before the kernel contract is stable.

## Loop-quality metrics that belong in the run record

The benchmark DB should carry scaffold-native metrics for each task and for each run summary. The minimum set for this phase is:

- turns to first edit
- turns to first verification attempt
- zero-edit tasks
- zero-verification tasks
- malformed tool-call recoveries
- blocked submissions
- terminal reason bucket

The terminal reason bucket should at least cover:

- `budget_exhausted`
- `unverified_diff_discarded`
- `wrong_patch_after_verification`
- `infra_failure`
- `submitted`

The point is to stop inferring scaffold behavior from logs after the fact. If a run is bad, the DB should already explain how it was bad.

## Benchmark ladder

We should stop using the full 500-task Verified run as the default inner loop.

The ladder should be:

- tiny smoke slice for basic regressions
- medium diagnostic slice with a mix of failure modes
- full Verified 500-task run only for milestone checkpoints

A medium slice is where scaffold work should mostly live. It is large enough to expose control-loop mistakes and still quick enough to rerun after changes.

## Why this stays minimal for now

OpenHands is a useful reference because it keeps runtime state and action-to-observation boundaries explicit. That part is worth copying at the boundary level. Its framework size is not. OpenHands earns its weight because it is trying to be a whole system.

Pi and Codex-style harnesses are useful for the opposite reason. They show that a compact, opinionated kernel can carry a lot of real work if the loop is disciplined and the session state is durable.

OpenAgent-style planner-executor systems are interesting, but they solve a later problem. Once the kernel contract is stable, more elaborate coordination might make sense. Right now it would just multiply abstractions before the basic loop is solid.

That is why this phase is subtraction-first. The benchmark already told us where the real losses are. We do not need more roles, more buses, or more benchmark-specific knobs. We need a smaller kernel that tells the truth about phase, verification, routing, and termination.

## Immediate code implications

In `mellea`, the reusable side should expose capability-family concepts and runtime-state helpers without blowing up the public surface.

In `mcode`, the benchmark-facing side should consume that contract, persist the new loop-quality metrics, and report terminal reasons in the results DB.

The benchmark remains the measuring stick, but the scaffold changes should now be judged by a different question: does this make the local coding agent more legible, more verifiable, and easier to route by capability family?

## Follow-up: Blue Vela medium diagnostic slice

After the contract and instrumentation landed, I ran a medium diagnostic slice on Blue Vela against `MiniMaxAI/MiniMax-M2.5`. I wanted one run that exercised the new counters and terminal-reason buckets on a mixed slice, while still staying small enough to inspect by hand.

HTML snapshot: [`diagnostic-swebench-report.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report.html) ([source](diagnostic-swebench-report.html))

### Setup

The slice used 16 Verified tasks. Ten were the Astropy smoke tasks that had already been useful for scaffold debugging, and six more came from Django, Matplotlib, Pylint, scikit-learn, Sphinx, and SymPy so the run would hit something other than the Astropy-heavy failure shape. The task list lives in `medium-diagnostic-task-ids.txt`.

I reused the running MiniMax vLLM endpoint on `http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1`. The benchmark itself finished cleanly and wrote all 16 task rows. The LSF wrapper still had to be killed after the run summary printed because the rootless podman cleanup never exited. That is the same annoying cleanup behavior that showed up in the full 500-task runs.

### Commands

Remote sync and dependency refresh:

```bash
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'htmlcov' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '.coverage*' --exclude '.bluevela-watch-*' \
  --exclude 'mnemos.db*' \
  /Users/skula/Documents/mcode/ \
  skula@login3.bluevela.rmf.ibm.com:/u/skula/mcode-scaffold-20260403/

ssh skula@login3.bluevela.rmf.ibm.com \
  'cd /u/skula/mcode-scaffold-20260403 && uv sync --extra dev --extra swebench --extra datasets'
```

Remote benchmark payload, submitted as one LSF job while reusing the running MiniMax server:

```bash
export OPENAI_BASE_URL=http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh

uv run mcode bench swebench-lite \
  --backend openai \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --n-samples 1 \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --db research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15/diagnostic.db
```

Local report generation after syncing the DB back:

```bash
uv run mcode report \
  --db-dir research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15 \
  --benchmark swebench-lite \
  --out research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report.html

uv run mcode results \
  --db-dir research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15 \
  --benchmark swebench-lite \
  --compare-configs \
  --time \
  > research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-results-summary.txt
```

### Key results

| Metric | Value |
|-|-:|
| Total tasks | 16 |
| Passed | 6 |
| Pass rate | 37.5% |
| Loop budget | 15 |
| Eval timeout | 300s |
| Benchmark job | 783893 |

Scaffold metrics from `final-summary.json`:

| Metric | Value |
|-|-:|
| Zero-edit tasks | 2 |
| Zero-verification tasks | 4 |
| Verification succeeded | 10 |
| Malformed tool-call recoveries | 19 |
| Blocked submissions | 0 |
| Avg turns to first edit | 8.21 |
| Avg turns to first verification | 9.42 |
| Budget exhausted | 2 |
| Unverified diff discarded | 4 |
| Wrong patch after verification | 4 |
| Infra failure | 0 |
| Submitted | 6 |

Passed tasks: `astropy__astropy-12907`, `astropy__astropy-13453`, `astropy__astropy-13579`, `astropy__astropy-14096`, `astropy__astropy-14309`, `sphinx-doc__sphinx-8120`.

### Findings

The run did exactly what I wanted from a medium slice. The average first edit came at turn 8.2, and the average first verification came even later at turn 9.4. That is too late for a 15-turn loop. The new counters made that obvious immediately.

The terminal-reason buckets were useful on the first try. Four tasks ended as `wrong_patch_after_verification`, four ended as `unverified_diff_discarded`, and two never reached an edit at all. The wrong-patch cases point at diagnosis and target selection problems. The unverified-discard cases point at loops that finally touched code but spent verification too late or too weakly to earn submission.

The Astropy smoke failures still line up with the older postmortem more than I would like. `astropy__astropy-13033`, `astropy__astropy-13977`, and `astropy__astropy-14182` are still in the verified-but-wrong bucket. `astropy__astropy-13398` is still an unverified discard. The instrumented run did not fix those tasks, but it made the failure shape much easier to see without reading raw logs.

The bash gating also showed up in a useful way. One trajectory tried to drift back into shell churn late in the loop, and the runtime blocked it with the new capability-family message instead of letting the model waste the last turns. That did not magically save the task, but it did keep the failure well-defined. 

There were still podman warnings on Blue Vela about rootless networking and cleanup. They did not stop the benchmark itself. They only showed up around image startup and teardown. I killed the wrapper after the run summary printed because the useful work was already done and the DB was complete.

### Files

- `diagnostic-swebench-report.html` - interactive report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report.html))
- `diagnostic-results-summary.txt` - CLI summary snapshot
- `medium-diagnostic-task-ids.txt` - exact 16-task slice used for the run
- `run-bluevela-minimax25-diagnostic-b15/diagnostic.db` - results DB
- `run-bluevela-minimax25-diagnostic-b15/final-summary.json` - scaffold metrics and per-task outcomes
- `run-bluevela-minimax25-diagnostic-b15/swb-diag-m25.log` - Blue Vela benchmark log


## Follow-up: same slice after verification hardening

I reran the exact same 16-task slice after tightening the verification path and pushing the phase policy to commit earlier. This was the cleanest possible comparison. Same model, same dataset slice, same loop budget, same Blue Vela endpoint. The only meaningful change was the scaffold.

HTML snapshot: [`diagnostic-swebench-report-after-verification-hardening.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-verification-hardening.html) ([source](diagnostic-swebench-report-after-verification-hardening.html))

### Rerun command

```bash
export OPENAI_BASE_URL=http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh

uv run mcode bench swebench-lite \
  --backend openai \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --n-samples 1 \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --db research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15-after-verification-hardening/diagnostic.db
```

### Before vs after

| Metric | Before | After hardening | Delta |
|-|-:|-:|-:|
| Passed | 6 | 7 | +1 |
| Pass rate | 37.5% | 43.75% | +6.25 pts |
| Zero-edit tasks | 2 | 4 | +2 |
| Zero-verification tasks | 4 | 5 | +1 |
| Verification succeeded | 10 | 10 | 0 |
| Malformed tool-call recoveries | 19 | 15 | -4 |
| Blocked verification commands | 0 | 12 | +12 |
| Avg turns to first edit | 8.21 | 5.25 | -2.96 |
| Avg turns to first verification | 9.42 | 7.73 | -1.69 |
| Budget exhausted | 2 | 5 | +3 |
| Unverified diff discarded | 4 | 2 | -2 |
| Wrong patch after verification | 4 | 2 | -2 |
| Submitted | 6 | 7 | +1 |

Passed tasks after hardening: `astropy__astropy-12907`, `astropy__astropy-13236`, `astropy__astropy-13453`, `astropy__astropy-13579`, `astropy__astropy-14096`, `astropy__astropy-14309`, `sympy__sympy-13877`.

### What changed in behavior

The good news is that the harness got noticeably faster to commit. First edit moved from 8.2 turns down to 5.25, and first verification moved from 9.4 down to 7.7. That is the strongest signal in the rerun. The phase tightening did what it was supposed to do.

The verification hardening also did real work. The new `blocked_verification_commands` counter fired 12 times, which means the model was still trying to sneak shell wrappers and pipe-based command shaping into `run_tests`. Before this pass those trajectories would have looked more trustworthy than they deserved. Now they are explicit and measurable.

The terminal buckets improved in the places I cared about most. `unverified_diff_discarded` dropped from 4 to 2, and `wrong_patch_after_verification` dropped from 4 to 2. That is a better failure mix even though the total number of budget exhaustions went up. In other words, the loop is wasting less time pretending a weak patch is verified, but it is still fully capable of stalling out without landing a first edit on some tasks.

That tradeoff is visible in the zero-edit count. It got worse, from 2 to 4. The tighter loop is better once it commits, but it is still not reliably choosing a file fast enough on the harder misses. The next pass should attack diagnosis quality and early file selection, not verification policy again.

The rerun still had the same Blue Vela cleanup problem. The DB reached all 16 task rows and the run summary printed, but the wrapper stayed alive in podman teardown, so I killed job `785508` after the useful work was complete.

### Additional files

- `diagnostic-swebench-report-after-verification-hardening.html` - rerun report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-verification-hardening.html))
- `diagnostic-results-summary-after-verification-hardening.txt` - rerun CLI summary
- `run-bluevela-minimax25-diagnostic-b15-after-verification-hardening/diagnostic.db` - rerun results DB
- `run-bluevela-minimax25-diagnostic-b15-after-verification-hardening/final-summary.json` - rerun scaffold metrics and per-task outcomes
- `run-bluevela-minimax25-diagnostic-b15-after-verification-hardening/swb-diag-m25-rerun.log` - rerun Blue Vela log


## Follow-up: same slice after diagnosis hints

After the verification-hardening rerun, I made one more pass aimed at diagnosis quality instead of verification policy. I kept the same small-kernel stance. No new planner, no extra role split, no benchmark-only shim layer. The only new behavior was a lightweight candidate-file shortlist built from the existing BM25 repository ranking and injected into the goal prompt so the agent would start from a narrower set of files.

HTML snapshot: [`diagnostic-swebench-report-after-diagnosis-hints.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-diagnosis-hints.html) ([source](diagnostic-swebench-report-after-diagnosis-hints.html))

### Rerun command

```bash
export OPENAI_BASE_URL=http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh

uv run mcode bench swebench-lite \
  --backend openai \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --n-samples 1 \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --db research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15-after-diagnosis-hints/diagnostic.db
```

### Three-run comparison

| Metric | Baseline | After verification hardening | After diagnosis hints |
|-|-:|-:|-:|
| Passed | 6 | 7 | 8 |
| Pass rate | 37.5% | 43.75% | 50.0% |
| Zero-edit tasks | 2 | 4 | 3 |
| Zero-verification tasks | 4 | 5 | 4 |
| Verification succeeded | 10 | 10 | 11 |
| Malformed tool-call recoveries | 19 | 15 | 25 |
| Blocked verification commands | 0 | 12 | 12 |
| Avg turns to first edit | 8.21 | 5.25 | 5.92 |
| Avg turns to first verification | 9.42 | 7.73 | 7.58 |
| Budget exhausted | 2 | 5 | 3 |
| Unverified diff discarded | 4 | 2 | 2 |
| Wrong patch after verification | 4 | 2 | 3 |
| Submitted | 6 | 7 | 8 |

Passed tasks after diagnosis hints: `astropy__astropy-12907`, `astropy__astropy-13236`, `astropy__astropy-13453`, `astropy__astropy-13579`, `astropy__astropy-14096`, `astropy__astropy-14309`, `scikit-learn__scikit-learn-13328`, `sphinx-doc__sphinx-8120`.

### What changed in behavior

This pass helped the benchmark in the way I hoped. The pass count moved again, from 7 to 8, and the pass rate on the 16-task slice reached 50%. That is only one task, but on a fixed slice this small it still matters because the surrounding metrics moved in the same direction.

The most useful change is that the diagnosis pass backed off the worst regression from the verification hardening run. Zero-edit tasks dropped from 4 to 3, and budget-exhausted tasks dropped from 5 to 3. That does not mean diagnosis is solved, but it does mean the shortlist is helping the model commit more often instead of freezing out.

The average first edit got slightly worse than the verification-only rerun, 5.92 instead of 5.25, but it stayed far better than the original 8.21. First verification improved a bit again, from 7.73 to 7.58. That combination suggests the shortlist is not making the model reckless. It is still getting to verification earlier than the baseline, while recovering some of the diagnosis quality that the stricter loop had lost.

The remaining weakness is precision after verification. `wrong_patch_after_verification` ticked back up from 2 to 3, and malformed tool-call recoveries jumped from 15 to 25. So this pass improved localization enough to get one more task over the line, but it also made some trajectories noisier. The next pass should probably focus on keeping the diagnosis shortlist while reducing tool-call churn and improving target precision inside the chosen file.

Just like the earlier runs, the benchmark DB completed and the run summary printed before the podman teardown finished. I killed job `788942` after the useful work was already on disk.

### Additional files

- `diagnostic-swebench-report-after-diagnosis-hints.html` - third-run report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-diagnosis-hints.html))
- `diagnostic-results-summary-after-diagnosis-hints.txt` - third-run CLI summary
- `run-bluevela-minimax25-diagnostic-b15-after-diagnosis-hints/diagnostic.db` - third-run results DB
- `run-bluevela-minimax25-diagnostic-b15-after-diagnosis-hints/final-summary.json` - third-run scaffold metrics and per-task outcomes
- `run-bluevela-minimax25-diagnostic-b15-after-diagnosis-hints/swb-diag-m25-diagnosis.log` - third-run Blue Vela log


## Follow-up: same slice after within-file hints and churn reduction

The next pass kept the diagnosis shortlist, but added two smaller changes on top of it. First, the prompt now includes likely definitions inside the shortlisted files instead of only file paths. Second, the runtime gets stricter about repeated discovery churn, so once the model is clearly looping on search-heavy diagnosis it loses the broad tools sooner and is pushed back toward `read_file` and `edit`.

HTML snapshot: [`diagnostic-swebench-report-after-churn-reduction.html`](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-churn-reduction.html) ([source](diagnostic-swebench-report-after-churn-reduction.html))

### Rerun command

```bash
export OPENAI_BASE_URL=http://p3-r13-n2.bluevela.rmf.ibm.com:8321/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
source /u/skula/.config/mcode/hf-env.sh

uv run mcode bench swebench-lite \
  --backend openai \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --loop-budget 15 \
  --timeout 300 \
  --mem-limit 4g \
  --pids-limit 512 \
  --n-samples 1 \
  --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt \
  --db research/2026-04-03-adapter-aware-orchestrator-contract/run-bluevela-minimax25-diagnostic-b15-after-churn-reduction/diagnostic.db
```

### Four-run comparison

| Metric | Baseline | After verification hardening | After diagnosis hints | After churn reduction |
|-|-:|-:|-:|-:|
| Passed | 6 | 7 | 8 | 9 |
| Pass rate | 37.5% | 43.75% | 50.0% | 56.25% |
| Zero-edit tasks | 2 | 4 | 3 | 2 |
| Zero-verification tasks | 4 | 5 | 4 | 4 |
| Verification succeeded | 10 | 10 | 11 | 12 |
| Malformed tool-call recoveries | 19 | 15 | 25 | 10 |
| Blocked verification commands | 0 | 12 | 12 | 17 |
| Avg turns to first edit | 8.21 | 5.25 | 5.92 | 5.50 |
| Avg turns to first verification | 9.42 | 7.73 | 7.58 | 6.42 |
| Budget exhausted | 2 | 5 | 3 | 2 |
| Unverified diff discarded | 4 | 2 | 2 | 2 |
| Wrong patch after verification | 4 | 2 | 3 | 3 |
| Submitted | 6 | 7 | 8 | 9 |

Passed tasks after churn reduction: `astropy__astropy-12907`, `astropy__astropy-13236`, `astropy__astropy-13453`, `astropy__astropy-13579`, `astropy__astropy-14096`, `astropy__astropy-14309`, `scikit-learn__scikit-learn-13328`, `sphinx-doc__sphinx-8120`, `sympy__sympy-13877`.

### What changed in behavior

This was the cleanest improvement since the first verification hardening pass. The slice moved from 8 to 9 solved tasks, which puts the same 16-task diagnostic set at 56.25%. More importantly, the supporting metrics improved in the same direction instead of trading one failure mode for another.

The biggest win is churn. Malformed tool-call recoveries dropped from 25 to 10, which is a much better sign than the raw pass count by itself. The model is still trying blocked verification commands, in fact the count rose from 12 to 17, but that is now mostly a guardrail firing instead of general trajectory noise. The runtime is spending less time recovering from malformed calls and more time staying on the structured path.

Within-file precision helped enough to land `sympy__sympy-13877`, which had been a verified-but-wrong case in the earlier rerun. The new symbol hints do not solve target precision outright, but they seem to help the model commit inside the right neighborhood once it has the right file.

The diagnosis metrics also improved. Zero-edit dropped from 3 back to 2, first edit improved from 5.92 to 5.50, and first verification improved more sharply from 7.58 to 6.42. That is the best verification timing of all four runs so far.

The remaining problem is that wrong verified patches are still sticky. `wrong_patch_after_verification` stayed at 3, which means we still need a better notion of what to verify once the model is inside the right file. But now the failure mix is much healthier. We have fewer wasted trajectories, earlier verification, and more total submissions reaching a real patch.

As before, the benchmark DB completed and the run summary printed before podman teardown finished. I killed job `789142` after the useful work was already written to disk.

### Additional files

- `diagnostic-swebench-report-after-churn-reduction.html` - fourth-run report ([view](https://raw.githack.com/0xCUB3/mcode/main/research/2026-04-03-adapter-aware-orchestrator-contract/diagnostic-swebench-report-after-churn-reduction.html))
- `diagnostic-results-summary-after-churn-reduction.txt` - fourth-run CLI summary
- `run-bluevela-minimax25-diagnostic-b15-after-churn-reduction/diagnostic.db` - fourth-run results DB
- `run-bluevela-minimax25-diagnostic-b15-after-churn-reduction/final-summary.json` - fourth-run scaffold metrics and per-task outcomes
- `run-bluevela-minimax25-diagnostic-b15-after-churn-reduction/swb-diag-m25-churn.log` - fourth-run Blue Vela log


## References

- OpenHands runtime and event architecture: https://docs.openhands.dev/usage/architecture/runtime and https://docs.openhands.dev/sdk/arch/events
- OpenAI practical guide to building agents: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- OpenAgent architecture notes: https://openagents.org/docs/concepts/architecture
