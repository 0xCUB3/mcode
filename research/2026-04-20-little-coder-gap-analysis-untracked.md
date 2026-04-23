# little-coder vs mcode/mellea gap analysis

Date: 2026-04-20

This note answers one question: what is little-coder doing that the current mcode + mellea stack is not, and where should those missing pieces live.

The first thing to clear up is the benchmark mismatch. little-coder's published `78.67%` number is not a SWE-bench Lite Verified result. It is a full 225-exercise Aider Polyglot run with `llamacpp/qwen3.6-35b-a3b`, documented in `README.md` and `docs/benchmark-qwen3.6-35b-a3b.md` inside the cloned little-coder repo. The same repo's whitepaper is explicit that transfer to SWE-bench has not been established yet. So the right use of little-coder here is mechanism comparison, not score comparison.

On the mcode side, the active benchmark path is already much tighter than the old harness. `src/mcode/agent/coding_agent.py` gives the model a compact tool surface, `src/mcode/agent/coding_policy.py` keeps the prompt narrow, and `src/mcode/llm/session.py` plus `src/mcode/llm/react_driver.py` enforce an actual verification boundary by discarding unverified diffs. The benchmark DB also tracks useful scaffold metrics such as first edit, first verification, zero-edit, zero-verification, and terminal reason buckets. That part is not the main weakness anymore.

The more important boundary comes from `memory://root/memory_summary.md` and the current design note in `research/2026-04-03-adapter-aware-orchestrator-contract/README.md`. Both point the same way. Reusable runtime and tool-layer behavior belongs in mellea. Benchmark-specific policy, prompts, task wiring, and reporting belong in mcode. That boundary still looks right after comparing with little-coder.

## What little-coder clearly has that matters

From little-coder's docs and extension code, the load-bearing mechanisms are:

- hard write-vs-edit enforcement (`.pi/extensions/write-guard/index.ts`)
- per-model and per-benchmark profiles (`.pi/settings.json`, `.pi/extensions/benchmark-profiles/index.ts`)
- thinking-budget control that aborts runaway reasoning and retries with thinking off (`.pi/extensions/thinking-budget/index.ts`)
- response-quality monitoring with follow-up corrections (`.pi/extensions/quality-monitor/index.ts`)
- malformed tool-call detection with repair nudges (`.pi/extensions/output-parser/index.ts`)
- per-turn tool-skill injection (`.pi/extensions/skill-inject/index.ts`)
- per-turn knowledge / protocol injection (`.pi/extensions/knowledge-inject/index.ts`)
- context compaction as a first-class part of the stack (documented in `docs/architecture.md` and enabled in `.pi/settings.json`)

The whitepaper and reproduction docs do not prove the exact causal weight of each item, but they do show that several of them are active at meaningful rates. The write guard fires on about 57% of Polyglot exercises. The thinking budget fires about 0.9 times per exercise. Workspace discovery via Glob shows up on about 67% of exercises. Retry is worth about +8 points in the Qwen3.5 runs.

## What is already present or intentionally covered in mcode/mellea

A few little-coder ideas are already present in spirit, or are simply not missing because mcode solved them a different way.

The write guard is the clearest example. mcode's default benchmark path does not expose a write tool at all. It only gives the model `edit`, not `write`, so there is nothing to port there unless the tool surface changes again.

The compact tool surface is also already present. The current mcode default is smaller than little-coder's Polyglot tool set, not larger. That is not the bottleneck.

Verification gating is stronger in mcode than in little-coder's published Polyglot loop. mcode explicitly tracks verification success and can discard an unverified diff before submission. That is already benchmark-aware and useful.

Mellea also already contains some generic runtime primitives that mcode is not fully using. In the installed package, `mellea/agent/runtime/loops.py` provides a reusable observe-act-verify loop, `mellea/agent/runtime/memory.py` provides condensed state and working memory, `mellea/agent/strategy/phased.py` provides phased tool access, and `mellea/agent/text_react.py` exposes `tool_gate`, `event_log`, and condensation hooks. Those are real primitives, but they are mostly hooks right now, not the active behavior of mcode's benchmark loop.

## What is actually missing, and where it belongs

### Missing in mellea, high priority

These are the biggest reusable gaps relative to little-coder.

1. A real small-model control layer.

Mellea has generic loop primitives, but it does not currently have the adaptive controller little-coder uses for small models: thinking-budget enforcement, response-quality correction, malformed-tool-call recovery, and dynamic prompt augmentation. Those are generic runtime behaviors, not SWE-bench specifics, so they belong in mellea.

2. First-class history condensation integrated into the active loop.

Mellea has condensation primitives, but mcode's active benchmark path still uses its own custom ReAct loop and only has a crude optional `_compress_old_tool_outputs` path in `src/mcode/llm/react_driver.py`. The gap is not "condensation does not exist anywhere." The gap is that the reusable condensation layer is not actually driving the benchmark loop. That is mainly a mellea integration problem.

3. A reusable phase / capability router that is actually wired in.

The design note wants `diagnose -> edit -> verify -> submit` and capability-family routing. Mellea exposes pieces of this, especially phased access and text-react hooks, but there is no default reusable controller that owns phase state and blocks or redirects actions based on it. That belongs in mellea.

4. Generic prompt-augmentation infrastructure.

little-coder's biggest structural difference is not one longer prompt. It is targeted per-turn injection of tool guidance and domain knowledge under a token budget. There is nothing comparable in mellea right now. The generic machinery for selecting and injecting small prompt fragments should live in mellea, even if the actual fragment library is project-specific.

### Missing in mcode, high priority

These are benchmark-facing policy gaps.

1. SWE-bench-specific prompt modules on top of a generic injection system.

Once mellea has a generic injector, mcode should own the actual SWE-bench-facing prompt content: verification workflow hints, workspace-doc discovery hints, Python-task execution guidance, and task-shape-specific fragments. Today mcode has repo maps, candidate files, repo customization, and a short verification block, but nothing like little-coder's dynamic skill / knowledge selection.

2. Model and benchmark profiles for the benchmark path.

little-coder tunes thinking budget, temperature, and turn caps per model and per benchmark. mcode mostly relies on CLI flags and environment variables. The profile concept itself could be generic, but the concrete mapping from Qwen3.6-on-SWE-bench to default budget / timeout / context / retry policy is mcode policy.

3. Better use of the benchmark metrics it already collects.

mcode already has better benchmark-native counters than little-coder in some ways, but those counters are mostly observational. The next step is to let them steer the run more directly, for example by changing phase pressure after repeated zero-edit or late-verification behavior. The metric store stays in mcode because it is benchmark reporting, but the steering hooks should be fed by reusable mellea runtime state.

### Missing in mcode, lower priority or probably not worth copying directly

1. ShellSession, browser, evidence, GAIA-specific tooling.

Those are useful for other benchmarks and interactive use, but they are not the obvious reason the current SWE-bench path is behind.

2. Write-guard as a direct port.

Again, the default mcode benchmark path already side-steps this by not exposing write.

3. Persistent long-term memory.

For isolated SWE-bench tasks, session-to-session memory is much less important than within-task loop quality.

## The biggest non-obvious conclusion

The main gap is not "mcode needs more prompt hacks," and it is not "mellea is missing basic tools." The main gap is that little-coder has a coherent small-model controller and mcode does not. That controller includes hard controls for runaway reasoning, soft recovery for malformed or low-quality turns, dynamic prompt shaping, and practical context management. mellea has pieces of the substrate for this, but not the full controller. mcode has benchmark metrics and policy intent, but not the adaptive layer that would use those signals well.

So if I had to assign ownership cleanly:

- reusable adaptive runtime behaviors, loop state, condensation, quality monitor, malformed-call recovery, prompt-injection framework: mellea
- SWE-bench-specific prompt content, model defaults, task wiring, reporting, and benchmark-level policy choices: mcode
- things already covered well enough: compact tool surface, no-write default, benchmark observability, unverified-diff discard

## Definitive conclusion

little-coder is not proof that the current Qwen3.6 mcode path should already be at 80% on SWE-bench. Its published 78.67% result is on Aider Polyglot, and the repo explicitly says transfer to SWE-bench is unproven.

But the comparison is still useful, and the verdict is pretty clear. mcode is not mainly missing infra or raw tool access. It is missing a small-model-aware control layer. Most of that layer should be built in mellea because it is reusable runtime behavior. mcode should then consume it with SWE-bench-specific prompt modules, per-model defaults, and benchmark reporting.

If the goal is to make the harness insane in the good way, the highest-value path is not another batch of benchmark-local nudges. It is to move the active benchmark loop onto a stronger reusable runtime in mellea, then let mcode supply the SWE-bench-specific policy on top.


## Aider Polyglot harness run, April 23
This is the concrete Polyglot pass that followed the gap analysis above. I added the Aider Polyglot benchmark path in mcode, ran Qwen3.6-35B-A3B through it, then kept only fixes that improved an observed score slice.

Full run command:
```bash
OPENAI_BASE_URL=http://127.0.0.1:18322/v1 \
OPENAI_API_KEY=dummy \
MCODE_CONTEXT_WINDOW=32768 \
MCODE_MAX_NEW_TOKENS=4096 \
MCODE_REACT_TIMEOUT=1800 \
uv run mcode bench aider-polyglot \
  --model Qwen/Qwen3.6-35B-A3B \
  --backend openai \
  --temperature 0.3 \
  --benchmark-root /Users/skula/Documents/polyglot-benchmark \
  --db experiments/results/aider-polyglot-qwen36-full-patched4.db
```

Rendered report: `research/2026-04-20-little-coder-gap-analysis-report.html`.

The first full run landed at 103/225, or 45.8%. The per-language split was Python 21/34, Go 25/39, Rust 18/30, JavaScript 37/49, C++ 2/26, and Java 0/47. Terminal reasons were 103 submitted, 89 unverified diff discarded, 33 budget exhausted, and 0 infra failures.

Java was a harness environment bug, not model behavior. The run initially had no valid JDK, then the wrong one. Installing `openjdk@21` and setting `JAVA_HOME` inside the Polyglot command runner moved Java from 0/47 to 30/47 in `experiments/results/aider-polyglot-java-only-patched.db`.

C++ was a tool-surface mismatch with little-coder. The model kept getting blocked by tree-sitter syntax rejection on C++ headers and sources, while little-coder's edit path is plain text and lets the compiler decide. Skipping the edit-time syntax guard for C-family files moved the C++ slice from 2/26 to 17/26 in `experiments/results/aider-polyglot-cpp-only-patched.db`. The corrected C++ terminal reasons were 17 submitted, 8 unverified diff discarded, and 1 budget exhausted.

Corrected by substituting the validated Java and C++ slices into the full run, the run is 148/225, or 65.8%. That is still below little-coder's published 78.67%, but the gap is now mostly controller behavior and malformed-call churn rather than deterministic Java or C++ harness failure.

The kept deterministic fixes are: Aider Polyglot benchmark support in mcode, Java 21 environment setup for Polyglot commands, and C-family edit syntax guard bypass. I would not keep broader Bash access yet because the C++ score improved substantially without it, and the remaining C++ failures are mostly real unverified diffs rather than blocked compiler access.