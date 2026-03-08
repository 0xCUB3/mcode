# Mellea Fork: Modular Agent Toolkit for Small LLMs

## Goal

Fork mellea to add a modular, language-agnostic agent toolkit optimized for small LLMs (3B-8B active params). Improve SWE-bench Live performance by fixing the specific failure modes identified in our Qwen3.5-35B-A3B runs: search loops, bad edits, malformed output, wasted budget on blind exploration.

## Context

Current results: 9/300 (3.0%) on SWE-bench Live Lite with budget=15. Key failure modes:
- 50% wrong patches (model edits wrong code or wrong lines)
- 24% budget exhausted without producing a patch
- 22% malformed model output (JSON errors)
- Search loops (same query repeated 60+ times)
- Slow search (Python rglob, 2-3 min per search on large repos)
- Python-only tooling (compile() check, .py-only BM25 indexing)

Research on SWE-agent, OpenHands, Agentless, Aider, and Moatless shows consistent patterns for what helps small models: linter gates, repo maps, constrained tool access, loop detection, and fast text search.

## Fork Setup

- GitHub fork of upstream mellea repo as `github.com/0xCUB3/mellea`
- Separate repo, mcode depends on it via `mellea @ git+https://github.com/0xCUB3/mellea`
- Additions only in `mellea/agent/` and `mellea/eval/` -- no changes to upstream core
- Follow mellea code style: immutable Contexts, Component protocol, SamplingStrategy subclasses, async-first, dataclasses, factory methods, extensive type hints

## Architecture

```
mellea (upstream, untouched)
  core/          Backend, Context, Component, SamplingStrategy
  backends/      OpenAI, vLLM, Ollama, etc.
  stdlib/        act/react, sampling strategies
  formatters/

mellea/agent/ (new)
  tools/
    search.py        ripgrep subprocess, 30-match cap, 5s timeout
    edit.py          str_replace + tree-sitter syntax validation
    read.py          200-line capped viewer with line numbers
    navigate.py      find_file (glob), list_dir
  repomap/
    __init__.py      tree-sitter repo map with PageRank ranking
    tags.py          tag extraction per language
    graph.py         file graph + PageRank
    render.py        skeleton formatter with token budget
  context/
    masking.py       observation masking Context subclass
  strategy/
    loop_detect.py   repeated tool call detection + forced switch
    phased.py        percentage-based tool access phases

mellea/eval/ (new)
  smoke.py           curated ~25 task smoke suite
  compare.py         A/B comparison CLI between two runs
```

## Components

### Tools

All tools are plain Python callables, registered via `MelleaTool.from_callable()`.

**search(query: str) -> str**
- Subprocess call to `rg` (ripgrep) with: `--max-count=5 --max-columns=200 -n --type-add '...'`
- Cap at 30 total matches across all files
- Skip .git, node_modules, __pycache__, .venv, build, dist
- Timeout at 5 seconds
- Returns `file:line: content` format

**edit(path: str, old_str: str, new_str: str) -> str**
- Find `old_str` in file, replace with `new_str`
- If `old_str` not found or not unique, return error with context
- After edit: parse file with tree-sitter, check for new ERROR nodes
- If new syntax errors introduced, revert the edit and return the error location
- Works for all 165+ tree-sitter languages

**read(path: str, start_line: int = 1, end_line: int | None = None) -> str**
- Show max 200 lines per read
- Include line numbers
- Show total file length header: `"file.py (450 lines total, showing 1-200)"`

**find_file(pattern: str) -> str**
- Glob search for filenames matching pattern
- Returns list of matching paths, max 50 results

**list_dir(path: str = ".") -> str**
- Directory listing with file types
- Non-recursive, shows immediate children only

### Repo Map

Built once before agent loop starts, included in initial context.

1. Parse every source file with `tree-sitter-language-pack`, extract definition tags (functions, classes, methods) and reference tags (calls, imports)
2. Build NetworkX MultiDiGraph: files as nodes, shared symbol names as edges
3. Run PageRank personalized toward BM25 top candidates from the issue description
4. Format top-ranked files as condensed skeletons (signatures only) within a configurable token budget (default 4096 tokens)
5. Cache parsed map per repo (doesn't change during a task)
6. Graceful fallback: files without a tree-sitter grammar still appear in BM25, just without structural info

### Observation Masking

Custom Context subclass. When generating the context for the LLM:
- Keep all action/reasoning history intact
- Replace tool outputs older than the last 3 turns with a one-line summary:
  `"[search('pattern') -> 12 matches, see turn 3]"`
- Cuts context size roughly in half
- Configurable window (default: keep last 3 turns unmasked)

### Loop Detection

Wraps the sampling strategy. Tracks recent tool calls:
- Same function + same args 2 times consecutively: inject nudge message ("You already tried this. Try a different approach.")
- Same function + same args 3 times: force different tool or final_answer
- Composable with other strategy wrappers

### Phased Tool Access

Percentage-based phases, configurable via `phases` parameter (default `[0.4, 0.8, 1.0]`):
- Phase 1 (first 40% of budget): search, read, find_file, list_dir only (explore)
- Phase 2 (40-80%): all tools including edit (implement)
- Phase 3 (last 20%): edit and final_answer only (commit)

With budget=15: turns 1-6, 7-12, 13-15.
With budget=5: turns 1-2, 3-4, 5.

## Evaluation

### Smoke Suite (~25 tasks, ~10 min)
Hand-picked from run2 results:
- 5 tasks we already solve (regression)
- 5 tasks that failed due to search loops
- 5 tasks that failed due to bad edits
- 5 non-Python repo tasks (multilingual)
- 5 near-misses (close to passing)

Stored as a JSON list of task IDs.

### A/B Comparison
CLI command: `mcode eval compare --baseline run2.db --candidate run3.db --out comparison.html`

Output: which tasks flipped (pass->fail, fail->pass), aggregate rates, per-task time deltas.

### Workflow
1. Make a change
2. Run smoke suite (~10 min)
3. `mcode eval compare` to see the diff
4. If promising, run full SWE-bench Lite (hours)
5. HumanEval/MBPP as regression checks

## mcode Changes

- Switch mellea dependency to fork (`mellea @ git+https://github.com/0xCUB3/mellea`)
- Delete `src/mcode/agent/tools.py` old tool implementations (rglob search, line-number edit, Python compile() check)
- Delete `src/mcode/context/localize.py` BM25-only localization (replaced by repo map + BM25)
- Update `src/mcode/llm/session.py` to use new tools, repo map, and strategy components
- Clean up any dead code paths

## Dependencies

Added to mellea fork:
- `tree-sitter-language-pack` (165+ languages, pre-built wheels)
- `networkx` (PageRank for repo map)
- ripgrep assumed available on system PATH

## Success Criteria

- Smoke suite passes in <10 min
- SWE-bench Live Lite resolve rate improves from 3.0% to >5%
- Effective rate (excl infra) improves from 6.8% to >10%
- No regression on HumanEval/MBPP
- All tools work on non-Python repos
