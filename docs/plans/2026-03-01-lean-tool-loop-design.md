# Lean tool loop design

## Problem

Mellea's react() adds ~1090 tokens of overhead over a 25-turn loop via ReactInitiator (~90 tok, verbose goal template with Think/Act/Observe instructions) and ReactThought (~40 tok/turn, "Think about what to do next" prompt). On qwen3-coder:30b with 32K context, this overhead costs 3.4% of the context window that small models need for actual code.

Results: react() scores 2/10 resolved on SWE-bench Lite vs 4/10 with a manual Ollama loop.

Additionally, ChatContext window_size pruning drops ReactInitiator (which carries tool schemas), causing tools to disappear mid-loop. Removing window_size fixes this but means unbounded context growth.

## Solution

Replace react() with a lean async loop built from Mellea primitives: `aact()`, `_call_tools()`, `ChatContext`, `CBlock`, `ModelOption.TOOLS`. No ReactInitiator, no ReactThought.

## Design

### New function: `_lean_tool_loop()`

Location: `src/mcode/llm/session.py` (private async function, called by `generate_patch()`).

```python
async def _lean_tool_loop(
    goal: str,
    context: ChatContext,
    backend,
    tools: list,
    budget: int,
    model_opts: dict,
    timeout_s: int,
) -> str | None:
```

Returns the final_answer string, or None if budget/timeout exhausted.

### Loop structure

1. Add `CBlock(goal)` to context as the initial user message.
2. Add a `final_answer` tool (simple passthrough: `def final_answer(answer: str) -> str: return answer`) to the tools list.
3. Put all tools in `model_opts[ModelOption.TOOLS]`.
4. Loop up to `budget` turns:
   a. Call `aact(action=CBlock(""), context, backend, strategy=None, tool_calls=True, model_options=model_opts)`
   b. If `step.tool_calls`: call `_call_tools(step, backend)`, add each ToolMessage to context.
   c. If any tool_call name is `"final_answer"`, return its content.
5. On budget exhaustion or timeout, return None.

### Integration with generate_patch()

`generate_patch()` changes:
- Replace `react()` import and call with `_lean_tool_loop()`.
- Keep `asyncio.wait_for()` timeout wrapper.
- After loop returns, always call `get_diff(repo_root)` regardless of whether final_answer was called (the diff is what matters, not the answer text).

### What stays the same

- `ChatContext()` (unbounded, no window_size)
- `mellea.backends.tools.tool` for creating tool wrappers
- `make_tools(repo_root)` for search_code/read_file/apply_edit
- `_model_options()` for system prompt, temperature, context window
- `asyncio.wait_for()` timeout
- `LLMSession.open()` context manager for backend lifecycle

### What changes

| Before (react) | After (lean loop) |
|-|-|
| ReactInitiator with verbose template (~90 tok) | CBlock(goal) (~goal tokens only) |
| ReactThought per turn (~40 tok) | CBlock("") (0 tok) |
| Tools via ReactInitiator's TemplateRepresentation | Tools via ModelOption.TOOLS |
| final_answer injected by ReactInitiator | Our own final_answer tool function |
| react() manages loop | Our ~30-line _lean_tool_loop() |

### Test changes

`tests/test_agent_generate.py`: mock `aact` at `mellea.stdlib.functional.aact` instead of `react`. The mock should return a ModelOutputThunk with tool_calls containing final_answer.

### Risk: empty CBlock action

An empty string CBlock becomes an empty user message each turn. If this confuses the model, we can change it to a minimal prompt like "Proceed." but start with empty to minimize overhead.

### Success criteria

Run the same 10 SWE-bench Lite instances with qwen3-coder:30b. Target: >= 4/10 resolved (matching original manual loop), no hangs, no tool schema loss.
