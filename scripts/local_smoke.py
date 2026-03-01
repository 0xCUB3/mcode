#!/usr/bin/env python3
"""Local smoke test: agentic patch generation against a local repo checkout.

Usage:
    BACKEND=ollama MODEL=devstral:24b REPO_ROOT=/tmp/pylint-testbed \
        uv run python scripts/local_smoke.py

    BACKEND=claude REPO_ROOT=/tmp/pylint-testbed \
        uv run python scripts/local_smoke.py

Environment:
    BACKEND        - ollama (default) or claude
    MODEL          - model name (required for ollama, ignored for claude)
    REPO_ROOT      - path to checked-out repo at the right commit
    LOOP_BUDGET    - react loop budget multiplier (default 5)
    CONTEXT_WINDOW - context window override (default 65536)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcode.context.localize import localize

PROBLEM_STATEMENT = """\
Annotated module level constant not checked for `invalid-name`

Originally reported by @mbyrnepr2 in https://github.com/pylint-dev/pylint/issues/7232#issuecomment-1197868763.

### Bug description

```python
\"\"\"should raise invalid-name\"\"\"
my_var: int = 1
```

### Configuration

_No response_

### Command used

```shell
pylint test.py
```

### Pylint output

_No output._

### Expected behavior

`invalid-name` should be raised for `my_var` since `UPPER_CASE` naming style is expected
for module-level constants.

### Pylint version

```
pylint 2.15.0-dev0
astroid 2.12.0-dev0
Python 3.10.4 (main, Apr  2 2022, 09:04:19) [GCC 11.2.0]
```

### OS / Environment

Pop!_OS 22.04
"""


def _run_claude(repo_root: str, problem_statement: str, file_paths: list[str]) -> str:
    """Run claude -p in the repo directory and return git diff."""
    file_hint = "\n".join(f"  - {f}" for f in file_paths)
    prompt = (
        "Fix this bug in pylint-dev/pylint by editing the existing source code.\n\n"
        f"Issue:\n{problem_statement.strip()}\n\n"
        f"Files likely relevant (from BM25 ranking):\n{file_hint}\n\n"
        "Edit the existing source files to fix the bug. "
        "Do not create new files or test scripts."
    )
    print(f"  prompt length: {len(prompt)} chars", flush=True)
    result = subprocess.run(
        ["claude", "-p", prompt, "--allowedTools", "Edit,Read,Grep,Glob,Bash"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout, flush=True)
    if result.returncode != 0:
        print(f"  claude stderr: {result.stderr[:500]}", flush=True)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True)
    return diff.stdout


def main():
    backend = os.environ.get("BACKEND", "ollama")
    model = os.environ.get("MODEL", "")
    repo_root = os.environ.get("REPO_ROOT", "/tmp/pylint-testbed")
    loop_budget = int(os.environ.get("LOOP_BUDGET", "5"))
    context_window = int(os.environ.get("CONTEXT_WINDOW", "65536"))

    if backend != "claude" and not model:
        print("MODEL env var required (unless BACKEND=claude)", file=sys.stderr)
        sys.exit(1)

    print(f"backend={backend}  model={model}  repo={repo_root}")

    t0 = time.time()
    print("\n--- localize ---")
    loc_files, _ = localize(repo_root, PROBLEM_STATEMENT)
    print(f"localized {len(loc_files)} files in {time.time() - t0:.1f}s")
    for f in loc_files[:10]:
        print(f"  {f}")

    print("\n--- generate_patch ---")
    t1 = time.time()

    if backend == "claude":
        patch = _run_claude(repo_root, PROBLEM_STATEMENT, loc_files[:10])
    else:
        from mcode.llm.session import LLMSession

        os.environ["MCODE_CONTEXT_WINDOW"] = str(context_window)
        session = LLMSession(
            model_id=model,
            backend_name=backend,
            loop_budget=loop_budget,
        )
        with session.open():
            patch = session.generate_patch(
                repo="pylint-dev/pylint",
                problem_statement=PROBLEM_STATEMENT,
                file_paths=loc_files[:10],
                repo_root=repo_root,
            )

    elapsed = time.time() - t1
    print(f"\n--- result (elapsed={elapsed:.1f}s) ---")
    print(f"patch_chars={len(patch)}")
    if patch:
        print("\n--- patch ---")
        print(patch[:3000])
        if len(patch) > 3000:
            print(f"... ({len(patch) - 3000} more chars)")
    else:
        print("(no patch produced)")


if __name__ == "__main__":
    main()
