#!/usr/bin/env python3
"""Local smoke test: agentic patch generation against a local repo checkout.

Usage:
    BACKEND=ollama MODEL=devstral:24b REPO_ROOT=/tmp/pylint-testbed \
        uv run python scripts/local_smoke.py

Environment:
    BACKEND        - ollama (default)
    MODEL          - model name (required)
    REPO_ROOT      - path to checked-out repo at the right commit
    LOOP_BUDGET    - react loop budget multiplier (default 5)
    CONTEXT_WINDOW - context window override (default 65536)
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcode.context.localize import localize
from mcode.llm.session import LLMSession

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


def main():
    backend = os.environ.get("BACKEND", "ollama")
    model = os.environ.get("MODEL")
    repo_root = os.environ.get("REPO_ROOT", "/tmp/pylint-testbed")
    loop_budget = int(os.environ.get("LOOP_BUDGET", "5"))
    context_window = int(os.environ.get("CONTEXT_WINDOW", "65536"))

    if not model:
        print("MODEL env var required", file=sys.stderr)
        sys.exit(1)

    print(f"model={model}  backend={backend}  repo={repo_root}")
    print(f"loop_budget={loop_budget}  context_window={context_window}")

    os.environ["MCODE_CONTEXT_WINDOW"] = str(context_window)

    session = LLMSession(
        model_id=model,
        backend_name=backend,
        loop_budget=loop_budget,
    )

    t0 = time.time()
    with session.open():
        print("\n--- localize ---")
        loc_files, _ = localize(repo_root, PROBLEM_STATEMENT)
        print(f"localized {len(loc_files)} files in {time.time() - t0:.1f}s")
        for f in loc_files[:10]:
            print(f"  {f}")

        print("\n--- generate_patch (agentic) ---")
        t1 = time.time()
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
