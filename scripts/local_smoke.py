#!/usr/bin/env python3
"""Local smoke test: localize + generate_patch against a local repo checkout.

Usage:
    BACKEND=ollama MODEL=devstral:24b REPO_ROOT=/tmp/pylint-testbed \
        uv run python scripts/local_smoke.py

Environment:
    BACKEND        - ollama (default)
    MODEL          - model name (required)
    REPO_ROOT      - path to checked-out repo at the right commit
    LOOP_BUDGET    - repair loop attempts (default 5)
    CONTEXT_WINDOW - context window override (default 65536)
"""

from __future__ import annotations

import os
import sys
import time

# Make mcode importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcode.context.localize import localize
from mcode.llm.session import LLMSession, line_edits_to_patch

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

    # Set context window for ollama
    os.environ["MCODE_CONTEXT_WINDOW"] = str(context_window)

    # Build validation function (syntax-gate only, no test execution)
    def _patch_test(raw_json: str):
        patch, edit_errors = line_edits_to_patch(raw_json, repo_root=repo_root)
        if edit_errors:
            for e in edit_errors:
                print(f"  >> {e}", flush=True)
        if not patch and edit_errors:
            return (False, "Edit errors:\n" + "\n".join(edit_errors))
        if not patch:
            return (False, "Empty patch produced")
        # No test execution locally - just return the patch
        print(f"\n--- patch ({len(patch)} chars) ---")
        print(patch[:2000])
        if len(patch) > 2000:
            print(f"... ({len(patch) - 2000} more chars)")
        return True

    from mellea.stdlib.requirements.requirement import Requirement, simple_validate

    req = Requirement(
        validation_fn=simple_validate(_patch_test),
        check_only=True,
    )

    session = LLMSession(
        model_id=model,
        backend_name=backend,
        loop_budget=loop_budget,
    )

    t0 = time.time()
    with session.open():
        # 1. Localize files (with LLM narrowing)
        print("\n--- localize ---")
        loc_files, loc_hints = localize(repo_root, PROBLEM_STATEMENT, llm_session=session)
        print(f"localized {len(loc_files)} files in {time.time() - t0:.1f}s")
        for f in loc_files:
            print(f"  {f}")

        # 2. Generate patch
        print("\n--- generate_patch ---")
        t1 = time.time()
        result = session.generate_patch(
            repo="pylint-dev/pylint",
            problem_statement=PROBLEM_STATEMENT,
            hints_text=loc_hints or "",
            file_paths=loc_files,
            requirements=[req],
            repo_root=repo_root,
        )
    elapsed = time.time() - t1
    print(f"\n--- result (elapsed={elapsed:.1f}s) ---")
    print(f"success={result.success}")
    print(f"attempts={len(result.sample_generations)}")

    # Final patch
    patch, errors = line_edits_to_patch(result.value or "", repo_root=repo_root)
    print(f"patch_chars={len(patch)}")
    if errors:
        print(f"errors: {errors}")
    if patch:
        print("\n--- final patch ---")
        print(patch)


if __name__ == "__main__":
    main()
