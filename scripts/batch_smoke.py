#!/usr/bin/env python3
"""Batch smoke test: run agentic pipeline on multiple SWE-bench Lite instances.

Usage:
    BACKEND=ollama MODEL=qwen3-coder:30b \
        uv run python scripts/batch_smoke.py

Environment:
    BACKEND        - ollama (default)
    MODEL          - model name (required)
    LOOP_BUDGET    - react loop budget multiplier (default 5)
    CONTEXT_WINDOW - context window override (default 32768)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

INSTANCES = [
    "django__django-11179",
    "django__django-14999",
    "django__django-16046",
    "pallets__flask-4045",
    "psf__requests-2674",
    "sympy__sympy-16503",
    "sympy__sympy-15345",
    "pytest-dev__pytest-5221",
    "scikit-learn__scikit-learn-13497",
    "scikit-learn__scikit-learn-13584",
]


def main():
    backend = os.environ.get("BACKEND", "ollama")
    model = os.environ.get("MODEL")
    loop_budget = int(os.environ.get("LOOP_BUDGET", "5"))
    context_window = int(os.environ.get("CONTEXT_WINDOW", "32768"))

    if not model:
        print("MODEL env var required", file=sys.stderr)
        sys.exit(1)

    os.environ["MCODE_CONTEXT_WINDOW"] = str(context_window)

    from mcode.bench.swebench_lite import load_swebench_lite
    from mcode.context.localize import localize as localize_files
    from mcode.execution.swebench import SWEbenchSandbox
    from mcode.llm.session import LLMSession

    tasks = load_swebench_lite(Path("/tmp"), instance_ids=INSTANCES)
    print(f"loaded {len(tasks)} tasks")

    sandbox = SWEbenchSandbox(namespace="swebench", arch="x86_64")
    session = LLMSession(
        model_id=model,
        backend_name=backend,
        loop_budget=loop_budget,
    )

    results = []
    with session.open():
        for i, task in enumerate(tasks):
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(tasks)}] {task.instance_id}")
            print(f"{'=' * 60}")
            t0 = time.time()

            try:
                with sandbox.repo_context(task.raw_instance) as repo_root:
                    loc_files, _ = localize_files(str(repo_root), task.problem_statement)
                    print(f"  localized {len(loc_files)} files in {time.time() - t0:.1f}s")
                    for f in loc_files[:5]:
                        print(f"    {f}")

                    patch = session.generate_patch(
                        repo=task.repo,
                        problem_statement=task.problem_statement,
                        hints_text=task.hints_text or "",
                        file_paths=loc_files[:10],
                        repo_root=str(repo_root),
                    )
                    gen_time = time.time() - t0

                    has_patch = bool(patch and patch.strip())
                    resolved = False
                    if has_patch:
                        print(f"  patch: {len(patch)} chars, evaluating...")
                        run = sandbox.evaluate_patch(
                            instance=task.raw_instance,
                            model_id=model,
                            patch=patch,
                            run_id=f"smoke-{i}",
                            timeout_s=300,
                        )
                        resolved = run.resolved
                        eval_time = time.time() - t0 - gen_time
                    else:
                        print("  (no patch produced)")
                        eval_time = 0

            except Exception as e:
                print(f"  ERROR: {e}")
                has_patch = False
                resolved = False
                gen_time = time.time() - t0
                eval_time = 0
                patch = ""

            total = time.time() - t0
            status = "RESOLVED" if resolved else ("PATCH" if has_patch else "EMPTY")
            print(f"  => {status}  gen={gen_time:.0f}s  eval={eval_time:.0f}s  total={total:.0f}s")

            results.append(
                {
                    "instance_id": task.instance_id,
                    "resolved": resolved,
                    "has_patch": has_patch,
                    "gen_time_s": round(gen_time, 1),
                    "total_time_s": round(total, 1),
                }
            )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    n_resolved = sum(1 for r in results if r["resolved"])
    n_patch = sum(1 for r in results if r["has_patch"])
    print(f"resolved: {n_resolved}/{len(results)}")
    print(f"produced patch: {n_patch}/{len(results)}")
    for r in results:
        status = "RESOLVED" if r["resolved"] else ("PATCH" if r["has_patch"] else "EMPTY")
        print(f"  {r['instance_id']:45s} {status:10s} {r['total_time_s']:6.0f}s")

    out_path = Path("results_smoke.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
