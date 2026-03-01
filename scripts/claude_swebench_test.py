#!/usr/bin/env python3
"""Smoke test: run 1-3 SWE-bench tasks through the agentic pipeline.

Run from a separate terminal (NOT inside Claude Code):
    cd /Users/skula/Documents/mcode
    PYTHONUNBUFFERED=1 uv run python scripts/claude_swebench_test.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from mcode.bench.swebench_live import load_swebench_live  # noqa: E402
from mcode.context.localize import localize  # noqa: E402
from mcode.execution.swebench_live import SWEbenchLiveSandbox  # noqa: E402
from mcode.llm.session import LLMSession  # noqa: E402

MODEL = os.environ.get("MODEL", "qwen3-coder:30b")
BACKEND = os.environ.get("BACKEND", "ollama")
N_TASKS = int(os.environ.get("N_TASKS", "3"))

os.environ.setdefault("MCODE_CONTEXT_WINDOW", "65536")


def main():
    tasks = load_swebench_live(None, split="verified")
    tasks = [t for t in tasks if "conan" not in t.instance_id and "matplotlib" not in t.instance_id]

    selected = []
    seen_repos: set[str] = set()
    for t in tasks:
        repo_key = t.repo.split("/")[0]
        if repo_key not in seen_repos and len(selected) < N_TASKS:
            selected.append(t)
            seen_repos.add(repo_key)

    print(f"Agentic smoke test: {MODEL} ({BACKEND}), {len(selected)} tasks")
    for t in selected:
        print(f"  {t.instance_id} ({t.repo})")

    results = []
    for t in selected:
        print(f"\n===== {t.instance_id} =====", flush=True)
        t0 = time.time()
        try:
            sandbox = SWEbenchLiveSandbox()
            with sandbox.repo_context(t) as repo_root:
                session = LLMSession(
                    model_id=MODEL,
                    backend_name=BACKEND,
                    loop_budget=3,
                )
                with session.open():
                    loc_files, _ = localize(str(repo_root), t.problem_statement)
                    patch = session.generate_patch(
                        repo=t.repo,
                        problem_statement=t.problem_statement,
                        file_paths=loc_files[:10],
                        repo_root=str(repo_root),
                    )
                has_patch = bool(patch.strip())
                print(f"patch: {has_patch} ({len(patch)} chars)", flush=True)
                if has_patch:
                    run = sandbox.evaluate_patch(
                        task=t,
                        patch=patch,
                        run_id="smoke",
                        timeout_s=600,
                    )
                    resolved = run.resolved
                else:
                    resolved = False
                elapsed = time.time() - t0
                print(
                    f"  >> resolved={resolved} patch={has_patch} time={elapsed:.0f}s",
                    flush=True,
                )
                results.append(
                    {
                        "id": t.instance_id,
                        "resolved": resolved,
                        "patch": has_patch,
                    }
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"ERROR: {e}", flush=True)
            results.append(
                {
                    "id": t.instance_id,
                    "resolved": False,
                    "patch": False,
                }
            )

    print("\n===== SUMMARY =====")
    for r in results:
        status = f"patch={r['patch']} resolved={r['resolved']}"
        print(f"  {r['id']}: {status}")
    resolved = sum(1 for r in results if r["resolved"])
    patches = sum(1 for r in results if r["patch"])
    print(f"\npatches: {patches}/{len(results)}, resolved: {resolved}/{len(results)}")


if __name__ == "__main__":
    main()
