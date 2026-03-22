from __future__ import annotations

import importlib.util
from pathlib import Path

from mcode.bench.results import ResultsDB


def _load_patch_merge_module():
    path = Path(__file__).resolve().parents[1] / "deploy" / "bluevela" / "patch_merge.py"
    spec = importlib.util.spec_from_file_location("patch_merge_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_patch_results_replaces_tasks_without_reusing_primary_keys(tmp_path: Path) -> None:
    main_path = tmp_path / "main.db"
    patch_path = tmp_path / "patch.db"

    with ResultsDB(main_path) as main_db:
        run_id = main_db.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "MiniMaxAI/MiniMax-M2.5",
                "loop_budget": 15,
                "retrieval": False,
                "timeout_s": 300,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        main_db.save_task_result(
            run_id,
            {
                "task_id": "task-old",
                "passed": False,
                "attempts_used": 1,
                "time_ms": 10,
                "exit_code": 1,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": "Not resolved",
                "code_sha256": "old",
            },
        )
        main_db.save_task_result(
            run_id,
            {
                "task_id": "task-keep",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 20,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "keep",
            },
        )

    with ResultsDB(patch_path) as patch_db:
        run_id = patch_db.start_run(
            "swebench-lite",
            {
                "backend_name": "openai",
                "model_id": "MiniMaxAI/MiniMax-M2.5",
                "loop_budget": 15,
                "retrieval": False,
                "timeout_s": 300,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
        patch_db.save_task_result(
            run_id,
            {
                "task_id": "task-old",
                "passed": True,
                "attempts_used": 1,
                "time_ms": 30,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
                "error": None,
                "code_sha256": "new",
            },
        )

    module = _load_patch_merge_module()
    passed, total = module.merge_patch_results(main_path, patch_path)

    assert (passed, total) == (2, 2)

    with ResultsDB(main_path) as merged:
        rows = merged.conn.execute(
            "SELECT id, task_id, passed, code_sha256 FROM task_results ORDER BY task_id"
        ).fetchall()

    assert [(row["task_id"], row["passed"], row["code_sha256"]) for row in rows] == [
        ("task-keep", 1, "keep"),
        ("task-old", 1, "new"),
    ]
    assert len({row["id"] for row in rows}) == 2
