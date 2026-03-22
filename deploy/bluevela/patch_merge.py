"""Merge patch results into main Verified DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

RESULTS_DIR = Path("/u/skula/mcode/results")


def merge_patch_results(main_path: str | Path, patch_path: str | Path) -> tuple[int, int]:
    main = sqlite3.connect(main_path)
    patch = sqlite3.connect(patch_path)
    try:
        cols = [d[0] for d in main.execute("SELECT * FROM task_results LIMIT 1").description]
        insert_cols = [col for col in cols if col != "id"]
        patch_rows = patch.execute(
            f"SELECT {','.join(insert_cols)} FROM task_results"
        ).fetchall()
        tid_idx = insert_cols.index("task_id")
        placeholders = ",".join(["?"] * len(insert_cols))
        col_names = ",".join(insert_cols)

        for row in patch_rows:
            tid = row[tid_idx]
            main.execute("DELETE FROM task_results WHERE task_id = ?", (tid,))
            main.execute(
                f"INSERT INTO task_results ({col_names}) VALUES ({placeholders})",
                row,
            )

        main.commit()
        total = main.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]
        passed = main.execute("SELECT COUNT(*) FROM task_results WHERE passed=1").fetchone()[0]
        return passed, total
    finally:
        main.close()
        patch.close()


def main() -> None:
    passed, total = merge_patch_results(
        RESULTS_DIR / "live-m25-verified.db",
        RESULTS_DIR / "live-m25-verified-patch.db",
    )
    print(f"Final: {passed}/{total} = {100 * passed / total:.1f}%")
    print("MiniMax published: 80.2%")


if __name__ == "__main__":
    main()
