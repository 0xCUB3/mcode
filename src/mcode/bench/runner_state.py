from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunStateUpdater:
    """Best-effort updates to the launch-state record for an active bench run."""

    run_id: str | None

    def patch_progress(self, **progress: object) -> None:
        if not self.run_id:
            return
        try:
            from mcode.bench import runstate

            runstate.patch_run(run_id=self.run_id, progress=progress)
        except Exception:
            pass

    def patch_metadata(self, **metadata: object) -> None:
        if not self.run_id:
            return
        try:
            from mcode.bench import runstate

            runstate.patch_run(run_id=self.run_id, metadata=metadata)
        except Exception:
            pass
