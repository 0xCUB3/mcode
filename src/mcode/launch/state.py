"""Atomic JSON state file with fcntl locking.

State shape (JSON):

    {
      "servers": [ServerRecord, ...],
      "runs":    [RunRecord,    ...]
    }

Invariants:

- Writes are atomic (tmp file + rename).
- All reads/writes are wrapped in an fcntl exclusive lock on a sibling
  .lock file. No concurrent writers on the same machine.
- The state file lives at default_state_path() by default; override for tests.

The fcntl/atomic-write logic is the one piece consciously ported from main's
launcher — it was proven across 19 bug-fix commits.
"""

from __future__ import annotations
