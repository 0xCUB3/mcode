from __future__ import annotations

import os

MELLEA_LOOP_DETECT_V1 = "mellea_loop_detect_v1"
MELLEA_TOOLKIT_V1 = "mellea_toolkit_v1"
REQUIRED_ARG_REPAIR_V1 = "required_arg_repair_v1"
FINALIZER_SUCCESS_GUARD_V1 = "finalizer_success_guard_v1"

_KNOWN_HARNESS_EXPERIMENTS = frozenset(
    {
        MELLEA_LOOP_DETECT_V1,
        MELLEA_TOOLKIT_V1,
        REQUIRED_ARG_REPAIR_V1,
        FINALIZER_SUCCESS_GUARD_V1,
    }
)


def parse_harness_experiments(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()

    out: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if not name or name in seen:
            continue
        if name not in _KNOWN_HARNESS_EXPERIMENTS:
            known = ", ".join(sorted(_KNOWN_HARNESS_EXPERIMENTS))
            raise ValueError(f"unknown harness experiment: {name!r}. Known: {known}")
        out.append(name)
        seen.add(name)
    return tuple(out)


def active_harness_experiments() -> tuple[str, ...]:
    return parse_harness_experiments(os.environ.get("MCODE_HARNESS_EXPERIMENTS"))
