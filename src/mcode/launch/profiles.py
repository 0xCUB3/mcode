from __future__ import annotations

from fnmatch import fnmatch

from mcode.launch.models import ServingProfile

_PROFILE_PATTERNS: list[tuple[str, ServingProfile]] = [
    (
        "*qwen3*",
        ServingProfile(
            name="qwen3",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3",
            ],
        ),
    ),
    (
        "google/gemma-4*",
        ServingProfile(
            name="gemma4",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "gemma4",
                "--reasoning-parser",
                "gemma4",
            ],
        ),
    ),
]


def resolve_serving_profile(model: str) -> ServingProfile:
    model_lower = model.lower()
    for pattern, profile in _PROFILE_PATTERNS:
        if fnmatch(model_lower, pattern.lower()):
            return ServingProfile(name=profile.name, flags=list(profile.flags))
    return ServingProfile()
