from __future__ import annotations

from mcode.launch.models import CommandResult, OpenAICompatibleTargetSpec


def openai_compatible_doctor_result(target: OpenAICompatibleTargetSpec) -> CommandResult:
    return CommandResult(
        ok=True,
        message="OpenAI-compatible endpoint configured.",
        data={"base_url": target.base_url, "api_key_env": target.api_key_env},
    )
