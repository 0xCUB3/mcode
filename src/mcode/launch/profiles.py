"""Model -> ServingProfile registry.

Every entry is tested with a golden fixture (see tests/launch/fixtures/).
Add a model here whenever you need mcode to serve it.

Design note: profile pattern matching uses first-match-wins order. Put more
specific patterns before generic ones.
"""

from __future__ import annotations

from fnmatch import fnmatch

from mcode.launch.models import ServingProfile

# Default container image when a profile doesn't override it.
DEFAULT_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.17.0"

# Order matters: more specific patterns first.
_PROFILES: list[tuple[str, ServingProfile]] = [
    # Qwen3.5-35B-A3B (MoE) — needs expert parallelism on TP >= 4.
    # At TP=2 (the H100-80GB sweet spot) plain tool-call flags are enough.
    (
        "qwen/qwen3.5-35b-a3b*",
        ServingProfile(
            name="qwen3.5-a3b",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3",
            ],
            tensor_parallel=2,
            max_model_len=131072,
            min_vllm="0.11.0",
        ),
    ),
    (
        "qwen/qwen3.5-27b*",
        ServingProfile(
            name="qwen3.5-27b",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3",
            ],
            tensor_parallel=2,
            max_model_len=131072,
            min_vllm="0.11.0",
        ),
    ),
    # Catch-all for other Qwen3.x instruct variants.
    (
        "qwen/qwen3*",
        ServingProfile(
            name="qwen3",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3",
            ],
            tensor_parallel=2,
            max_model_len=32768,
            min_vllm="0.11.0",
        ),
    ),
    # Gemma 4 instruct — chat template is REQUIRED for tool calls (vLLM #39043).
    # Native Gemma4 support landed in vLLM v0.19.0 (PR #38826, 2026-04-02).
    # Pin to the exact tag: `:latest` and `:nightly` on DockerHub have been
    # observed to ship older transformers versions that don't register the
    # `gemma4` arch, causing a pydantic ValidationError at engine init.
    #
    # Flash-attn can't handle Gemma4's head_dim=512; fall back to XFORMERS
    # (or FLASHINFER). See transformers issue #45202.
    (
        "google/gemma-4*",
        ServingProfile(
            name="gemma4",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "functiongemma",
                "--attention-backend",
                "TORCH_SDPA",
            ],
            tensor_parallel=2,
            max_model_len=32768,
            chat_template="tool_chat_template_gemma4.jinja",
            image="docker.io/vllm/vllm-openai:gemma4",
            min_vllm="0.19.0+gemma4",
        ),
    ),
    # Granite 4.x — uses hermes parser (NOT "granite", which is 3.x).
    (
        "ibm-granite/granite-4*",
        ServingProfile(
            name="granite4",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ],
            tensor_parallel=1,
            max_model_len=32768,
            min_vllm="0.10.2",
        ),
    ),
    # Granite 3.x (older generation, different parser).
    (
        "ibm-granite/granite-3*",
        ServingProfile(
            name="granite3",
            flags=[
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "granite",
            ],
            tensor_parallel=1,
            max_model_len=32768,
        ),
    ),
    # MiniMax M2 / M2.5 — MoE, needs expert parallel + SAFETENSORS_FAST_GPU +
    # a compilation-config escape hatch for illegal memory access errors.
    # Nightly vLLM required (past commit cf3eacfe58fa9e).
    (
        "minimaxai/minimax-m2*",
        ServingProfile(
            name="minimax-m2",
            flags=[
                "--trust-remote-code",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "minimax_m2",
                "--reasoning-parser",
                "minimax_m2_append_think",
                "--enable_expert_parallel",
                "--compilation-config",
                '{"cudagraph_mode":"PIECEWISE"}',
            ],
            tensor_parallel=4,
            max_model_len=32768,
            extra_env={"SAFETENSORS_FAST_GPU": "1"},
            min_vllm="nightly@cf3eacfe",
        ),
    ),
]


def resolve(model: str) -> ServingProfile:
    """Return the ServingProfile for a model id, first-match-wins.

    Fallback is a minimal default profile suitable for a small local test
    model. Callers that care about correctness for a specific model should
    add it to _PROFILES above and gate it with a golden-fixture test.
    """

    m = model.lower()
    for pattern, profile in _PROFILES:
        if fnmatch(m, pattern.lower()):
            return ServingProfile(
                name=profile.name,
                flags=list(profile.flags),
                tensor_parallel=profile.tensor_parallel,
                max_model_len=profile.max_model_len,
                extra_env=dict(profile.extra_env),
                chat_template=profile.chat_template,
                min_vllm=profile.min_vllm,
                image=profile.image,
            )
    return ServingProfile(
        name="default",
        flags=[],
        tensor_parallel=1,
        max_model_len=8192,
    )
