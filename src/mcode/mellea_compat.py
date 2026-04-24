from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


def import_requirements():
    import mellea.stdlib.components.instruction  # noqa: F401
    import mellea.stdlib.requirements as requirements

    return requirements


def import_sampling():
    import mellea.stdlib.functional  # noqa: F401
    import mellea.stdlib.sampling as sampling

    return sampling


def requirements_available() -> bool:
    try:
        import_requirements()
        return True
    except Exception:
        return False


def sampling_available() -> bool:
    try:
        import_sampling()
        return True
    except Exception:
        return False


def hooks_available() -> bool:
    try:
        import cpex.framework.base  # noqa: F401

        return True
    except Exception:
        return False


def apply_runtime_patches() -> None:
    _patch_tool_validation()
    _patch_openai_tool_ordering()


async def acall_tools(result, backend):
    import mellea.stdlib.functional as functional

    _normalize_tool_calls(result)
    return await functional._acall_tools(result, backend)


def _normalize_tool_calls(result) -> None:
    tool_calls = getattr(result, "tool_calls", None)
    if not isinstance(tool_calls, dict):
        return
    for tool_call in tool_calls.values():
        if not hasattr(tool_call, "args"):
            continue
        args = getattr(tool_call, "args", None)
        if isinstance(args, Mapping):
            continue
        tool_call.args = _coerce_tool_args(tool_call)


def _coerce_tool_args(tool_call) -> dict[str, Any]:
    args = getattr(tool_call, "args", None)
    func = getattr(tool_call, "func", None)
    param_names = _tool_param_names(func)
    required_param_names = _tool_required_param_names(func)

    return _coerce_raw_tool_args(args, param_names, required_param_names)


def _coerce_raw_tool_args(
    args: object,
    param_names: list[str],
    required_param_names: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return dict(args)

    if isinstance(args, str):
        text = args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
    else:
        parsed = args

    if isinstance(parsed, Mapping):
        return dict(parsed)

    target_param_names = required_param_names or param_names
    if len(target_param_names) == 1 and parsed is not None:
        return {target_param_names[0]: parsed}

    return {}


def build_tool_from_callable(func: Callable[..., Any], *, name: str | None = None):
    from mellea.backends.tools import MelleaTool

    tool = MelleaTool.from_callable(func, name=name)
    return MelleaTool(
        tool.name,
        func,
        _patch_tool_schema_defaults(tool.as_json_tool, func),
    )


def _tool_schema(func) -> Mapping[str, Any] | None:
    schema = getattr(func, "as_json_tool", None)
    if not isinstance(schema, Mapping):
        return None
    function_schema = schema.get("function", {})
    if not isinstance(function_schema, Mapping):
        return None
    parameters = function_schema.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return None
    return parameters


def _tool_param_names(func) -> list[str]:
    parameters = _tool_schema(func)
    if parameters is None:
        return []
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        return []
    return [str(name) for name in properties]


def _tool_required_param_names(func) -> list[str]:
    parameters = _tool_schema(func)
    if parameters is None:
        return []
    required = parameters.get("required", [])
    if not isinstance(required, list):
        return []
    return [str(name) for name in required if isinstance(name, str)]


def _patch_tool_schema_defaults(
    schema: object,
    func: Callable[..., Any],
) -> dict[str, Any] | object:
    if not isinstance(schema, Mapping):
        return schema
    patched = deepcopy(schema)
    function_schema = patched.get("function")
    if not isinstance(function_schema, dict):
        return patched
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return patched

    required: list[str] = []
    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect._empty:
            required.append(name)
    parameters["required"] = required
    return patched


def _drop_unspecified_optional_nones(
    validated_args: object,
    provided_args: Mapping[str, Any],
) -> object:
    if not isinstance(validated_args, Mapping):
        return validated_args
    return {
        key: value
        for key, value in validated_args.items()
        if key in provided_args or value is not None
    }


def _patch_tool_validation() -> None:
    import mellea.backends.tools as backend_tools
    import mellea.helpers.openai_compatible_helpers as helpers

    if not getattr(backend_tools, "_mcode_tool_validation_patch", False):
        original_validate = backend_tools.validate_tool_arguments

        def wrapped_validate(tool, args, *args2: Any, **kwargs: Any) -> object:
            param_names = _tool_param_names(tool)
            required_param_names = _tool_required_param_names(tool)
            normalized_args = _coerce_raw_tool_args(args, param_names, required_param_names)
            validated_args = original_validate(tool, normalized_args, *args2, **kwargs)
            return _drop_unspecified_optional_nones(validated_args, normalized_args)

        backend_tools.validate_tool_arguments = wrapped_validate
        backend_tools._mcode_tool_validation_patch = True

    helpers.validate_tool_arguments = backend_tools.validate_tool_arguments
    _patch_cached_tool_validators(backend_tools.validate_tool_arguments)


def _patch_cached_tool_validators(validate_tool_arguments: Callable[..., object]) -> None:
    for module_name in (
        "mellea.backends.litellm",
        "mellea.backends.utils",
        "mellea.backends.watsonx",
        "mellea.helpers.openai_compatible_helpers",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "validate_tool_arguments"):
            module.validate_tool_arguments = validate_tool_arguments


def _patch_openai_tool_ordering() -> None:
    import mellea.backends.openai as openai_backend

    if getattr(openai_backend, "_mcode_tool_ordering_patch", False):
        return

    def fixed_tool_ordering(conversation: list[dict]) -> list[dict]:
        import uuid

        fixed: list[dict] = []
        for original in conversation:
            msg = deepcopy(original)
            if msg.get("role") == "tool":
                if not msg.get("tool_call_id"):
                    msg["tool_call_id"] = f"call_{uuid.uuid4().hex[:24]}"
                if not fixed or fixed[-1].get("role") != "assistant":
                    fixed.append({"role": "assistant", "content": None, "tool_calls": []})
                prev = fixed[-1]
                if prev.get("tool_calls") is None:
                    prev["tool_calls"] = []
                if not prev.get("content"):
                    prev["content"] = None
                prev["tool_calls"].append(
                    {
                        "id": msg["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": msg.get("name", "unknown"),
                            "arguments": "{}",
                        },
                    }
                )
            if msg.get("role") == "user" and fixed and fixed[-1].get("role") == "tool":
                fixed.append({"role": "assistant", "content": "Continuing."})
            fixed.append(msg)
        return fixed

    openai_backend._fix_tool_call_ordering = fixed_tool_ordering
    openai_backend._mcode_tool_ordering_patch = True
