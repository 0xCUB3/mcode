from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def apply_provider_compatibility_patches() -> None:
    _patch_openai_tool_validation()
    _patch_openai_tool_ordering()


async def acall_tools_with_arg_compat(result, backend):
    import mellea.stdlib.functional as functional

    _normalize_tool_calls(result)
    return await functional._acall_tools(result, backend)


def inspect_tool_call_arg_compat(tool_calls: object) -> dict[str, int]:
    if isinstance(tool_calls, Mapping):
        calls = list(tool_calls.values())
    elif isinstance(tool_calls, list | tuple):
        calls = list(tool_calls)
    else:
        return {"raw_arg_call_count": 0, "recoverable_call_count": 0}

    raw_arg_call_count = 0
    recoverable_call_count = 0
    for tool_call in calls:
        if not hasattr(tool_call, "args"):
            continue
        args = getattr(tool_call, "args", None)
        if isinstance(args, Mapping):
            continue
        raw_arg_call_count += 1
        if _coerce_tool_args(tool_call):
            recoverable_call_count += 1
    return {
        "raw_arg_call_count": raw_arg_call_count,
        "recoverable_call_count": recoverable_call_count,
    }


_TOOL_ARG_ALIASES = {
    "run_tests": {
        "timeout": "timeout_s",
        "max_output": "max_output_chars",
    },
}


def _normalize_tool_calls(result) -> None:
    tool_calls = getattr(result, "tool_calls", None)
    if not isinstance(tool_calls, dict):
        return
    for tool_call in tool_calls.values():
        if not hasattr(tool_call, "args"):
            continue
        args = getattr(tool_call, "args", None)
        if isinstance(args, Mapping):
            tool_call.args = _normalize_mapping_tool_args(
                tool_call,
                args,
                _tool_param_names(getattr(tool_call, "func", None)),
            )
            continue
        tool_call.args = _coerce_tool_args(tool_call)


def _coerce_tool_args(tool_call) -> dict[str, Any]:
    args = getattr(tool_call, "args", None)
    func = getattr(tool_call, "func", None)
    param_names = _tool_param_names(func)
    required_param_names = _tool_required_param_names(func)

    return _coerce_raw_tool_args(args, param_names, required_param_names)


def _normalize_mapping_tool_args(
    tool_or_name: object,
    args: Mapping[str, Any],
    param_names: list[str],
) -> dict[str, Any]:
    out = dict(args)
    tool_name = _tool_name(tool_or_name) if not isinstance(tool_or_name, str) else tool_or_name
    for old_name, new_name in _TOOL_ARG_ALIASES.get(tool_name, {}).items():
        if old_name in out and new_name in param_names and new_name not in out:
            out[new_name] = out.pop(old_name)
    if not param_names:
        return out
    return {key: value for key, value in out.items() if key in param_names}


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


def _tool_name(func) -> str:
    schema = getattr(func, "as_json_tool", None)
    if isinstance(schema, Mapping):
        function_schema = schema.get("function", {})
        if isinstance(function_schema, Mapping):
            name = function_schema.get("name")
            if isinstance(name, str) and name:
                return name
    return str(getattr(func, "name", ""))


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


def _patch_openai_tool_validation() -> None:
    import mellea.helpers.openai_compatible_helpers as helpers

    if getattr(helpers, "_mcode_tool_validation_patch", False):
        return

    original_validate = helpers.validate_tool_arguments

    def wrapped_validate(tool, args, *args2, **kwargs):
        param_names = _tool_param_names(tool)
        required_param_names = _tool_required_param_names(tool)
        normalized_args = _coerce_raw_tool_args(args, param_names, required_param_names)
        normalized_args = _normalize_mapping_tool_args(
            tool,
            normalized_args,
            param_names,
        )
        if (
            _tool_name(tool) == "final_answer"
            and "answer" in required_param_names
            and "answer" not in normalized_args
        ):
            normalized_args["answer"] = "Done."
        validated_args = original_validate(tool, normalized_args, *args2, **kwargs)
        return _drop_unspecified_optional_nones(validated_args, normalized_args)

    helpers.validate_tool_arguments = wrapped_validate
    helpers._mcode_tool_validation_patch = True


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
