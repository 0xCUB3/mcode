from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from mcode.agent.coding_policy import CodingPolicy, build_coding_policy
from mcode.agent.repo_customization import load_repo_customization
from mcode.agent.verification import (
    VerificationPolicy,
    build_run_tests_tool,
    build_verification_policy,
)


@dataclass(frozen=True)
class CodingAgentAssembly:
    repo: str
    repo_root: str
    coding_policy: CodingPolicy
    verification_policy: VerificationPolicy
    tools: list
    model_options: dict
    loop_budget: int
    timeout_s: int
    workspace: object = None
    event_log: object = None
    condensed_state: object = None
    condensation: object = None
    capability_contract: object = None
    max_retries_per_turn: int = 0

    @property
    def system_prompt(self) -> str:
        return self.coding_policy.system_prompt

    @property
    def goal(self) -> str:
        return self.coding_policy.goal

    @property
    def verification_cmds(self) -> list[str]:
        return self.verification_policy.test_cmds

    @property
    def verification_test_fn(self):
        return self.verification_policy.test_fn


def build_coding_agent(
    *,
    session,
    repo: str,
    problem_statement: str,
    hints_text: str = "",
    repo_root: str,
    visible_repo_root: str | None = None,
    test_cmds: object | None = None,
    test_fn=None,
    command_fn: Callable[[str], str] | None = None,
) -> CodingAgentAssembly:
    budget = max(1, session.loop_budget)
    timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))

    verification_policy = build_verification_policy(
        repo_root=repo_root,
        test_cmds=test_cmds,
        test_fn=test_fn,
        timeout_s=timeout_s,
    )

    repo_map_text = ""
    try:
        repo_map_text = build_repo_map(repo_root, problem_statement, max_tokens=4096)
    except Exception as e:
        print(f"  [repo_map] failed: {e}", flush=True)
    repo_customization = load_repo_customization(repo_root)

    coding_policy = build_coding_policy(
        repo=repo,
        problem_statement=problem_statement,
        hints_text=hints_text,
        repo_map_text=repo_map_text,
        repo_customization_text=repo_customization.text,
        verification_prompt=verification_policy.prompt_block,
    )
    prompt_inputs_fn = getattr(coding_policy, "prompt_inputs", None)
    if callable(prompt_inputs_fn):
        prompt_inputs = prompt_inputs_fn()
    else:
        prompt_inputs = {
            "system_prompt": coding_policy.system_prompt,
            "goal": coding_policy.goal,
        }
    workspace, event_log, condensed_state, condensation = build_agent_runtime(
        repo=repo,
        repo_root=repo_root,
        visible_repo_root=visible_repo_root,
        session=session,
    )

    tool_kwargs_fn = getattr(verification_policy, "tool_kwargs", None)
    if callable(tool_kwargs_fn):
        tool_kwargs = tool_kwargs_fn()
    else:
        tool_kwargs = {
            "test_cmds": verification_policy.test_cmds,
            "test_fn": verification_policy.test_fn,
        }

    tools = make_agent_tools(
        repo_root,
        command_fn=command_fn,
        workspace=workspace,
        **tool_kwargs,
    )
    capability_contract = build_orchestrator_contract(
        tool_names=[getattr(tool, "name", "") for tool in tools],
        default_verification_commands=verification_policy.test_cmds,
    )
    workspace_metadata = getattr(workspace, "metadata", None)
    if isinstance(workspace_metadata, dict):
        workspace_metadata["orchestrator_contract"] = capability_contract.snapshot()

    return CodingAgentAssembly(
        repo=repo,
        repo_root=repo_root,
        coding_policy=coding_policy,
        verification_policy=verification_policy,
        tools=tools,
        model_options=session._model_options(system_prompt=prompt_inputs["system_prompt"]),
        loop_budget=budget,
        timeout_s=timeout_s,
        workspace=workspace,
        event_log=event_log,
        condensed_state=condensed_state,
        condensation=condensation,
        capability_contract=capability_contract,
    )


def build_repo_map(repo_root: str, query: str, *, max_tokens: int = 4096) -> str:
    from mellea.agent.repomap import build_repo_map as _build_repo_map

    return _build_repo_map(repo_root, query, max_tokens=max_tokens)


def build_agent_runtime(
    *,
    repo: str,
    repo_root: str,
    visible_repo_root: str | None = None,
    session,
):
    try:
        runtime_module = importlib.import_module("mellea.agent.runtime")
        memory_module = importlib.import_module("mellea.agent.runtime.memory")
        loops_module = importlib.import_module("mellea.agent.runtime.loops")
    except Exception as exc:
        raise RuntimeError(
            "mellea runtime primitives are required for coding agent assembly"
        ) from exc

    try:
        safety_policy = runtime_module.SafetyPolicy(
            mode=os.environ.get("MCODE_RUNTIME_SAFETY_MODE", "workspace-write"),
            network_access=True,
            writable_roots=(repo_root,),
        )
        workspace = runtime_module.Workspace(
            cwd=repo_root,
            safety_policy=safety_policy,
            session=runtime_module.SessionMetadata(
                executor="mcode",
                metadata={
                    "repo": repo,
                    "backend_name": session.backend_name,
                    "model_id": session.model_id,
                },
            ),
            metadata={
                "repo": repo,
                **({"display_cwd": visible_repo_root} if visible_repo_root is not None else {}),
            },
        )
        event_log = runtime_module.EventLog(workspace=workspace)
        condensed_state = memory_module.CondensedState(working_memory=memory_module.WorkingMemory())
        condensation = loops_module.CondensationConfig(
            working_memory=memory_module.WorkingMemory(),
            max_messages=int(os.environ.get("MCODE_CONDENSE_MAX_MESSAGES", "24")),
            preserve_recent=int(os.environ.get("MCODE_CONDENSE_PRESERVE_RECENT", "6")),
            preserve_head=int(os.environ.get("MCODE_CONDENSE_PRESERVE_HEAD", "2")),
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to assemble coding agent runtime state from mellea primitives"
        ) from exc

    return workspace, event_log, condensed_state, condensation


def build_orchestrator_contract(
    *,
    tool_names: list[str],
    default_verification_commands: list[str],
):
    try:
        capability_module = importlib.import_module("mellea.agent.capabilities")
    except Exception as exc:
        raise RuntimeError(
            "mellea capability contract primitives are required for coding agent assembly"
        ) from exc

    try:
        return capability_module.OrchestratorContract.from_tool_names(
            tool_names,
            default_verification_commands=default_verification_commands,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to assemble the orchestrator capability contract from mellea primitives"
        ) from exc


def make_agent_tools(
    repo_root: str,
    *,
    test_cmds: list[str] | None = None,
    test_fn=None,
    command_fn: Callable[[str], str] | None = None,
    workspace=None,
):
    from mellea.agent.tools import make_agent_tools as _make_agent_tools

    tools = _make_agent_tools(
        repo_root,
        test_cmds=test_cmds,
        test_fn=test_fn,
        command_fn=command_fn,
        workspace=workspace,
    )
    if test_fn is not None or (not test_cmds and command_fn is None):
        return tools

    hardened_run_tests = build_run_tests_tool(
        repo_root=repo_root,
        test_cmds=test_cmds or [],
        command_fn=command_fn,
        workspace=workspace,
    )
    replaced = False
    out = []
    for tool in tools:
        if getattr(tool, "name", "") == "run_tests":
            out.append(hardened_run_tests)
            replaced = True
            continue
        out.append(tool)
    if not replaced:
        out.append(hardened_run_tests)
    return out
