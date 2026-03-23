from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from mcode.agent.coding_policy import CodingPolicy, build_coding_policy
from mcode.agent.verification import VerificationPolicy, build_verification_policy


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
    use_text_tools: bool
    use_budget_warning: bool
    use_mid_nudge: bool
    workspace: object = None
    event_log: object = None
    condensed_state: object = None
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
    test_cmds: object | None = None,
    test_fn=None,
) -> CodingAgentAssembly:
    budget = max(1, session.loop_budget)
    timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))
    explore_prompt = os.environ.get("MCODE_EXPLORE_PROMPT", "0") == "1"
    use_text_tools = os.environ.get("MELLEA_TEXT_TOOLS", "0") == "1"
    use_budget_warning = os.environ.get("MCODE_BUDGET_WARNING", "1") == "1"
    use_mid_nudge = os.environ.get("MCODE_MID_NUDGE", "0") == "1"

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

    coding_policy = build_coding_policy(
        repo=repo,
        problem_statement=problem_statement,
        hints_text=hints_text,
        repo_map_text=repo_map_text,
        verification_prompt=verification_policy.prompt_block,
        explore_prompt=explore_prompt,
    )
    prompt_inputs_fn = getattr(coding_policy, "prompt_inputs", None)
    if callable(prompt_inputs_fn):
        prompt_inputs = prompt_inputs_fn()
    else:
        prompt_inputs = {
            "system_prompt": coding_policy.system_prompt,
            "goal": coding_policy.goal,
        }
    workspace, event_log, condensed_state = build_agent_runtime(
        repo=repo,
        repo_root=repo_root,
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
        workspace=workspace,
        **tool_kwargs,
    )

    return CodingAgentAssembly(
        repo=repo,
        repo_root=repo_root,
        coding_policy=coding_policy,
        verification_policy=verification_policy,
        tools=tools,
        model_options=session._model_options(system_prompt=prompt_inputs["system_prompt"]),
        loop_budget=budget,
        timeout_s=timeout_s,
        use_text_tools=use_text_tools,
        use_budget_warning=use_budget_warning,
        use_mid_nudge=use_mid_nudge,
        workspace=workspace,
        event_log=event_log,
        condensed_state=condensed_state,
    )


def build_repo_map(repo_root: str, query: str, *, max_tokens: int = 4096) -> str:
    from mellea.agent.repomap import build_repo_map as _build_repo_map

    return _build_repo_map(repo_root, query, max_tokens=max_tokens)


def build_agent_runtime(*, repo: str, repo_root: str, session):
    try:
        runtime_module = importlib.import_module("mellea.agent.runtime")
        memory_module = importlib.import_module("mellea.agent.runtime.memory")
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
            metadata={"repo": repo},
        )
        event_log = runtime_module.EventLog(workspace=workspace)
        condensed_state = memory_module.CondensedState(
            working_memory=memory_module.WorkingMemory()
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to assemble coding agent runtime state from mellea primitives"
        ) from exc

    return workspace, event_log, condensed_state


def make_agent_tools(
    repo_root: str,
    *,
    test_cmds: list[str] | None = None,
    test_fn=None,
    workspace=None,
):
    from mellea.agent.tools import make_agent_tools as _make_agent_tools

    return _make_agent_tools(
        repo_root,
        test_cmds=test_cmds,
        test_fn=test_fn,
        workspace=workspace,
    )
