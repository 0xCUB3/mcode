from __future__ import annotations

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

    tools = make_agent_tools(
        repo_root,
        test_cmds=verification_policy.test_cmds,
        test_fn=verification_policy.test_fn,
    )

    return CodingAgentAssembly(
        repo=repo,
        repo_root=repo_root,
        coding_policy=coding_policy,
        verification_policy=verification_policy,
        tools=tools,
        model_options=session._model_options(system_prompt=coding_policy.system_prompt),
        loop_budget=budget,
        timeout_s=timeout_s,
        use_text_tools=use_text_tools,
        use_budget_warning=use_budget_warning,
        use_mid_nudge=use_mid_nudge,
    )


def build_repo_map(repo_root: str, query: str, *, max_tokens: int = 4096) -> str:
    from mellea.agent.repomap import build_repo_map as _build_repo_map

    return _build_repo_map(repo_root, query, max_tokens=max_tokens)


def make_agent_tools(
    repo_root: str,
    *,
    test_cmds: list[str] | None = None,
    test_fn=None,
):
    from mellea.agent.tools import make_agent_tools as _make_agent_tools

    return _make_agent_tools(repo_root, test_cmds=test_cmds, test_fn=test_fn)
