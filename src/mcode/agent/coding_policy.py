from __future__ import annotations

from dataclasses import dataclass

_BASE_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in an open-source repository. "
    "You MUST edit existing source files to fix the bug. Do NOT create new files. "
    "Do NOT write test scripts. Only modify the existing code that contains the bug.\n\n"
    "Use the structured code tools to search, read, edit, and verify. Start narrow, make one "
    "concrete edit once you have a target, then use `run_tests` before `final_answer`. "
    "When you call `final_answer`, keep the answer short."
)


@dataclass(frozen=True)
class CodingPolicy:
    system_prompt: str
    goal: str

    def prompt_inputs(self) -> dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "goal": self.goal,
        }


def build_system_prompt() -> str:
    return _BASE_SYSTEM_PROMPT


def build_goal(
    *,
    repo: str,
    problem_statement: str,
    repo_map_text: str = "",
    candidate_files_text: str = "",
    hints_text: str = "",
    repo_customization_text: str = "",
    workspace_context_text: str = "",
    verification_prompt: str = "",
) -> str:
    repo_map_block = f"\n\nRepository structure:\n{repo_map_text}" if repo_map_text else ""
    candidate_files_block = (
        f"\n\n{candidate_files_text}\nStart with one of these before widening the search. "
        "Do not read more than three candidate files before you either edit or have a narrower "
        "hypothesis."
        if candidate_files_text
        else ""
    )
    hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""
    customization_block = (
        f"\n\nRepository-specific guidance:\n{repo_customization_text.strip()}"
        if repo_customization_text.strip()
        else ""
    )
    workspace_context_block = (
        f"\n\n{workspace_context_text.strip()}" if workspace_context_text.strip() else ""
    )
    return (
        f"Fix this bug in {repo} by editing the existing source code.\n\n"
        f"Issue:\n{problem_statement.strip()}"
        f"{repo_map_block}{candidate_files_block}{hints_block}{customization_block}"
        f"{workspace_context_block}{verification_prompt}\n\n"
        "Do not open a second solving path. Diagnose, edit, verify, then submit."
    )


def build_coding_policy(
    *,
    repo: str,
    problem_statement: str,
    hints_text: str = "",
    repo_map_text: str = "",
    candidate_files_text: str = "",
    repo_customization_text: str = "",
    workspace_context_text: str = "",
    verification_prompt: str = "",
) -> CodingPolicy:
    return CodingPolicy(
        system_prompt=build_system_prompt(),
        goal=build_goal(
            repo=repo,
            problem_statement=problem_statement,
            hints_text=hints_text,
            repo_map_text=repo_map_text,
            candidate_files_text=candidate_files_text,
            repo_customization_text=repo_customization_text,
            workspace_context_text=workspace_context_text,
            verification_prompt=verification_prompt,
        ),
    )
