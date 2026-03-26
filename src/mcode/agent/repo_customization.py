from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoCustomization:
    text: str = ""
    source_path: Path | None = None


def load_repo_customization(repo_root: str) -> RepoCustomization:
    candidate = Path(repo_root) / ".mcode" / "repo.md"
    if not candidate.exists():
        return RepoCustomization()

    text = candidate.read_text().strip()
    if not text:
        return RepoCustomization()

    return RepoCustomization(text=text, source_path=candidate)
