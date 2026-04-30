# mcode

## Git

- Commit messages: one short line describing what changed. No body unless large/complex.
- No conventional commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`, etc.)
- No plan or phase references in commit messages. Never include plan numbers, phase numbers, or plan names.
- No words like "enhance", "streamline", "robust", "leverage", "comprehensive".

## Code style

- Match existing patterns. No docstrings or comments beyond what already exists.
- All new deps as optional extras in pyproject.toml, lazy imports only.
- Tests mock all external calls, no network.
- Must pass `ruff check` and `ruff format --check` before each commit.
