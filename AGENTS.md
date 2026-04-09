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

## Run workflows

- Prefer `uv run mcode ...` commands over raw cluster commands in docs, examples, and day-to-day instructions.
- For Blue Vela runs, prefer:
  - `uv run mcode launch doctor`
  - `uv run mcode launch sync`
  - `uv run mcode launch`
  - `uv run mcode launch status`
  - `uv run mcode launch attach`
  - `uv run mcode launch fetch`
  - `uv run mcode launch stop`
- Use raw `ssh`, `rsync`, `bsub`, `bjobs`, `podman`, or the legacy scripts in `deploy/bluevela/` only when debugging the launcher or when the user explicitly asks for the old flow.
- For dependency setup, prefer `uv run mcode deps sync ...` over handwritten environment bootstrap commands.

## Docs maintenance

- When adding or changing user-facing run commands, update `docs/COMMANDS.md`.
- If the change affects primary usage, also update `README.md`.
- If the change is Blue Vela-specific, also update `deploy/bluevela/README.md`.
- Historical research notes under `research/` are records, not the source of truth for current operational commands.
