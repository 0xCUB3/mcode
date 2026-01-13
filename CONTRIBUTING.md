# Contributing

## Setup

```bash
uv pip install -e '.[dev]'
```

## Run checks

```bash
uv run pytest
uv run ruff check src tests
```

## Updating documentation

Wiki pages live in `docs/wiki/`. The `CLI.md` page is generated:

```bash
uv run python scripts/generate_cli_docs.py
```

If this repository is configured with a GitHub Wiki, it is auto-synced from `docs/wiki/` on pushes to
`main` via `.github/workflows/wiki-sync.yml`.
