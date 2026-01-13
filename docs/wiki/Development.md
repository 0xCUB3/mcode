# Development

Install:

```bash
uv pip install -e '.[dev]'
```

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check src tests
```

Generate wiki pages locally:

```bash
uv run python scripts/generate_cli_docs.py
```
