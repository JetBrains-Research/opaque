# Contributing

This guide is the minimal contributor reference for Opaque.

## Setup

```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync
```

## Tests

```bash
uv run pytest
```

## Formatting and linting

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
```

## Docs

```bash
uv sync --group docs
uv run mkdocs serve
```

## Contribution checklist

- Add tests for new functionality.
- Update docs and docstrings for user-facing changes.
- Keep APIs explicit and functional.
