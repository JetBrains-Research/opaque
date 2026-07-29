# Development

This guide is for contributors working from a GitHub fork or a local clone of
Opaque. It summarizes the maintained workflow in
[CONTRIBUTING.md](https://github.com/JetBrains-Research/opaque/blob/main/CONTRIBUTING.md);
consult that file for the full test, documentation, and pull-request conventions.

## Get the source

Fork [JetBrains-Research/opaque](https://github.com/JetBrains-Research/opaque),
then clone your fork and add the upstream repository if you use it to keep your
branch current:

```bash
git clone https://github.com/<your-account>/opaque.git
cd opaque
git remote add upstream https://github.com/JetBrains-Research/opaque.git
```

Opaque requires Python 3.11 or 3.12, Rust stable, and
[uv](https://docs.astral.sh/uv/). Create the complete contributor environment:

```bash
uv sync --group dev --all-packages --extra all
```

Add only the dependency groups required for the work at hand:

```bash
uv sync --group docs --all-packages
uv sync --group examples --all-packages --extra all
```

## Navigate the workspace

Opaque is a `uv` workspace of independently installable packages that
contribute to the shared `opaque` and `opaque.api` PEP 420 namespaces. Package
implementations live under `packages/opaque-*/src/opaque/api/`; the matching
`src/opaque/` directories are public façades that re-export the supported user
API.

Keep the namespace roots implicit: do not add `opaque/__init__.py`,
`opaque/api/__init__.py`, or `opaque/api/accounting/__init__.py`. Place tests
in the package that owns every dependency they import, and use public façade
imports in user-facing documentation and examples.

The accounting package includes the Rust/PyO3 extension. Run its Cargo tests
from the repository root along with the relevant Python tests when changing the
native boundary.

## Validate a change

Run the smallest relevant package test first, then the normal contributor
checks before opening a pull request:

```bash
uv run pytest -m "not cuda and not mps and not slow"
uv run ruff check packages/
uv run ruff format --check packages/
uv run --group docs mkdocs build --strict
cargo test --workspace
```

CUDA, MPS, and slow tests have separate pytest markers. Tests that require a
hardware backend automatically skip when it is unavailable; see the
[test-marker reference](https://github.com/JetBrains-Research/opaque/blob/main/CONTRIBUTING.md#test-markers-and-filtering)
for the available selections.

## Submit a pull request

Use a focused branch from your fork, add regression coverage for behavior
changes, and ensure public API additions include type annotations and
Google-style docstrings. The pull-request title follows Conventional Commits,
for example `fix(accounting): handle empty compositions`; its body should
briefly explain the problem and solution. See the
[pull-request checklist](https://github.com/JetBrains-Research/opaque/blob/main/CONTRIBUTING.md#pull-request-process)
for the complete checklist.
