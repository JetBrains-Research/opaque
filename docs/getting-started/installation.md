# Installation

> **Work in progress:** Opaque is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

## Requirements

- Python 3.11 or 3.12
- PyTorch 2.9 or later

## From JetBrains Packages

Install `opaque` as the single public package entrypoint:

```bash
pip install opaque \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Or with uv:

```bash
uv add opaque \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

### Extras

```bash
pip install "opaque[auditing]"      # + opaque-auditing (empirical privacy auditing)
pip install "opaque[dpftrl]"        # + opaque-dpftrl (correlated-noise mechanisms)
pip install "opaque[transformers]"  # + opaque-transformers + opaque-patches[transformers]
pip install "opaque[all]"           # everything above
```

## From Source

Clone the repository when developing Opaque, inspecting its implementation, or
running its test suite. For ordinary use, prefer the published package above.

```bash
# Clone the repository
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque

# Install with uv (recommended)
uv sync --group dev --all-packages --extra all
```

## Development Installation

```bash
# Install dev dependencies for all packages
uv sync --group dev --all-packages --extra all

# Verify installation (non-GPU tests)
uv run pytest -m "not cuda and not mps and not slow"
```

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Visit <http://localhost:8000> to view the docs.

## Verify Installation

```python
from importlib.metadata import version

print("opaque-base version:", version("opaque-base"))
print("opaque-engine version:", version("opaque-engine"))
print("opaque-dpsgd version:", version("opaque-dpsgd"))
```

`opaque` itself is a [PEP 420] namespace with no top-level Python code, so
query the installed distributions individually.

## PyCharm

In PyCharm, select the `uv` interpreter for the project where you ran
`uv add`. Completion and Quick Documentation follow Opaque's public façade
imports, such as `opaque.dpsgd.clipping`, `opaque.dpsgd.noise`, and
`opaque.accounting`; avoid copying `opaque.api.*` paths from implementation
tracebacks into application code. Clone Opaque and use an editable workspace
only when you need to debug or change its implementation.

[PEP 420]: https://peps.python.org/pep-0420/
