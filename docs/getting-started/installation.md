# Installation

## Requirements

- Python 3.11 or 3.12
- PyTorch 2.10 or later

## From GCP Artifact Registry

Install `opaque` as the single public package entrypoint:

```bash
pip install opaque \
  --extra-index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/
```

Or with uv:

```bash
uv add opaque \
  --index https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/
```

### Extras

```bash
pip install "opaque[auditing]"      # + opaque-auditing (empirical privacy auditing)
pip install "opaque[dpftrl]"        # + opaque-dpftrl (correlated-noise mechanisms)
pip install "opaque[transformers]"  # + opaque-transformers + opaque-patches[transformers]
pip install "opaque[all]"           # everything above, including [optimizers] extras
```

## From Source

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

[PEP 420]: https://peps.python.org/pep-0420/
