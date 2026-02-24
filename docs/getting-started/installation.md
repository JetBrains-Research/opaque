# Installation

## Requirements

- Python 3.11 or 3.12
- PyTorch 2.0 or later

## From Source

Install version `0.1.0+test` from the JetBrains Artifact Registry:

```bash
pip install opaque-dp==0.1.0+test \
  --extra-index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Development Installation

For development, install with dev dependencies:

```bash
# Install dev dependencies
uv sync --group dev

# Verify installation
uv run pytest
```

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Visit http://localhost:8000 to view the docs.

## Verify Installation

```python
import opaque

print("Opaque import OK:", bool(opaque))
```
