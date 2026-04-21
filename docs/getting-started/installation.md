# Installation

## Requirements

- Python 3.11 or 3.12
- PyTorch 2.10 or later

## From GCP Artifact Registry

Install the umbrella meta-package (pulls in `opaque-core`, `opaque-dpsgd`,
`opaque-auditing`, and `opaque-accounting` by default):

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
pip install "opaque[dpftrl]"        # + opaque-dpftrl (correlated-noise mechanisms)
pip install "opaque[performance]"   # + opaque-performance (fused Triton kernels + HF kernel patches + checkpoint patches)
pip install "opaque[huggingface]"   # + opaque-huggingface (Transformers compat patches) + performance
pip install "opaque[all]"           # everything above, including [optimizers] extras
```

Each sub-package is also installable directly, e.g. `pip install opaque-core`
or `pip install opaque-dpsgd`.

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
import opaque.core
import opaque.dpsgd

print("opaque.core version:", opaque.core.__version__)
print("opaque.dpsgd version:", opaque.dpsgd.__version__)
```
