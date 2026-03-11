# opaque

Functional DP-SGD library for PyTorch fine-tuning.

## Installation

```bash
pip install opaque-dp
```

This automatically installs `opaque-accounting` as a dependency.

## Development

From monorepo root:

```bash
uv sync
cd packages/opaque
uv run pytest tests/
```

## Architecture

Built on:

- `opaque.clipping` – Per-example gradient clipping
- `opaque.noise` – Gaussian + correlated noise
- `opaque.accounting` – Privacy accounting (via opaque-accounting backend)
- `opaque.sampling` – Batch sampling
