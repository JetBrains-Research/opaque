# Opaque

**Functional DP-SGD for PyTorch**

Opaque provides composable primitives for differentially private model training
in PyTorch: per-example gradient clipping, calibrated noise injection,
privacy accounting, and Poisson sampling. Built on `torch.func`, it uses a
functional API with explicit state -- no hooks, no subclassing, no hidden
mutation.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)
[![Accounting CI](https://github.com/JetBrains-Research/opaque/actions/workflows/accounting-ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/accounting-ci.yml)

## Installation

```bash
# Full DP-SGD library (includes PyTorch, noise, clipping, accounting)
pip install opaque-dp

# Privacy accounting only (no PyTorch required)
pip install opaque-accounting

# Development
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque && uv sync
```

> **Two packages in this repository:**
>
> | Package | PyPI name | Import | PyTorch? | Use case |
> |---------|-----------|--------|----------|----------|
> | `opaque` | `opaque-dp` | `import opaque` | ✅ required | Full DP-SGD training |
> | `opaque-accounting` | `opaque-accounting` | `import opaque_accounting` | ❌ not required | Privacy accounting only |
>
> `pip install opaque-dp` automatically installs `opaque-accounting` as a dependency.
> The Python wrapper at `opaque.accounting` (part of `opaque-dp`) adds typed constructors,
> calibration utilities, and documentation on top of the Rust engine.

## Example

A minimal DP-SGD training loop:

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()

# Calibrate noise for target privacy budget
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=10.0,
)

# DP-SGD components
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(
    stddev=result.param * clip_state.sensitivity(), key=key(42),
)

# Training loop
params = torch.randn(10, requires_grad=False)
for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads  # or use torchopt optimizer
```

## Features

- **Per-example gradient clipping** via `torch.func.vmap` + `torch.func.grad`,
  with fixed and adaptive clip norms
- **Noise injection**: Gaussian, truncated Gaussian, and matrix-factorization
  correlated noise (BandMF, BLT, dense)
- **Privacy accounting**: Rust-based PLD engine with tight composition,
  multiple privacy metrics (epsilon-delta, f-DP advantage, error rates),
  and noise calibration via binary search
- **Poisson sampling**: standard, truncated, and cyclic variants
- **Privacy auditing**: empirical privacy validation via membership inference
- **Distributed training**: DDP-compatible with synchronized noise and
  gradient aggregation
- **HuggingFace compatibility**: automatic `vmap` patching for LLaMA, Mistral,
  Qwen2, Phi, OLMo, Gemma2

## Documentation

- [Getting Started](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Tutorials](docs/tutorials/README.md)
- [API Reference](docs/api/index.md)
- [Examples](examples/)

## Development

```bash
# Full opaque library
uv sync --group test                       # Install test dependencies
uv run pytest                              # Run all tests
uv run pytest -m "not slow"               # Skip slow tests
uv run ruff format src/ tests/             # Format
uv run ruff check src/ tests/             # Lint

# opaque-accounting crate (standalone, no PyTorch)
cd crates/dp-accounting
cargo test                                 # Rust unit tests
cargo clippy -- -D warnings               # Rust lint
uv run maturin develop --release          # Build Python extension
uv run pytest tests/                      # PyO3 binding smoke tests
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow.

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (Abadi et al. 2016)
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) -- original inspiration
- [Opacus](https://opacus.ai/) -- alternative PyTorch DP library (hook-based design)

## License

Apache 2.0. See [LICENSE](./LICENSE).
