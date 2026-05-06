# Opaque

Functional DP-SGD and DP-FTRL for PyTorch.

Opaque provides composable primitives for differentially private model
training in PyTorch: per-example gradient clipping, calibrated noise
injection, privacy accounting, and Poisson sampling. Built on `torch.func`,
it uses a functional API with explicit state — no hooks, no subclassing, no
hidden mutation.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10+](https://img.shields.io/badge/pytorch-2.10+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)

## Packages

Install and depend on `opaque` only. The repository is implemented as
[PEP 420] namespace sub-packages under the shared `opaque.*` namespace:

| Distribution | Import roots | Purpose |
|---|---|---|
| `opaque` | — | Convenience installer; pulls in a curated bundle of sub-packages |
| `opaque-core` | `opaque.{clipping,random,pytree,functional,distributed,optimizers,scheduling,profiling,types}` | RNG, pytree, clipping, `PerGroup`, `empty_collate`, `make_functional`, DDP plumbing, optimizers |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / truncated / per-group noise, AdamW-BC, Poisson samplers, adaptive + auto clipping |
| `opaque-dpftrl` | `opaque.dpftrl` | DP-FTRL mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, correlated-noise samplers |
| `opaque-auditing` | `opaque.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) |
| `opaque-patches` | `opaque.patches` | Unified patching entrypoint for PyTorch checkpointing, Hugging Face compat wrappers, Triton kernels, and PEFT/LoRA fusion |
| `opaque-transformers` | `opaque.transformers` | Transformers dependency bundle and namespace package for Hugging Face integrations |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (Rust/PyO3 backend) |

[PEP 420]: https://peps.python.org/pep-0420/

### Import layout

```
opaque.{clipping,random,pytree,types}                      <- opaque-core
opaque.distributed.{collectives,gradients,state,shard}     <- opaque-core
opaque.functional                                          <- opaque-core
opaque.optimizers                                          <- opaque-core
opaque.scheduling                                          <- opaque-core
opaque.profiling                                           <- opaque-core
opaque.dpsgd.{noise,clipping,sampling,optimizers}          <- opaque-dpsgd
opaque.dpftrl.{noise,sampling,optimizers}                  <- opaque-dpftrl
opaque.auditing                                            <- opaque-auditing
opaque.patches.{kernels,torch,transformers,peft}           <- opaque-patches
opaque.transformers                                        <- opaque-transformers
opaque.accounting (._native)                               <- opaque-accounting
```

## Installation

```bash
# From the JetBrains Artifact Registry
pip install opaque \
  --extra-index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/

# Or with uv
uv add opaque \
  --index https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/
```

Extras:

```bash
pip install "opaque[auditing]"      # auditing components
pip install "opaque[dpftrl]"        # correlated-noise DP-FTRL components
pip install "opaque[transformers]"  # Hugging Face + patching components
pip install "opaque[all]"           # all optional components
```

### Patching

Hugging Face and checkpoint patches are now applied explicitly through
`opaque.patches`:

```python
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

# ... build / wrap the model, then patch the concrete instance
apply_model_patches(model)
```

`apply_runtime_patches()` enables the runtime-side checkpoint, collator, and
loss-mapping fixes. `apply_model_patches(model)` wires compat wrappers and
Triton kernels into the specific model instance, including PEFT/LoRA modules.
See `docs/user-guide/huggingface.md` for the model matrix and tuning knobs.

## Example

A minimal DP-SGD training loop:

```python
import torch
import opaque.accounting as acc
from opaque.clipping import clipped_grad
from opaque.random import key
from opaque.dpsgd.noise import gaussian_noise

def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()

# Calibrate noise for target privacy budget
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=10.0,
)
batch_size = 64  # expected batch size for Poisson sampling

# DP-SGD components
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=(1, 2),
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
  noise_multiplier=result.param, key=key(42),
)

# Training loop
params = torch.randn(10, requires_grad=False)
lr = 0.01
for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads  # or use torchopt optimizer
```

## Features

- **Per-example gradient clipping** via `torch.func.vmap` + `torch.func.grad`,
  with fixed, adaptive (Andrew et al. 2021), and AUTO-S (Bu et al. 2023) variants.
- **Noise injection**: Gaussian, truncated Gaussian, and correlated
  matrix-factorization noise (band-MF, BLT, BSR, BiSR, DP-λCGD), including
  private second-moment streams for adaptive optimizers.
- **Privacy accounting**: Rust-based PLD engine with tight composition,
  multiple privacy metrics (ε-δ, f-DP advantage, error rates), and noise
  calibration via binary search.
- **Sampling**: standard Poisson, truncated Poisson, cyclic Poisson,
  balls-in-bins, b-min-separation, and sequential batch samplers.
- **Privacy auditing**: empirical privacy validation via membership inference.
- **Distributed training**: DDP-compatible with synchronized noise and
  gradient aggregation via `opaque.distributed`.
- **HuggingFace compatibility**: automatic `vmap` patching for LLaMA, Mistral,
  Qwen2/3, Phi-3, Gemma/Gemma2, Granite, Cohere/Cohere2, plus fused Triton
  kernels via `opaque.patches`.

## Documentation

- [Getting Started](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Tutorials](docs/tutorials/README.md)
- [API Reference](docs/api/index.md)
- [Examples](examples/)

## Development

```bash
uv sync --group dev --all-packages --extra all
uv run pytest -m "not cuda and not mps and not slow"        # PR-equivalent suite
uv run ruff format packages/                                # Format
uv run ruff check packages/                                 # Lint
uv run --group docs mkdocs build --strict                   # Build docs
cargo test --manifest-path packages/opaque-accounting/Cargo.toml
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development workflow.

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (Abadi et al. 2016)
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) — original inspiration
- [Opacus](https://opacus.ai/) — alternative PyTorch DP library (hook-based design)

## License

Apache 2.0. See [LICENSE](./LICENSE).
