# Opaque

Functional DP-SGD and DP-FTRL for Torch, JAX, and MLX.

Opaque provides composable primitives for differentially private model
training: per-example gradient clipping, calibrated noise injection, privacy
accounting, and sampling. Its backend-neutral functional API uses explicit
state and provider-native arrays — no hooks, no subclassing, no hidden
mutation. DP-FTRL matrix-factorization noise runs eagerly with Torch, JAX,
and MLX.

> **Work in progress:** Opaque is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/pytorch-2.9+-red.svg)](https://pytorch.org/)
[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)

## Packages

Install `opaque` for the curated default bundle. Libraries can instead depend
on `opaque-engine` and the provider wheel they use. The repository is
implemented as [PEP 420] namespace packages under the shared `opaque.*`
namespace:

| Distribution | Import roots | Purpose |
|---|---|---|
| `opaque` | — | Convenience installer; pulls in a curated bundle of sub-packages |
| `opaque-base` | `opaque.serialization` | Pure-Python serialization registry + dispatcher; the seam every other wheel registers handlers against |
| `opaque-engine` | `opaque.{backend,primitive,ops,autodiff,types,pytree,random,distributed,functional,scheduling,profiling}` | Backend-neutral primitives, transforms, algorithms, pytrees, RNG keys, schedules, and optional runtime seams |
| `opaque-torch` | `opaque.torch` | PyTorch primitive/runtime implementations, serialization handlers, functionalization, and Torch RNG bridges |
| `opaque-jax` | `opaque.jax` | JAX primitive/runtime implementations and native-array serialization |
| `opaque-mlx` | `opaque.mlx` | MLX primitive/runtime implementations and native-array serialization |
| `opaque-optimizers` | `opaque.optimizers` | Backend-neutral functional optimizers with DP-aware AdamW-BC and related rules |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / truncated / per-group noise, Poisson samplers, adaptive clipping, DP-SGD-specific accounting factories |
| `opaque-dpftrl` | `opaque.dpftrl` | DP-FTRL mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, correlated-noise samplers, DP-FTRL-specific accounting factories |
| `opaque-auditing` | `opaque.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) |
| `opaque-patches` | `opaque.patches` | Unified patching entrypoint for PyTorch checkpointing, Hugging Face compat wrappers, Triton kernels, and PEFT/LoRA fusion |
| `opaque-transformers` | `opaque.transformers` | Hugging Face trainer + integration; TRL-style `SFTTrainer` / `DPOTrainer` (`opaque.transformers.trl`) built on `DPTrainer` |
| `opaque-alignment` | `opaque.alignment` | Functional, mechanism-agnostic DP-safe SFT / DPO primitives: per-example losses, log-prob helpers, collators, reference helpers, reward metrics |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (Rust/PyO3 backend); torch-free standalone |

[PEP 420]: https://peps.python.org/pep-0420/

### Import layout

```
opaque.serialization                                       <- opaque-base
opaque.{backend,primitive,ops,autodiff}                    <- opaque-engine
opaque.{types,pytree,random,distributed}                   <- opaque-engine
opaque.{functional,scheduling,profiling}                   <- opaque-engine
opaque.torch{,.random}                                     <- opaque-torch
opaque.jax                                                  <- opaque-jax
opaque.mlx                                                  <- opaque-mlx
opaque.optimizers                                          <- opaque-optimizers
opaque.dpsgd.{clipping,noise,sampling,accounting}          <- opaque-dpsgd
opaque.dpftrl.{clipping,noise,sampling,accounting}         <- opaque-dpftrl
opaque.auditing                                            <- opaque-auditing
opaque.patches.{kernels,torch,transformers,peft}           <- opaque-patches
opaque.transformers{,.trl}                                 <- opaque-transformers
opaque.alignment.{sft,dpo,data,metric}                     <- opaque-alignment
opaque.accounting                                          <- opaque-accounting
```

## Installation

```bash
# From JetBrains Packages
pip install opaque \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/

# Or with uv
uv add opaque \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Extras:

```bash
pip install "opaque[auditing]"      # empirical privacy auditing
pip install "opaque[dpftrl]"        # DP-FTRL with the default Torch provider
pip install "opaque[dpftrl,jax]"    # also install the JAX provider
pip install "opaque[dpftrl,mlx]"    # also install the MLX provider
pip install "opaque[transformers]"  # Torch-only Hugging Face integration
pip install "opaque[jax]"           # JAX backend provider
pip install "opaque[mlx]"           # MLX backend provider
pip install "opaque[all]"           # all optional components
```

The default `opaque` bundle installs `opaque-torch`. Backend-neutral users can
instead install `opaque-engine` with `opaque-torch`, `opaque-jax`, or
`opaque-mlx` as needed. The first Torch tensor or module, JAX array, or MLX
array passed to an Opaque execution API selects the matching provider; that
selection remains active until `opaque.backend.clear_backend()` is called.
Use `opaque.backend.use_backend(...)` for a temporary, context-local switch.
For a provider-neutral DP-FTRL installation without the default bundle, pair
`opaque-dpftrl` with exactly the provider wheel the application uses.

`opaque.transformers` and `opaque.patches` model integration remain Torch-only.
Provider-neutral DP-FTRL covers the eager functional clipping, noise, optimizer,
state, and serialization paths; it does not make Hugging Face models portable
to JAX or MLX.

### Patching

Hugging Face and checkpoint patches are applied explicitly through
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

See [`docs/user-guide/huggingface/model-patches.md`](docs/user-guide/huggingface/model-patches.md)
for patching details, model compatibility, and tuning knobs.

## Example

A minimal DP-SGD training loop:

```python
import torch
import opaque.accounting as acc  # cross-cutting (calibrate, budget)
import opaque.dpsgd.accounting as dpsgd_acc  # DP-SGD per-step factories
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import apply_updates, sgd
from opaque.random import key


def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()


# Calibrate noise for target privacy budget
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=10.0,
)
batch_size = 64  # expected batch size for Poisson sampling

# DP-SGD components
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=(1, 2),
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=result.param,
    key=key(42),
)

# Training loop
params = torch.randn(10, requires_grad=False)
optimizer_step, opt_state = sgd(params, lr=0.01)
for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer_step(noisy_grads, opt_state, params=params)
    params = apply_updates(params, updates)
```

## Features

- **Per-example gradient clipping** via provider-dispatched `vmap` + `grad`,
  with fixed, adaptive (Andrew et al. 2021), and AUTO-S (Bu et al. 2023) variants.
- **Noise injection**: Gaussian, truncated Gaussian, and correlated
  matrix-factorization noise (band-MF, BLT, BSR, BiSR, DP-λCGD), including
  private second-moment streams for adaptive optimizers.
- **Privacy accounting**: Rust-based PLD engine with tight composition,
  multiple privacy metrics (ε-δ, f-DP advantage, error rates), and noise
  calibration via binary search.
- **Sampling**: DP-SGD Poisson (plain or capped), DP-FTRL Poisson (identity or banded),
  balls-in-bins, b-min-separation, and sequential batch samplers.
- **Privacy auditing**: empirical privacy validation via membership inference.
- **Distributed training**: DDP-compatible with synchronized noise and
  gradient aggregation via `opaque.distributed`.
- **Torch-only Hugging Face compatibility**: automatic `vmap` patching for LLaMA, Mistral,
  Qwen2/3, Phi-3, Gemma/Gemma2, Granite, Cohere/Cohere2, plus fused Triton
  kernels via `opaque.patches`.

## Documentation

- [Getting Started](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Tutorials](docs/tutorials/README.md)
- [API Reference](docs/reference/index.md)
- [Examples](examples/)

## Development

```bash
uv sync --group dev --all-packages --extra all
uv run pytest -m "not cuda and not mps and not slow"        # PR-equivalent suite
uv run ruff format packages/                                # Format
uv run ruff check packages/                                 # Lint
uv run --group docs mkdocs build --strict                   # Build docs
cargo test --workspace
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development workflow.
The [development guide](docs/development/index.md) includes fork-based setup.

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (Abadi et al. 2016)
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) — original inspiration
- [Opacus](https://opacus.ai/) — alternative PyTorch DP library (hook-based design)

## License

Apache 2.0. See [LICENSE](./LICENSE).
