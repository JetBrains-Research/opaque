# Opaque

**Functional Differential Privacy for PyTorch**

Opaque provides per-example gradient clipping, noise injection, and privacy
accounting for DP-SGD training.  It uses a functional API built on
`torch.func` — no hooks, no subclassing, no hidden state.

## What Opaque Provides

| Module | Purpose |
|--------|---------|
| `opaque.clipping` | Per-example gradient clipping (`clipped_grad`, `adaptive_clipped_grad`) |
| `opaque.noise` | Gaussian noise, bounded Gaussian, matrix factorization noise |
| `opaque.accounting` | Rust PLD privacy accounting — composition, calibration, budgets |
| `opaque.sampling` | Poisson, truncated Poisson, and cyclic Poisson samplers |
| `opaque.auditing` | Empirical privacy auditing via membership inference |
| `opaque.distributed` | DDP utilities (`sum_gradients`, state sync) |
| `opaque.compat` | HuggingFace auto-patching for `vmap` compatibility |

## Quick Example

```python
import torch
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

# Loss function (works with any model)
def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()

# DP-SGD components
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))

# Training loop
for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

## How It Works

1. `clipped_grad` uses `torch.func.vmap` + `torch.func.grad` to compute
   per-example gradients, clips each to L2 norm ≤ `l2_clip_norm`, and sums.
2. `gaussian_noise` adds $\mathcal{N}(0, \sigma^2 I)$ noise where
   $\sigma = \texttt{stddev}$.
3. Privacy accounting tracks cumulative privacy loss via PLD composition.

## Installation

```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync
```

## Next Steps

**Learning from scratch:**

- [Tutorial 01: Gradient Clipping from Basics](tutorials/01_gradient_clipping_from_basics.ipynb)
- [Tutorial 02: Noise and Accounting](tutorials/02_differential_privacy_noise_and_accounting.ipynb)
- [Quick Start](getting-started/quickstart.md)

**I know DP, show me the API:**

- [API Reference](api/index.md)
- [User Guide](user-guide/index.md)
- [LoRA Fine-tuning](user-guide/lora.md)
- [Distributed Training](user-guide/distributed.md)
