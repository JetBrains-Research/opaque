# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

## Features

- **Functional API**: Composable DP primitives inspired by JAX-Privacy
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` functional transformations
- **Test-Driven**: Validated against JAX-Privacy reference implementation

## Status

!!! success "Stage 1 & 2 Complete!"
🎉 DP-SGD is ready! All core components implemented:

    - ✅ **Stage 1**: Gradient clipping with `clipped_grad()`
    - ✅ **Stage 2**: Noise injection and functional privacy accounting
    - ✅ **111 tests passing** (55 accounting + 56 optimizer tests)
    - ✅ **Numerical equivalence** with JAX-Privacy confirmed
    - ✅ **Functional API**: Immutable state, composable privacy accounting

    🔜 Next: Stage 3 (Additional Optimizer Features)

## Quick Example: Complete DP-SGD

```python
import torch
import opaque.accounting as acc
from opaque import (
    clipped_grad,
    add_gaussian_noise,
)

# 1. Calibrate noise for target privacy
sample_rate = 0.01  # batch_size / dataset_size
num_steps = 1000

noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# 2. Create clipped gradient function
clip_norm = 1.0
clipped_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,
    argnums=0,
    batch_argnums=1,
)

# 3. Training loop with privacy accounting
privacy_state = acc.create()

for step in range(num_steps):
    grads = clipped_grad_fn(params, batch)
    noisy_grads = add_gaussian_noise(grads, stddev=noise_multiplier * clip_norm)
    params = update(params, noisy_grads)

    # Compose privacy
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        count=1,
    )

# 4. Get final privacy guarantee
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")
```

## Installation

!!! note
    Opaque is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/evgri243/opaque.git
cd opaque
uv sync
```

## Next Steps

### For Learners

- 📚 [Tutorial 01: Gradient Clipping from Basics](tutorials/01_gradient_clipping_from_basics.ipynb) - Learn gradient
  clipping
-
📚 [Tutorial 02: Differential Privacy - Noise and Accounting](tutorials/02_differential_privacy_noise_and_accounting.ipynb) -
Complete DP-SGD
- 📖 [Quick Start Guide](getting-started/quickstart.md)
- 📖 [DP Basics](user-guide/dp-basics.md)

### For Developers
- 🔧 [API Reference](api/core/clipping.md)
- 🔧 [Roadmap](development/roadmap.md) - Project timeline and stages
- 🔧 [Contributing Guide](development/contributing.md)
