# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

## Features

- **Functional API**: Composable DP primitives inspired by JAX-Privacy
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` functional transformations
- **Test-Driven**: Validated against JAX-Privacy reference implementation

## Status

!!! success "Production-Ready Core"
    🎉 DP-SGD is ready! All core components implemented:

    - ✅ **Gradient clipping** with `clipped_grad()`
    - ✅ **Noise injection** with `gaussian_noise()`
    - ✅ **Privacy accounting** via Rust PLD engine (`DpProcess` composition)
    - ✅ **Adaptive clipping** with `adaptive_clipped_grad()`
    - ✅ **Privacy auditing** for empirical validation
    - ✅ **Numerical equivalence** with JAX-Privacy confirmed

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# 1. Calibrate noise for target privacy
sample_rate = 0.01  # batch_size / dataset_size
num_steps = 1000

def build(nm):
    return acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * num_steps

result = acc.calibrate(acc.epsilon(3.0, delta=1e-5), build, 0.1, 10.0)
noise_multiplier = result.param

# 2. Create clipped gradient function
clip_norm = 1.0
grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,
    argnums=0,
    batch_argnums=1,
)

# 3. Create noise function and training loop
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_norm)

for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)

# 4. Get final privacy guarantee
training = acc.poisson(acc.gaussian(noise_multiplier), sample_rate=sample_rate) * num_steps
epsilon = training.epsilon_at(1e-5)
print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")
```

## Installation

!!! note
    Opaque is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/JetBrains-Research/opaque.git
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
