# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

## Features

- **Functional API**: Composable DP primitives inspired by JAX-Privacy
- **LoRA-First**: Optimized for parameter-efficient fine-tuning with lower gradient norms
- **PyTorch Native**: Built on `torch.func` functional transformations
- **Privacy Accounting**: Rust PLD engine with compositional `DpProcess` API
- **Calibration**: Binary search for noise multiplier against any privacy metric
- **Distributed Training**: DDP-compatible with Poisson subsampling
- **Privacy Auditing**: Empirical validation of privacy guarantees
- **Test-Driven**: Validated against JAX-Privacy reference implementation

## Quick Example

```python
import torch
import torchopt
import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque import clipped_grad, gaussian_noise
from opaque.sampling import PoissonSampler

# 1. Calibrate noise for target privacy
dataset_size = 50_000
sample_rate = 256 / dataset_size
num_steps = 1000

result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param

# 2. Create clipped gradient function and noise
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier)

# 3. Set up optimizer and sampling
optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)
sampler = PoissonSampler(dataset_size, sample_rate=sample_rate)
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# 4. Training loop with per-step accounting
from opaque.accounting.accountant import Accountant

step_proc = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
acct = Accountant(budget=cal.epsilon_budget(3.0, delta=1e-5))

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

    acct = acct | step_proc
    if acct.budget_exceeded:
        break

eps = acct.epsilon_at(1e-5)
```

## Installation

Opaque is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync
```

## Next Steps

### For Learners

- [Tutorial 01: Gradient Clipping from Basics](tutorials/01_gradient_clipping_from_basics.ipynb)
- [Tutorial 02: Differential Privacy - Noise and Accounting](tutorials/02_differential_privacy_noise_and_accounting.ipynb)
- [Quick Start Guide](getting-started/quickstart.md)
- [DP Basics](user-guide/dp-basics.md)

### For Developers
- [API Reference](api/core/clipping.md)
- [Contributing Guide](development/contributing.md)
