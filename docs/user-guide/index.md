# User Guide

This guide explains each component of Opaque's DP-SGD pipeline: what it does,
how the API works, and the practical decisions you need to make. For hands-on
practice, see the [Tutorials](../tutorials/README.md). For complete function
signatures, see the [API Reference](../api/index.md).

## End-to-end DP-SGD training

A complete DP-SGD training loop uses five components: calibration, clipping,
noise, sampling, and accounting. Here is a minimal working example that ties
them together:

```python
import torch
import torchopt
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.sampling import PoissonSampler
from opaque.random import key, split

# --- Privacy parameters ---
dataset_size = 50_000
batch_size = 256
sample_rate = batch_size / dataset_size
num_steps = 1000

# --- Calibrate noise multiplier ---
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param

# --- Create DP components ---
key_sampling, key_noise = split(key(42), num=2)

grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, argnums=0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity, key=key_noise,
)

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

sampler = PoissonSampler(dataset, sample_rate=sample_rate, key=key_sampling)
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# --- Training loop ---
from opaque.accounting.accountant import Accountant

step_proc = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

    acct = acct | step_proc
    if acct.budget_exceeded:
        break

print(f"Final privacy: epsilon={acct.epsilon_at(1e-5):.2f}")
```

The sections below explain each part. Read them in order for a complete
understanding, or jump to a specific topic.

## Topics

### Foundations

- **[Differential Privacy Concepts](dp-concepts.md)** -- What DP guarantees,
  how DP-SGD works, privacy budgets, composition, and amplification.
- **[Random Number Generation](rng-key.md)** -- Explicit RNG keys, splitting,
  fold_in, and reproducibility in distributed training.

### Core pipeline

- **[Per-Example Gradient Clipping](clipping.md)** -- `clipped_grad`,
  `adaptive_clipped_grad`, `auto_clipped_grad`, microbatching, and
  per-group clipping.
- **[Noise Addition](noise.md)** -- Gaussian noise, bounded Gaussian variants
  (truncated, rectified), and matrix-factorization correlated noise for DP-FTRL.
- **[Privacy Accounting](accounting.md)** -- Composable `DpProcess` objects,
  privacy metrics, calibration, and the `Accountant` helper.
- **[Sampling & Microbatching](sampling.md)** -- Poisson, truncated Poisson,
  and cyclic samplers with distributed support.

### Integration

- **[Optimizers](optimizers.md)** -- Using TorchOpt functional optimizers with
  DP-SGD.
- **[Distributed Training](distributed.md)** -- DDP with synchronized noise
  and gradient aggregation.
- **[HuggingFace Compatibility](huggingface.md)** -- Using HuggingFace
  Transformers models with Opaque, including LoRA, fused Triton kernels,
  and model compatibility.
- **[Memory Optimizations](memory-optimizations.md)** -- Microbatching,
  gradient checkpointing, fused kernels, profiling, and configuration.
- **[Privacy Auditing](auditing.md)** -- Empirical privacy validation via
  membership inference.

### Reference

- **[Known Limitations](../limitations.md)** -- Flash Attention, DDP-only,
  in-place operations, and other constraints.
