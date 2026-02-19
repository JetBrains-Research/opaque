# User Guide

Opaque provides a functional API for training PyTorch models with differential
privacy (DP).

## End-to-End DP Training

A complete DP-SGD training loop has five parts: calibration, clipping, noise,
sampling, and accounting. Here is a minimal working example:

```python
import torch
import torchopt
import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque import clipped_grad, gaussian_noise
from opaque.sampling import PoissonSampler

# --- Calibrate noise multiplier for target privacy ---
dataset_size = 50_000
sample_rate = 256 / dataset_size
num_steps = 1000

result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1,
    param_max=5.0,
)
noise_multiplier = result.param

# --- Set up gradient clipping, noise, optimizer ---
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier)

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# --- Set up privacy-amplifying sampling ---
sampler = PoissonSampler(dataset_size, sample_rate=sample_rate)
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# --- Training loop with per-step accounting ---
step_proc = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
acct = acc.Accountant(budget=cal.epsilon_budget(3.0, delta=1e-5))

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

The sections below explain each part in detail.

## Core Concepts

### Per-Sample Gradient Clipping

DP-SGD computes *per-example* gradients and clips each to a maximum L2 norm
before summing. This bounds the sensitivity of the gradient query.

```python
from opaque import clipped_grad

grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    argnums=0,        # differentiate w.r.t. first arg (params)
    batch_argnums=1,  # second arg is the batched input
)

grads, clip_state = grad_fn(params, batch, state=clip_state)
```

See [Gradient Clipping](clipping.md) for details on `clipped_grad`,
`clipped_fun`, and `clip_pytree`.

### Noise Addition

After clipping and summing, Gaussian noise scaled to the sensitivity is added:

```python
from opaque import gaussian_noise

noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

The noise standard deviation is `noise_multiplier * l2_clip_norm` in absolute
terms. When `l2_clip_norm=1.0`, the noise multiplier is used directly as
`stddev`.

See [Noise](noise.md) for bounded Gaussian noise and matrix-factorization
correlated noise (DP-FTRL).

### Calibration

Use binary search to find the noise multiplier achieving a target privacy
budget:

```python
from opaque.accounting import calibration as cal

result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1,
    param_max=5.0,
)
noise_multiplier = result.param
```

`calibrate()` works with any float parameter, not just noise multiplier. Pass a
different `build` lambda to calibrate sample rate, number of steps, or any
other quantity.

See [Privacy Accounting](accounting.md) for all target types and calibration
options.

### Privacy Accounting

Build composable privacy processes and query metrics:

```python
import opaque.accounting as acc

step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
training = step * num_steps
eps = training.epsilon_at(1e-5)
adv = training.advantage()
```

Use `Accountant` to track privacy spend per step during training:

```python
acct = acc.Accountant(budget=cal.epsilon_budget(3.0, delta=1e-5))
for batch in dataloader:
    # ... train ...
    acct = acct | step
    if acct.budget_exceeded:
        break
```

See [Privacy Accounting](accounting.md) and the
[API reference](../api/accounting.md).

### Poisson Sampling

Poisson subsampling amplifies privacy: each example is included independently
with probability `sample_rate = expected_batch_size / dataset_size`.

```python
from opaque.sampling import PoissonSampler

sampler = PoissonSampler(dataset_size, sample_rate=sample_rate)
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
```

See [Sampling](sampling.md) for `TruncatedPoissonSampler`, `CyclicPoissonSampler`,
distributed modes, and microbatching.

### TorchOpt Optimizers

Opaque does not bundle optimizers. Use any TorchOpt functional optimizer:

```python
import torchopt

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

`torchopt.sgd`, `torchopt.adam`, and `torchopt.adamw` all work.

### Adaptive Clipping

`adaptive_clipped_grad()` auto-tunes the clip norm using the geometric
adaptation rule from Andrew et al. 2021. This replaces manual clip norm tuning
with a target quantile:

```python
from opaque.clipping import adaptive_clipped_grad

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.5,
    batch_argnums=1,
)
```

See [Adaptive Clipping](optimizers.md) for parameters and privacy accounting
with `acc.adaclip()`.

## LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is particularly effective for DP training because
LoRA adapters have far fewer parameters, which means per-example gradient norms
are naturally smaller. Lower gradient norms mean less clipping distortion and
less noise needed to achieve the same privacy guarantee.

```python
from peft import get_peft_model, LoraConfig
from opaque.clipping import adaptive_clipped_grad

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.5,
    batch_argnums=1,
)
```

See [LoRA Guide](lora.md) and
[Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb).

## Privacy Metrics

Opaque supports three families of privacy metrics on any `DpProcess`:

| Method              | Metric                        | Reference              |
|---------------------|-------------------------------|------------------------|
| `epsilon_at(delta)` | (epsilon, delta)-DP            | Dwork et al. 2006      |
| `advantage()`       | f-DP total-variation advantage | Dong et al. 2019       |
| `beta_at(alpha)`    | (alpha, beta) error rates      | Wasserman & Zhou 2010  |

```python
training = acc.poisson(acc.gaussian(noise_multiplier), sample_rate) * num_steps
eps = training.epsilon_at(1e-5)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
```

## Troubleshooting

**Model does not train (accuracy at chance)**: The noise multiplier may be too
large for the number of training steps, or `l2_clip_norm` is too small.
Increase epsilon or increase the number of steps.

**Privacy budget exceeded early**: Use `TruncatedPoissonSampler` for tighter
bounds or increase the target epsilon.

**Out of memory**: Use LoRA to reduce parameter count, or use microbatching
(see [Sampling](sampling.md)).

## Next Steps

- [API Reference](../api/index.md) -- Detailed function documentation
- [Tutorials](../tutorials/README.md) -- Interactive Jupyter notebooks
- [Contributing](../development/contributing.md) -- Contribute to Opaque
