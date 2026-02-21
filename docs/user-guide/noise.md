# Noise Addition

After [clipping gradients](clipping.md), the next step in DP-SGD is **adding calibrated Gaussian noise**.
The noise obscures individual contributions, providing the differential privacy guarantee.

## `gaussian_noise()`

All noise functions return `(noise_fn, state)`:

```python
from opaque import gaussian_noise
from opaque.random import key

noise_fn, state = gaussian_noise(stddev=noise_multiplier * clip_norm, key=key(42))

noisy_grads, state = noise_fn(clipped_grads, state)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stddev` | `float` | Standard deviation of Gaussian noise. Typically `noise_multiplier * clip_norm`. |
| `key` | `RngKey` | Explicit RNG key for deterministic noise. Create with `key(seed)`. |

## Calibrating Noise

Use the accounting module to find the minimum noise for your target privacy level.

### (ε, δ)-DP

```python
import opaque.accounting as acc

sample_rate = batch_size / dataset_size
num_steps = num_epochs * (dataset_size // batch_size)

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=10.0,
)
noise_multiplier = result.param
```

### f-DP Advantage

```python
result = acc.calibrate(
    acc.advantage_budget(0.1),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=10.0,
)
```

### (α, β) Error Rates

```python
result = acc.calibrate(
    acc.beta_budget(0.8, alpha=1e-4),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=10.0,
)
```

## Typical Noise Multipliers

| Privacy (ε, δ) | Sample Rate | Steps | Noise Multiplier |
|----------------|-------------|-------|------------------|
| (1.0, 1e-5)   | 0.01        | 1000  | ~3.5             |
| (3.0, 1e-5)   | 0.01        | 1000  | ~1.2             |
| (10.0, 1e-5)  | 0.01        | 1000  | ~0.4             |
| (3.0, 1e-5)   | 0.1         | 1000  | ~0.5             |

!!! tip "Higher sample rate = less noise"
    Larger batches provide privacy amplification, requiring less noise for the same ε.

## Complete DP-SGD Loop

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

clip_norm = 1.0
batch_size, dataset_size = 32, 10_000
sample_rate = batch_size / dataset_size
num_steps = 1000

# Calibrate
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    0.1, 10.0,
)
noise_multiplier = result.param

# DP components
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=clip_norm, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_norm, key=key(42),
)

# Train
for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads

# Verify privacy
training = acc.poisson(acc.gaussian(noise_multiplier), sample_rate) * num_steps
print(f"ε = {training.epsilon_at(1e-5):.2f}")
```

## Bounded Gaussian Noise

For applications where outputs must stay in a fixed range
([Chen & Hale 2024](https://arxiv.org/abs/2211.17230)):

```python
from opaque import bounded_gaussian_noise
from opaque.random import key

noise_fn, state = bounded_gaussian_noise(
    stddev=1.0, bounds=(-3.0, 3.0), key=key(42),
)
noisy_grads, state = noise_fn(clipped_grads, state)
```

## Noise Schedule (Advanced)

You can vary noise across steps.  Track per-step composition with `Accountant`:

```python
from opaque.accounting.accountant import Accountant

acct = Accountant()

for step in range(num_steps):
    current_noise = 2.0 - 1.5 * (step / num_steps)  # decreasing
    noise_fn, noise_state = gaussian_noise(stddev=current_noise * clip_norm, key=key(step))
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    # ...
    acct = acct | acc.poisson(acc.gaussian(current_noise), sample_rate)

print(f"ε = {acct.epsilon_at(1e-5):.2f}")
```

!!! warning "Use with caution"
    Noise schedules can improve accuracy but complicate privacy analysis.
    Stick with fixed noise unless you understand composition.

## PyTree Support

`gaussian_noise()` works with any PyTree structure (nested dicts of tensors):

```python
grads = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}
noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

## Reproducibility

Same key → same noise sequence:

```python
from opaque.random import key

noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
noisy1, state = noise_fn(grads, state)

noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
noisy2, state = noise_fn(grads, state)
# noisy1 == noisy2 (deterministic)
```

In distributed training, pass the same `key(seed)` on all ranks to produce
identical noise.  See [Distributed Training](distributed.md) and
[RNG Key](rng-key.md) for details.

## See Also

- [Gradient Clipping](clipping.md) — step before noise
- [Privacy Accounting](accounting.md) — track budget after noise
- [Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb) — interactive tutorial
- [API Reference](../api/noise.md) — full API docs
