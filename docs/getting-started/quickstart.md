# Quick Start

Train a linear regression model with differential privacy using Opaque.

## Prerequisites

Install Opaque following the [Installation Guide](installation.md).

## Complete example

```python
import torch
import torch.nn as nn
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import Accountant
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.torch.functional import make_functional
from opaque.random import key

# Synthetic data
torch.manual_seed(42)
n_samples, n_features = 1000, 10
X = torch.randn(n_samples, n_features)
y = X @ torch.randn(n_features) + torch.randn(n_samples)

# Convert model to functional form
model = nn.Linear(n_features, 1)
fmodel, params = make_functional(model)

# Per-example loss (takes a single example, returns a scalar)
def loss_fn(params, example):
    x, y_true = example
    y_pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (y_pred - y_true) ** 2

# Privacy parameters
epsilon, delta = 3.0, 1e-5
batch_size = 32
sample_rate = batch_size / n_samples
num_steps = 10 * (n_samples // batch_size)

# Calibrate noise multiplier for target epsilon
result = acc.calibrate(
    budget=acc.epsilon_budget(epsilon, delta=delta),
    process=lambda nm: dpsgd_acc.poisson(
        dpsgd_acc.gaussian(nm), sample_rate
    ) * num_steps,
    param_min=0.1,
    param_max=100.0,
)
noise_multiplier = result.param

# Create DP components
grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=1,
    clipping_norm=1.0,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key(42),
)

# Privacy tracker
step_proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), sample_rate)
accountant = Accountant(budget=acc.epsilon_budget(epsilon, delta=delta))

# Training loop
from opaque.optimizers import apply_updates, sgd

optimizer_step, opt_state = sgd(params, lr=0.01)

for epoch in range(10):
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        batch = (X[perm[i:i+batch_size]], y[perm[i:i+batch_size]])

        grads, clip_state = grad_fn(params, batch, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer_step(noisy_grads, opt_state, params=params)
        params = apply_updates(params, updates)

        accountant = accountant | step_proc

    print(f"Epoch {epoch+1}/10 - epsilon={accountant.epsilon_at(delta):.2f}")
```

## What this does

1. **`make_functional`** converts the model so parameters are passed
   explicitly, which is required for `torch.func.vmap`.
2. **`clipped_grad`** computes per-example gradients, clips each to an L2
   norm 1.0, and sums the result.
3. **`acc.calibrate`** performs a binary search for the noise multiplier
   that achieves ε = 3.0 over the full training run.
4. **`gaussian_noise`** adds calibrated Gaussian noise to the clipped
   gradient sum.
5. **`Accountant`** tracks cumulative privacy cost and checks against the
   budget.

## Next steps

- [User Guide](../user-guide/index.md) — detailed explanations of each
  component.
- [Tutorials](../tutorials/README.md) — hands-on Jupyter notebooks.
- [API Reference](../reference/index.md) — complete function signatures.
