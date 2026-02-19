# Quick Start

This guide will walk you through training your first differentially private model with Opaque in 5 minutes.

## Prerequisites

Make sure you have Opaque installed. If not, see the [Installation Guide](installation.md).

```bash
# From source (recommended for now)
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync
```

## Your First DP-SGD Model

Let's train a simple linear regression model with differential privacy. This example demonstrates Opaque's core
training components:

1. **Gradient clipping** for bounded sensitivity
2. **Noise injection** for privacy
3. **Privacy accounting** to track the privacy budget

### Complete Example

```python
import torch
import torch.nn as nn
import opaque.accounting as acc
from opaque import make_functional, clipped_grad, gaussian_noise

# Generate synthetic data
torch.manual_seed(42)
n_samples, n_features = 1000, 10
X = torch.randn(n_samples, n_features)
y = X @ torch.randn(n_features) + torch.randn(n_samples)

# 1. Create model and convert to functional form
model = nn.Linear(n_features, 1)
fmodel, params = make_functional(model)

# 2. Define per-example loss function
def loss_fn(params, example):
    """Compute loss for a single example."""
    x, y_true = example
    y_pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (y_pred - y_true) ** 2

# 3. Set privacy parameters
epsilon = 3.0  # Privacy budget
delta = 1e-5  # Failure probability
batch_size = 32
sample_rate = batch_size / n_samples
num_epochs = 10
num_steps = num_epochs * (n_samples // batch_size)

# 4. Calibrate noise multiplier for target privacy
result = acc.calibrate(
    budget=acc.epsilon_budget(epsilon, delta=delta),
    process=lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1,
    param_max=100.0,
)
noise_multiplier = result.param
print(f"Using noise multiplier: {noise_multiplier:.3f}")

# 5. Create clipped gradient function (returns fn + state)
clip_norm = 1.0
grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,  # Differentiate w.r.t. first argument (params)
    batch_argnums=1,  # Second argument (example) is batched
    l2_clip_norm=clip_norm,
)

# 6. Training loop with noise and per-step accounting
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity()
)
learning_rate = 0.01

accountant = acc.Accountant(budget=acc.epsilon_budget(epsilon, delta=delta))
step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)

for epoch in range(num_epochs):
    # Shuffle data
    perm = torch.randperm(n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    for i in range(0, n_samples, batch_size):
        # Get batch
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # Compute clipped gradients (per-example clipping + sum)
        grads, clip_state = grad_fn(params, (X_batch, y_batch), state=clip_state)

        # Add calibrated Gaussian noise
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Update parameters
        params = tuple(p - learning_rate * g for p, g in zip(params, noisy_grads))

        # Track privacy
        accountant = accountant | step

    eps = accountant.epsilon_at(delta)
    print(f"Epoch {epoch+1}/{num_epochs} - Privacy: ε={eps:.2f}")

print(f"\nTraining complete!")
print(f"Final privacy: (ε={accountant.epsilon_at(delta):.2f}, δ={delta})")
```

### Expected Output

```
Using noise multiplier: 1.234
Epoch 1/10 - Privacy: ε=0.32
Epoch 2/10 - Privacy: ε=0.64
...
Epoch 10/10 - Privacy: ε=3.00

Training complete!
Final privacy: (ε=3.00, δ=1e-05)
```

## Understanding the Code

### 1. Functional Model Conversion

```python
fmodel, params = make_functional(model)
```

Opaque uses PyTorch's functional API (`torch.func`). We convert the model to a functional form where parameters are
passed explicitly rather than stored in the module.

### 2. Per-Example Loss

```python
def loss_fn(params, example):
    x, y_true = example
    y_pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (y_pred - y_true) ** 2
```

DP-SGD requires computing **per-example gradients**, not batch-average gradients. The loss function takes a single
example.

### 3. Calibration

```python
result = acc.calibrate(
    budget=acc.epsilon_budget(3.0, delta=1e-5),
    process=lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1,
    param_max=100.0,
)
noise_multiplier = result.param
```

Calibration binary-searches for the noise multiplier that achieves a target privacy guarantee. It composes a full training run as a `DpProcess` and queries its epsilon. The `process` lambda describes how the training run depends on the noise multiplier.

### 4. Clipped Gradients

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=1,
    l2_clip_norm=clip_norm,
)
```

`clipped_grad()` returns a tuple of `(grad_fn, clip_state)`:

- `grad_fn` computes per-example gradients using `torch.func.vmap`, clips each to max L2 norm, and sums them
- `clip_state` tracks the clipping parameters and provides `sensitivity()` for noise calibration

Call it with explicit state: `grads, clip_state = grad_fn(params, batch, state=clip_state)`

### 5. Privacy Accounting

```python
# Compose a single step
step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)

# Track during training with Accountant
accountant = acc.Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
for i in range(num_steps):
    accountant = accountant | step
    eps = accountant.epsilon_at(delta)
```

Opaque's accounting uses **composable `DpProcess` objects**:

- **Mechanisms**: `acc.gaussian()`, `acc.poisson()`, `acc.truncated_poisson()`, etc.
- **Composition**: `*` repeats a process, `|` composes different processes
- **Metrics**: `.epsilon_at()`, `.delta_at()`, `.advantage()`, `.beta_at()`, `.risk_at()`

The `Accountant` class provides step-by-step tracking with optional budget checking:

```python
accountant = acc.Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)

for i in range(num_steps):
    # ... train ...
    accountant = accountant | step
    if accountant.budget_exceeded:
        print("Privacy budget exhausted!")
        break
```

## What's Next?

### Deep Dives

- **[Tutorial 01](../tutorials/01_gradient_clipping_from_basics.ipynb)**: Learn gradient clipping in depth
- **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)**: Understand noise injection and
  privacy accounting
- **[Tutorial 03](../tutorials/03_complete_dp_sgd_training.ipynb)**: Complete DP-SGD training workflow

### Conceptual Guides

- **[DP Basics](../user-guide/dp-basics.md)**: What is differential privacy?
- **[Clipping](../user-guide/clipping.md)**: Per-sample gradient clipping explained
- **[Accounting](../user-guide/accounting.md)**: Privacy budget tracking

### Real-World Use Cases

- **[Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb)**: Fine-tune LLMs with LoRA and DP
- **[LoRA Guide](../user-guide/lora.md)**: Why LoRA + DP is a great combination

## Common Gotchas

### Privacy Budget

!!! warning "Don't exceed your privacy budget!"
Once you've spent your privacy budget (ε, δ), you cannot train more without weakening privacy guarantees. Plan your
`num_steps` carefully!

### Learning Rate

!!! tip "DP training needs higher learning rates"
Noise addition reduces effective gradient magnitude. Try learning rates 2-5x higher than non-private training.

### Clipping Norm

!!! info "Clip norm affects privacy-utility tradeoff"
- **Higher clip norm**: Less clipping, more noise needed, weaker privacy
- **Lower clip norm**: More clipping, less noise needed, stronger privacy

    Start with `clip_norm=1.0` and tune based on your data.

## Getting Help

- **[API Reference](../api/index.md)**: Detailed function documentation
- **[GitHub Issues](https://github.com/JetBrains-Research/opaque/issues)**: Report bugs or request features
- **[GitHub Discussions](https://github.com/JetBrains-Research/opaque/discussions)**: Ask questions

---

**Next**: Explore the [Tutorials](../tutorials/README.md) for interactive learning!
