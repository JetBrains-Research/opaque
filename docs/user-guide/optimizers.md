# Optimizers & Adaptive Clipping

Opaque integrates with **TorchOpt** functional optimizers and provides **adaptive clipping** to automatically tune the
clip norm during training, improving the privacy-utility tradeoff.

## Why Adaptive Clipping?

In standard DP-SGD, you manually set a **fixed clip norm** (e.g., C=1.0). But the "right" clip norm depends on your data
and changes during training:

- **Early training**: Gradients are large → need higher clip norm
- **Late training**: Gradients are small → can use lower clip norm

**Adaptive clipping solves this** by automatically adjusting the clip norm based on gradient statistics (e.g., median
gradient norm).

### Benefits

✅ **Better accuracy**: Automatically finds optimal clip norm
✅ **Less tuning**: No manual hyperparameter search
✅ **Stable training**: Adapts to changing gradient magnitudes
✅ **Same privacy**: No weakening of privacy guarantees

## The `adaptive_clipping()` Wrapper

Opaque provides a **wrapper** that adds adaptive clipping to any TorchOpt optimizer:

```python
from opaque.optimizers import adaptive_clipping
import torchopt

# 1. Choose base optimizer (any TorchOpt optimizer!)
base_optimizer = torchopt.sgd(lr=0.01)

# 2. Wrap with adaptive clipping
optimizer = adaptive_clipping(
    base_optimizer,
    initial_clip_norm=1.0,  # Starting clip norm
    target_quantile=0.5,  # Adapt to median gradient norm
)

# 3. Use like any optimizer
for step in range(num_steps):
    # Get gradients (clipped with current adaptive norm)
    grads, current_clip_norm = optimizer.compute_clipped_grads(
        params, loss_fn, batch
    )

    # Add noise
    noisy_grads = gaussian_noise(grads, stddev=noise_mult * current_clip_norm)

    # Update parameters
    params = optimizer.step(params, noisy_grads)

    # Optimizer automatically updates clip norm for next step!
```

### Key Parameters

**`initial_clip_norm`**: Starting clip norm (default: 1.0)

- Use same value you'd use for fixed clipping

**`target_quantile`**: Gradient norm quantile to track (default: 0.5)

- 0.5 = median (recommended)
- 0.75 = 75th percentile (more aggressive clipping)
- 0.25 = 25th percentile (less aggressive clipping)

**`ema_decay`**: Exponential moving average decay (default: 0.9)

- Controls how fast clip norm adapts
- Higher = slower adaptation, more stable
- Lower = faster adaptation, more reactive

**`lr_scale_factor`**: Learning rate scaling (default: 1.0)

- Optionally scale LR when gradients are heavily clipped
- See "Learning Rate Scaling" below

## Complete Example with TorchOpt

Here's a full training loop using adaptive clipping:

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.optimizers import adaptive_clipping
import torchopt

# Setup
clip_norm_initial = 1.0
batch_size = 32
dataset_size = 10000
sample_rate = batch_size / dataset_size
num_steps = 1000

# Calibrate noise (use initial clip norm)
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# Create adaptive clipping optimizer
base_opt = torchopt.adam(lr=0.001, betas=(0.9, 0.999))
optimizer = adaptive_clipping(
    base_opt,
    initial_clip_norm=clip_norm_initial,
    target_quantile=0.5,
)

# Define per-example loss
def loss_fn(params, example):
    x, y = example
    pred = model(params, x)
    return (pred - y) ** 2

# Training loop
privacy_state = acc.create()

for step in range(num_steps):
    # Compute clipped gradients with adaptive norm
    grads, current_clip_norm = optimizer.compute_clipped_grads(
        params, loss_fn, batch
    )

    # Add calibrated noise (using current clip norm)
    noisy_grads = gaussian_noise(
        grads,
        stddev=noise_multiplier * current_clip_norm,
    )

    # Update parameters
    params = optimizer.step(params, noisy_grads)

    # Track privacy
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        count=1,
    )

    if step % 100 == 0:
        eps = acc.get_epsilon(privacy_state, delta=1e-5)
        print(f"Step {step}: ε={eps:.2f}, clip_norm={current_clip_norm:.3f}")
```

## Supported TorchOpt Optimizers

Opaque's `adaptive_clipping()` works with **any** TorchOpt optimizer:

### SGD

```python
base_opt = torchopt.sgd(lr=0.01, momentum=0.9)
optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)
```

### Adam

```python
base_opt = torchopt.adam(lr=0.001, betas=(0.9, 0.999))
optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)
```

### AdamW

```python
base_opt = torchopt.adamw(lr=0.001, weight_decay=0.01)
optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)
```

### RMSprop

```python
base_opt = torchopt.rmsprop(lr=0.001, alpha=0.99)
optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)
```

!!! tip "Try different optimizers"
Adam often works better than SGD for DP training. Experiment to find what works for your task!

## How Adaptive Clipping Works

Adaptive clipping maintains a **clip buffer** that tracks gradient norm statistics:

1. **Compute gradient norms** for current batch
2. **Update statistics** with exponential moving average
3. **Adjust clip norm** to target quantile
4. **Clip gradients** with updated norm

### Example Evolution

```
Step 0: clip_norm=1.000 (initial)
Step 100: clip_norm=1.234 (gradients larger than expected)
Step 500: clip_norm=0.876 (gradients getting smaller)
Step 1000: clip_norm=0.543 (converged, small gradients)
```

The clip norm automatically **increases** when gradients are large and **decreases** as training progresses.

## Learning Rate Scaling (Optional)

When gradients are heavily clipped, the effective learning rate is reduced. You can **compensate** by scaling the LR:

```python
optimizer = adaptive_clipping(
    base_opt,
    initial_clip_norm=1.0,
    lr_scale_factor=2.0,  # Scale LR when clipping is high
)
```

**How it works**:

- If 80% of gradients are clipped → scale LR by `lr_scale_factor`
- If 20% of gradients are clipped → use original LR
- Smooth interpolation in between

!!! warning "Use with caution"
LR scaling can help but may make training unstable. Start with `lr_scale_factor=1.0` (disabled).

## Comparison: Fixed vs Adaptive Clipping

| Feature        | Fixed Clipping         | Adaptive Clipping             |
|----------------|------------------------|-------------------------------|
| **Clip norm**  | Constant (e.g., C=1.0) | Dynamic (adapts to gradients) |
| **Tuning**     | Requires manual search | Automatic                     |
| **Accuracy**   | Good if C chosen well  | Often better                  |
| **Privacy**    | Same for equal noise   | Same for equal noise          |
| **Complexity** | Simple                 | Slightly more complex         |

**Recommendation**: Use adaptive clipping for best results, fall back to fixed if issues arise.

## Monitoring Adaptive Clipping

Track how the clip norm evolves:

```python
clip_norms = []

for step in range(num_steps):
    grads, current_clip_norm = optimizer.compute_clipped_grads(params, loss_fn, batch)
    clip_norms.append(current_clip_norm)

    noisy_grads = gaussian_noise(grads, stddev=noise_mult * current_clip_norm)
    params = optimizer.step(params, noisy_grads)

# Plot clip norm evolution
import matplotlib.pyplot as plt
plt.plot(clip_norms)
plt.xlabel("Training Step")
plt.ylabel("Clip Norm")
plt.title("Adaptive Clip Norm Evolution")
plt.show()
```

## Fixed Clipping (Comparison)

For comparison, here's standard DP-SGD with **fixed clipping**:

```python
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# Fixed clip norm
clip_norm = 1.0

# Create clipped gradient function
dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,  # Fixed!
    argnums=0,
    batch_argnums=1,
)

# Training loop
for step in range(num_steps):
    grads = dp_grad_fn(params, batch)
    noisy_grads = gaussian_noise(grads, stddev=noise_mult * clip_norm)
    params = update(params, noisy_grads)
```

**When to use fixed clipping**:

- Debugging: Simpler to reason about
- Reproduction: Match published results
- Simplicity: Don't want adaptive complexity

## Best Practices

### 1. Start with Median Quantile

```python
optimizer = adaptive_clipping(base_opt, target_quantile=0.5)  # Recommended
```

### 2. Monitor Clip Norm

```python
if step % 100 == 0:
    print(f"Current clip norm: {current_clip_norm:.3f}")
```

### 3. Use Adam for DP Training

```python
# Adam often works better than SGD for DP
base_opt = torchopt.adam(lr=0.001)
optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)
```

### 4. Tune EMA Decay for Stability

```python
# More stable (slower adaptation)
optimizer = adaptive_clipping(base_opt, ema_decay=0.99)

# More reactive (faster adaptation)
optimizer = adaptive_clipping(base_opt, ema_decay=0.7)
```

## Integration with TorchOpt

Opaque's functional API integrates seamlessly with TorchOpt's functional optimizers:

```python
import torchopt

# TorchOpt optimizer (functional!)
base_opt = torchopt.adam(lr=0.001)

# Add adaptive clipping
dp_optimizer = adaptive_clipping(base_opt, initial_clip_norm=1.0)

# Use functional update
params = dp_optimizer.step(params, noisy_grads)
```

**Why TorchOpt?** Functional optimizers fit naturally with Opaque's functional design, avoiding mutable optimizer state.

## See Also

- **[Tutorial 04](../tutorials/04_dp_optimizers.ipynb)**: Interactive tutorial on DP optimizers
- **[Gradient Clipping](clipping.md)**: Understanding clipping mechanics
- **[API Reference](../api/optimizers.md)**: Detailed optimizer API
- **[TorchOpt Documentation](https://torchopt.readthedocs.io/)**: Learn more about functional optimizers

---

**Next**: Learn about [Poisson Sampling & Microbatching](sampling.md) for privacy amplification
