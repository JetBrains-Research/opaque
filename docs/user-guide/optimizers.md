# Optimizers & Adaptive Clipping

Opaque integrates with **TorchOpt** functional optimizers and provides **adaptive clipping** to automatically tune the
clip norm during training, improving the privacy-utility tradeoff.

## Why Adaptive Clipping?

In standard DP-SGD, you manually set a **fixed clip norm** (e.g., C=1.0). But the "right" clip norm depends on your data
and changes during training:

- **Early training**: Gradients are large → need higher clip norm
- **Late training**: Gradients are small → can use lower clip norm

**Adaptive clipping solves this** by automatically adjusting the clip norm based on gradient statistics.
The clip threshold adapts geometrically: `C_{t+1} = C_t * exp(η * sign(ρ_t - γ))` where `ρ_t` is the clipping rate and `γ` is the target quantile.

### Benefits

✅ **Better accuracy**: Automatically finds optimal clip norm
✅ **Less tuning**: No manual hyperparameter search
✅ **Stable training**: Adapts to changing gradient magnitudes
✅ **Same privacy**: Formal privacy cost tracked via `acc.adaclip()`

## The `adaptive_clipped_grad()` Function

Opaque provides `adaptive_clipped_grad()` for DP gradients with adaptive clip norm tracking:

```python
from opaque.clipping import adaptive_clipped_grad
from opaque import gaussian_noise
import torchopt

# 1. Create adaptive gradient function with initial state
grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,     # Starting clip norm
    target_quantile=0.5,       # Adapt to median gradient norm
    learning_rate=0.2,         # Clip norm adaptation speed
    batch_argnums=1,
)

# 2. Setup optimizer and noise
optimizer = torchopt.adam(lr=0.001)
opt_state = optimizer.init(params)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity()
)

# 3. Training loop with explicit state-passing
for step in range(num_steps):
    # Compute clipped gradients — state passed explicitly
    grads, clip_state = grad_fn(params, batch, state=clip_state)

    # Add noise scaled to current sensitivity
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    # Update parameters
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

    # Monitor adaptation
    if step % 100 == 0:
        print(f"Step {step}: C={clip_state.clip_norm:.4f}, "
              f"ρ={clip_state.clipping_rate:.2%}")
```

### Key Parameters

**`initial_clip_norm`**: Starting clip norm (default: 0.1)

- Use same value you'd use for fixed clipping

**`target_quantile`**: Target fraction of gradients to clip (default: 0.5)

- 0.5 = median (recommended, from Andrew et al. 2021)
- 0.75 = clip more aggressively
- 0.25 = clip less aggressively

**`learning_rate`**: Adaptation speed for clip norm (default: 0.2)

- Controls how fast the clip norm updates geometrically
- Higher = faster adaptation, potentially less stable
- Lower = slower adaptation, more stable

**`clip_norm_min` / `clip_norm_max`**: Bounds on clip norm (default: 0.01 / 100.0)

- Prevents clip norm from becoming too small or too large

## Complete Example with TorchOpt

Here's a full training loop using adaptive clipping:

```python
import torch
import opaque.accounting as acc
from opaque.clipping import adaptive_clipped_grad
from opaque import gaussian_noise
import torchopt

# Setup
batch_size = 32
dataset_size = 10000
sample_rate = batch_size / dataset_size
num_steps = 1000
noise_multiplier = 1.2

# Create adaptive gradient function
grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.5,
    batch_argnums=1,
)

# Create optimizer and noise
optimizer = torchopt.adam(lr=0.001, betas=(0.9, 0.999))
opt_state = optimizer.init(params)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity()
)

# Training loop
for step in range(num_steps):
    # Compute clipped gradients with adaptive norm
    grads, clip_state = grad_fn(params, batch, state=clip_state)

    # Add calibrated noise
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    # Update parameters
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

    if step % 100 == 0:
        print(f"Step {step}: clip_norm={clip_state.clip_norm:.3f}, "
              f"clipping_rate={clip_state.clipping_rate:.2%}")

# Check privacy (adaclip accounts for the quantile estimation cost)
training = acc.adaclip(
    noise_multiplier=noise_multiplier,
    quantile_noise_std=50.0,
) * num_steps
eps = training.epsilon_at(1e-5)
print(f"Final privacy: (ε={eps:.2f}, δ=1e-5)")
```

## Supported TorchOpt Optimizers

`adaptive_clipped_grad()` computes the clipped gradients — you can then use **any** TorchOpt optimizer for the update:

### SGD

```python
optimizer = torchopt.sgd(lr=0.01, momentum=0.9)
opt_state = optimizer.init(params)
```

### Adam

```python
optimizer = torchopt.adam(lr=0.001, betas=(0.9, 0.999))
opt_state = optimizer.init(params)
```

### AdamW

```python
optimizer = torchopt.adamw(lr=0.001, weight_decay=0.01)
opt_state = optimizer.init(params)
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
Step 0:    clip_norm=1.000 (initial)
Step 100:  clip_norm=1.234 (gradients larger than expected)
Step 500:  clip_norm=0.876 (gradients getting smaller)
Step 1000: clip_norm=0.543 (converged, small gradients)
```

The clip norm automatically **increases** when gradients are large and **decreases** as training progresses.

## Learning Rate Scaling (Optional)

When gradients are heavily clipped, the effective learning rate may need adjustment. Handle this by adjusting the optimizer LR based on the `clip_state`:

```python
if clip_state.clipping_rate > 0.8:
    # Most gradients are being clipped — consider a higher LR
    print(f"High clipping rate: {clip_state.clipping_rate:.0%}")
```

!!! warning "Use with caution"
    LR scaling when combined with DP noise is experimental. Start simple and adjust only if needed.

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

Track how the clip norm evolves via the `clip_state`:

```python
clip_norms = []

for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    clip_norms.append(clip_state.clip_norm)

    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

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
from opaque import clipped_grad, gaussian_noise

# Fixed clip norm
clip_norm = 1.0

# Create clipped gradient function
dp_grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,  # Fixed!
    argnums=0,
    batch_argnums=1,
)

# Training loop
noise_fn, noise_state = gaussian_noise(stddev=noise_mult * clip_norm)
for step in range(num_steps):
    grads, clip_state = dp_grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)
```

**When to use fixed clipping**:

- Debugging: Simpler to reason about
- Reproduction: Match published results
- Simplicity: Don't want adaptive complexity

## Best Practices

### 1. Start with Median Quantile

```python
grad_fn, clip_state = adaptive_clipped_grad(loss_fn, target_quantile=0.5, batch_argnums=1)  # Recommended
```

### 2. Monitor Clip Norm

```python
if step % 100 == 0:
    print(f"Current clip norm: {clip_state.clip_norm:.3f}")
```

### 3. Use Adam for DP Training

```python
# Adam often works better than SGD for DP
optimizer = torchopt.adam(lr=0.001)
opt_state = optimizer.init(params)
```

### 4. Tune Adaptation Speed

```python
# More stable (slower adaptation)
grad_fn, state = adaptive_clipped_grad(loss_fn, learning_rate=0.1, batch_argnums=1)

# More reactive (faster adaptation)
grad_fn, state = adaptive_clipped_grad(loss_fn, learning_rate=0.5, batch_argnums=1)
```

## Integration with TorchOpt

Opaque's functional API integrates seamlessly with TorchOpt's functional optimizers:

```python
import torchopt
from opaque.clipping import adaptive_clipped_grad

# Functional gradient computation
grad_fn, clip_state = adaptive_clipped_grad(loss_fn, batch_argnums=1)

# Functional optimizer
optimizer = torchopt.adam(lr=0.001)
opt_state = optimizer.init(params)

# Functional update
grads, clip_state = grad_fn(params, batch, state=clip_state)
updates, opt_state = optimizer.update(grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

**Why TorchOpt?** Functional optimizers fit naturally with Opaque's functional design — no mutable optimizer state.

## See Also

- **[Tutorial 04](../tutorials/04_dp_optimizers.ipynb)**: Interactive tutorial on DP optimizers
- **[Gradient Clipping](clipping.md)**: Understanding clipping mechanics
- **[API Reference](../api/optimizers.md)**: Detailed optimizer API
- **[TorchOpt Documentation](https://torchopt.readthedocs.io/)**: Learn more about functional optimizers

---

**Next**: Learn about [Poisson Sampling & Microbatching](sampling.md) for privacy amplification
