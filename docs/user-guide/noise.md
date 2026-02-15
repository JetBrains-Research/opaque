# Noise Addition

After [clipping gradients](clipping.md), the next step in DP-SGD is **adding calibrated Gaussian noise**. This noise is
what actually provides the privacy guarantee by obscuring individual contributions.

## Why Add Noise?

Even with gradient clipping, we can still detect whether a specific person's data was in the training set by observing
the exact gradient values. **Noise solves this** by making the gradients random enough that we can't distinguish "with
person X" from "without person X".

### The Intuition

Imagine you're trying to figure out if Alice's data was used to train a model:

- **Without noise**: Gradients change predictably when Alice is added/removed → Easy to detect
- **With noise**: Gradients are randomized → Can't tell if differences are from Alice or random noise

The more noise we add, the stronger the privacy guarantee, but the worse the model accuracy.

## The `gaussian_noise()` Function

Opaque provides a simple, stateless function for adding noise:

```python
from opaque import gaussian_noise

# After computing clipped gradients
clipped_grads = clipped_grad_fn(params, batch)

# Add Gaussian noise (std = noise_multiplier * clip_norm)
noisy_grads = gaussian_noise(
    clipped_grads,
    stddev=noise_multiplier * clip_norm,
)

# Update parameters with noisy gradients
params = update(params, noisy_grads)
```

### Key Parameters

**`stddev`**: Standard deviation of Gaussian noise

- Higher stddev → Stronger privacy, lower accuracy
- Lower stddev → Weaker privacy, higher accuracy
- Typically: `stddev = noise_multiplier * clip_norm`

**`generator`** (optional): PyTorch random number generator for reproducibility

```python
gen = torch.Generator().manual_seed(42)
noisy_grads = gaussian_noise(grads, stddev=1.0, generator=gen)
```

## Calibrating Noise

**Don't guess the noise level!** Use Opaque's calibration to find the minimum noise for your target privacy:

### Calibration for (ε, δ)-DP

```python
import opaque.accounting as acc

# Find minimum noise for target privacy
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,  # Target privacy loss
    delta=1e-5,  # Failure probability
    sample_rate=batch_size / dataset_size,
    num_steps=total_training_steps,
)

print(f"Use noise multiplier: {noise_multiplier:.3f}")
# Example output: Use noise multiplier: 1.234
```

**Result**: Training with this noise will give you *exactly* (ε=3.0, δ=1e-5) privacy after `num_steps` steps.

### Calibration for f-DP Advantage

For tighter bounds using f-DP:

```python
noise_multiplier = acc.find_noise_multiplier_for_advantage(
    advantage=0.1,  # Target advantage (lower = stronger privacy)
    sample_rate=batch_size / dataset_size,
    num_steps=total_training_steps,
)
```

### Calibration for (α, β) Error Rates

For hypothesis testing interpretation:

```python
noise_multiplier = acc.find_noise_multiplier_for_err_rates(
    alpha=1e-4,  # False positive rate
    beta=0.8,  # True positive rate (1 - power)
    sample_rate=batch_size / dataset_size,
    num_steps=total_training_steps,
)
```

## How Much Noise is Needed?

The amount of noise depends on:

1. **Target privacy** (ε): Stronger privacy → More noise
2. **Dataset size**: Larger dataset → Less noise (privacy amplification)
3. **Training steps**: More steps → More noise (privacy degradation)
4. **Clip norm**: Higher clip norm → More noise

### Typical Noise Multipliers

For reference, here are typical noise multipliers for different privacy levels:

| Privacy (ε, δ) | Sample Rate | Steps | Noise Multiplier |
|----------------|-------------|-------|------------------|
| (1.0, 1e-5)    | 0.01        | 1000  | ~3.5             |
| (3.0, 1e-5)    | 0.01        | 1000  | ~1.2             |
| (10.0, 1e-5)   | 0.01        | 1000  | ~0.4             |
| (3.0, 1e-5)    | 0.1         | 1000  | ~0.5             |

!!! tip "Higher sample rate = less noise"
Larger batches (higher sample rate) provide privacy amplification, requiring less noise for the same ε.

## Noise and the Privacy-Utility Tradeoff

Adding noise is where the **privacy-utility tradeoff** becomes concrete:

### Strong Privacy (ε=1, noise_multiplier~3.5)

- ✅ Very strong privacy guarantee
- ❌ High noise → Slower convergence
- ❌ May need more epochs or larger batch sizes

### Moderate Privacy (ε=3, noise_multiplier~1.2)

- ✅ Good privacy guarantee (industry standard)
- ✅ Reasonable noise level
- ✅ Often achieves good accuracy

### Weak Privacy (ε=10, noise_multiplier~0.4)

- ⚠️ Weak privacy guarantee
- ✅ Low noise → Faster convergence
- ✅ Closer to non-private accuracy

!!! warning "ε=10 is often too weak"
For sensitive data (medical, financial), aim for ε ≤ 3. ε=10 provides limited privacy.

## Complete DP-SGD Training Loop

Here's how gradient clipping and noise addition work together:

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# 1. Setup
clip_norm = 1.0
batch_size = 32
dataset_size = 10000
sample_rate = batch_size / dataset_size
num_steps = 1000

# 2. Calibrate noise
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# 3. Create DP gradient function
dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,
    argnums=0,
    batch_argnums=1,
)

# 4. Training loop
privacy_state = acc.create()

for step in range(num_steps):
    # Compute clipped gradients
    grads = dp_grad_fn(params, batch)

    # Add calibrated noise
    noisy_grads = gaussian_noise(
        grads,
        stddev=noise_multiplier * clip_norm,
    )

    # Update parameters
    params = update(params, noisy_grads)

    # Track privacy
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        count=1,
    )

# 5. Verify privacy
final_epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Final privacy: (ε={final_epsilon:.2f}, δ=1e-5)")
```

## Bounded Gaussian Noise

For applications where noisy gradient values must stay within a valid range, Opaque provides a **bounded Gaussian
mechanism** based on [Chen & Hale (2024)](https://arxiv.org/abs/2211.17230). Instead of unbounded Gaussian noise, this
uses a **truncated normal distribution** restricted to a domain `[lower, upper]`.

```python
from opaque import bounded_gaussian_noise

# Outputs are guaranteed to lie in [-3.0, 3.0]
noise_fn = bounded_gaussian_noise(stddev=noise_multiplier * clip_norm, bounds=(-3.0, 3.0))

noisy_grads = noise_fn(clipped_grads)
```

### When to Use Bounded Gaussian

| Scenario | Recommended Mechanism |
|----------|----------------------|
| General DP-SGD training | `gaussian_noise()` (standard) |
| Gradient values must stay in a valid range | `bounded_gaussian_noise()` |
| Queries with bounded output domains | `bounded_gaussian_noise()` |

### Reproducibility

Use `bounded_gaussian_noise_stateful()` for deterministic noise:

```python
from opaque import bounded_gaussian_noise_stateful

noise_fn, state = bounded_gaussian_noise_stateful(stddev=1.0, bounds=(-3.0, 3.0), seed=42)
noisy_grads = noise_fn(grads, state)
```

## Advanced: Noise Multiplier Schedule

Instead of fixed noise, you can use a **noise schedule** that decreases over time:

```python
def noise_schedule(step, total_steps):
    """Decrease noise over time (experimental!)"""
    initial_noise = 2.0
    final_noise = 0.5
    progress = step / total_steps
    return initial_noise + progress * (final_noise - initial_noise)

for step in range(num_steps):
    grads = dp_grad_fn(params, batch)

    current_noise = noise_schedule(step, num_steps)
    noisy_grads = gaussian_noise(grads, stddev=current_noise * clip_norm)

    params = update(params, noisy_grads)

    # Important: Track with actual noise used!
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state,
        noise_multiplier=current_noise,
        sample_rate=sample_rate,
        count=1,
    )
```

!!! warning "Use with caution"
Noise schedules can improve accuracy but may weaken privacy if not carefully analyzed. Stick with fixed noise unless you
understand privacy composition.

## Comparison with Standard SGD

| Feature         | Standard SGD        | DP-SGD with Noise  |
|-----------------|---------------------|--------------------|
| **Gradients**   | Exact batch average | Clipped + noisy    |
| **Updates**     | Deterministic       | Randomized         |
| **Convergence** | Smooth              | Noisier trajectory |
| **Privacy**     | ❌ None              | ✅ (ε, δ)-DP        |
| **Accuracy**    | Maximum             | Slightly lower     |

## Debugging Noise Issues

### Problem: Model doesn't learn at all

**Symptoms**: Loss stays constant or decreases very slowly

**Solutions**:

1. Reduce noise: Increase ε (e.g., ε=10) temporarily
2. Increase learning rate: Try 2-5x higher LR
3. Check clip norm: May be too aggressive
4. Increase batch size: Provides privacy amplification

### Problem: Privacy budget exceeded

**Symptoms**: `final_epsilon > target_epsilon`

**Solutions**:

1. Reduce training steps: Train fewer epochs
2. Increase noise: Accept lower accuracy
3. Increase batch size: More privacy amplification
4. Use truncated Poisson sampling: Tighter bounds

### Problem: Training too slow

**Symptoms**: Each step takes much longer than non-DP training

**Solutions**:

1. Use microbatching: Reduce memory usage
2. Decrease batch size: Less per-example gradients
3. Use LoRA: Fewer parameters to differentiate

## PyTree Support

`gaussian_noise()` works with any PyTree structure:

```python
# Nested dictionaries of tensors
grads = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}

noisy_grads = gaussian_noise(grads, stddev=1.0)
# Noise added independently to each tensor
```

## Reproducibility

For reproducible noise across runs:

```python
# Set global seed
torch.manual_seed(42)

# Or use per-call generator
gen = torch.Generator().manual_seed(42)
noisy_grads = gaussian_noise(grads, stddev=1.0, generator=gen)
```

## See Also

- **[Gradient Clipping](clipping.md)**: Previous step before noise addition
- **[Privacy Accounting](accounting.md)**: Track privacy after adding noise
- **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)**: Interactive noise and accounting
  tutorial
- **[API Reference](../api/noise.md)**: Detailed API documentation

---

**Next**: Learn about [Privacy Accounting](accounting.md) to track your privacy budget
