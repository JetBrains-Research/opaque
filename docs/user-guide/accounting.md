# Privacy Accounting

Privacy accounting is how we **track and query the privacy budget** consumed during DP training. Opaque uses a *
*functional API** with immutable state for composable, principled privacy tracking.

## Why Accounting Matters

Every time you train on data with DP-SGD, you "spend" some privacy budget. Once you've spent your budget (ε, δ), you
cannot train more without weakening your privacy guarantee.

**Privacy accounting answers**:

- How much privacy have I spent so far?
- How many more training steps can I afford?
- What noise level do I need for my target privacy?

## The Functional Accounting API

Opaque's accounting uses **immutable state** and **pure functions**:

```python
import opaque.accounting as acc

# 1. Create initial state (immutable)
privacy_state = acc.create()

# 2. Compose privacy over training steps (returns NEW state)
privacy_state = acc.compose_poisson_gaussian(
    privacy_state,
    noise_multiplier=1.2,
    sample_rate=0.01,
    count=100,  # 100 training steps
)

# 3. Query current privacy (pure function)
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Privacy spent: ε={epsilon:.2f}")
```

### Key Principles

1. **Immutable state**: `compose_*()` functions return NEW states, don't modify existing
2. **Pure functions**: Same inputs → same outputs, no side effects
3. **Composable**: Chain multiple mechanisms naturally

## Composition Functions

### `compose_poisson_gaussian()`

**Use when**: Training with Poisson sampling (batch size ≈ sample_rate × dataset_size)

```python
privacy_state = acc.compose_poisson_gaussian(
    privacy_state,
    noise_multiplier=1.2,  # Noise stddev / clip_norm
    sample_rate=32 / 10000,  # batch_size / dataset_size
    count=1000,  # Number of training steps
)
```

**Why Poisson?** Each example is sampled independently with probability `sample_rate`, providing **privacy amplification
** through subsampling.

### `compose_truncated_poisson_gaussian()` ⭐

**Use when**: You want tight privacy bounds with bounded batch sizes

```python
privacy_state = acc.compose_truncated_poisson_gaussian(
    privacy_state,
    noise_multiplier=1.2,
    sample_rate=32 / 10000,
    truncated_batch_size=32,  # Maximum batch size
    dataset_size=10000,
    count=1000,
)
```

**Advantage**: Tighter privacy bounds than standard Poisson (up to 20% improvement)!

**When to use**: Always, unless you have a specific reason not to

### `compose_sampled_gaussian()`

**Use when**: Fixed batch sizes without Poisson sampling

```python
privacy_state = acc.compose_sampled_gaussian(
    privacy_state,
    noise_multiplier=1.2,
    sample_rate=0.01,
    count=1000,
)
```

**Note**: Provides weaker privacy bounds than Poisson sampling

### `compose_gaussian()`

**Use when**: No sampling (e.g., processing entire dataset)

```python
privacy_state = acc.compose_gaussian(
    privacy_state,
    noise_multiplier=1.2,
    count=1,
)
```

**Rarely used** in practice since DP-SGD almost always uses sampling

## Privacy Queries

Opaque supports three privacy metrics. You can query any of them from the same privacy state!

### 1. (ε, δ)-Differential Privacy

The **standard metric** used in most DP papers:

```python
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")
```

**Interpretation**: Adding/removing any single person changes the model with probability ≤ e^ε with probability 1-δ

**Typical values**:

- Strong privacy: ε ≤ 1
- Moderate privacy: ε ∈ [1, 3]
- Weak privacy: ε > 10

### 2. f-DP Advantage

A **tighter bound** than (ε, δ)-DP, from [Dong et al. 2019](https://arxiv.org/abs/1905.02383):

```python
advantage = acc.get_advantage(privacy_state)
print(f"f-DP advantage: {advantage:.4f}")
```

**Interpretation**: Maximum advantage in distinguishing neighboring datasets

**Advantage**: Can be 10-30% tighter than ε for same noise level

### 3. (α, β) Error Rates

**Hypothesis testing** interpretation:

```python
beta = acc.get_beta(privacy_state, alpha=0.01)
print(f"Error rates: (α={0.01}, β={beta:.3f})")
```

**Interpretation**:

- α: Probability of false positive (detecting person when not present)
- β: Probability of false negative (missing person when present)

## Calibration: Finding the Right Noise

Instead of guessing noise levels, **calibrate** to find the minimum noise for your target privacy:

### Calibrate for (ε, δ)

```python
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,  # Target privacy
    delta=1e-5,  # Failure probability
    sample_rate=32 / 10000,
    num_steps=1000,
)

print(f"Use noise multiplier: {noise_multiplier:.3f}")
```

### Calibrate for Advantage

```python
noise_multiplier = acc.find_noise_multiplier_for_advantage(
    advantage=0.1,  # Target advantage
    sample_rate=0.01,
    num_steps=1000,
)
```

### Calibrate for Error Rates

```python
noise_multiplier = acc.find_noise_multiplier_for_err_rates(
    alpha=1e-4,  # False positive rate
    beta=0.8,  # True positive rate (1 - power)
    sample_rate=0.01,
    num_steps=1000,
)
```

## Complete Training Example

Here's a full DP-SGD training loop with accounting:

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# Setup
clip_norm = 1.0
batch_size = 32
dataset_size = 10000
sample_rate = batch_size / dataset_size
target_epsilon = 3.0
target_delta = 1e-5
num_epochs = 10
steps_per_epoch = dataset_size // batch_size
num_steps = num_epochs * steps_per_epoch

# Calibrate noise
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=target_epsilon,
    delta=target_delta,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# Create DP gradient function
dp_grad_fn = clipped_grad(loss_fn, l2_clip_norm=clip_norm, ...)

# Initialize privacy state
privacy_state = acc.create()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Compute noisy gradients
        grads = dp_grad_fn(params, batch)
        noisy_grads = gaussian_noise(grads, stddev=noise_multiplier * clip_norm)

        # Update parameters
        params = update(params, noisy_grads)

        # Update privacy accounting
        privacy_state = acc.compose_poisson_gaussian(
            privacy_state,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            count=1,
        )

    # Check privacy at end of each epoch
    current_epsilon = acc.get_epsilon(privacy_state, delta=target_delta)
    print(f"Epoch {epoch+1}: ε={current_epsilon:.2f}/{target_epsilon:.2f}")

# Verify final privacy
final_epsilon = acc.get_epsilon(privacy_state, delta=target_delta)
assert final_epsilon <= target_epsilon + 0.1, "Privacy budget exceeded!"
print(f"Training complete! Final privacy: (ε={final_epsilon:.2f}, δ={target_delta})")
```

## Privacy Composition Basics

Privacy degrades as you train more:

```python
# After 100 steps
privacy_state = acc.compose_poisson_gaussian(state, noise=1.2, rate=0.01, count=100)
epsilon_100 = acc.get_epsilon(privacy_state, delta=1e-5)  # ε ≈ 0.3

# After 1000 steps
privacy_state = acc.compose_poisson_gaussian(state, noise=1.2, rate=0.01, count=1000)
epsilon_1000 = acc.get_epsilon(privacy_state, delta=1e-5)  # ε ≈ 3.0

# After 10000 steps
privacy_state = acc.compose_poisson_gaussian(state, noise=1.2, rate=0.01, count=10000)
epsilon_10000 = acc.get_epsilon(privacy_state, delta=1e-5)  # ε ≈ 30.0
```

!!! warning "Privacy degrades with training"
More training steps → higher ε → weaker privacy. Plan your training budget carefully!

## Privacy Amplification Through Sampling

Subsampling **amplifies privacy** — you get stronger guarantees for the same noise:

```python
# No sampling (full batch)
state_full = acc.compose_gaussian(state, noise_multiplier=1.0, count=100)
eps_full = acc.get_epsilon(state_full, delta=1e-5)  # ε ≈ 15

# With sampling (sample_rate=0.01)
state_sample = acc.compose_poisson_gaussian(state, noise=1.0, rate=0.01, count=100)
eps_sample = acc.get_epsilon(state_sample, delta=1e-5)  # ε ≈ 0.1

print(f"Full batch: ε={eps_full:.1f}")
print(f"Sampled: ε={eps_sample:.1f}")  # 150x better!
```

**Key insight**: Larger batches (higher sample rate) provide less amplification but enable more stable training.

## Monitoring Privacy During Training

Track privacy consumption in real-time:

```python
privacy_state = acc.create()

for step in range(num_steps):
    # Training step
    grads = dp_grad_fn(params, batch)
    noisy_grads = gaussian_noise(grads, stddev=noise_multiplier * clip_norm)
    params = update(params, noisy_grads)

    # Update privacy
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state, noise_multiplier=noise_multiplier, sample_rate=sample_rate, count=1
    )

    # Log every 100 steps
    if step % 100 == 0:
        eps = acc.get_epsilon(privacy_state, delta=1e-5)
        advantage = acc.get_advantage(privacy_state)
        print(f"Step {step}: ε={eps:.2f}, advantage={advantage:.4f}")

        # Early stop if budget exceeded
        if eps > target_epsilon:
            print(f"Privacy budget exceeded at step {step}!")
            break
```

## Understanding δ (Delta)

δ is the **failure probability** — the probability that privacy guarantee fails:

**Typical values**:

- δ = 1/n (inverse of dataset size)
- δ = 1/n² (more conservative)
- δ = 1e-5 or 1e-6 (fixed small value)

**Guideline**: Set δ much smaller than 1/dataset_size

```python
dataset_size = 10000
delta = 1 / dataset_size  # δ = 1e-4
# Or more conservatively:
delta = 1 / (dataset_size ** 2)  # δ = 1e-8
```

## Best Practices

### 1. Always Calibrate Noise

!!! success "Use calibration functions"
Don't guess noise multipliers! Use `find_noise_multiplier_for_epsilon_delta()`.

### 2. Monitor Privacy During Training

```python
if step % 100 == 0:
    current_eps = acc.get_epsilon(privacy_state, delta)
    if current_eps > target_epsilon * 1.1:  # 10% buffer
        raise RuntimeError("Privacy budget exceeded!")
```

### 3. Use Truncated Poisson When Possible

```python
# Tighter bounds (preferred)
privacy_state = acc.compose_truncated_poisson_gaussian(
    privacy_state, noise, rate, truncated_batch_size, dataset_size, count
)

# vs standard Poisson
privacy_state = acc.compose_poisson_gaussian(privacy_state, noise, rate, count)
```

### 4. Query Multiple Metrics

```python
# Compare different privacy metrics
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
advantage = acc.get_advantage(privacy_state)
beta = acc.get_beta(privacy_state, alpha=1e-4)

print(f"(ε, δ)-DP: (ε={epsilon:.2f}, δ=1e-5)")
print(f"f-DP advantage: {advantage:.4f}")
print(f"Error rates: (α=1e-4, β={beta:.3f})")
```

## Comparison with OOP Accountants

Opaque v0.1.0 used OOP accountants (`PLDAccountant`, `RDPAccountant`). The functional API is now **preferred**:

| Feature           | OOP API (deprecated)            | Functional API (current)                           |
|-------------------|---------------------------------|----------------------------------------------------|
| **State**         | Mutable object                  | Immutable                                          |
| **Composition**   | `accountant.step_poisson(...)`  | `state = acc.compose_poisson_gaussian(state, ...)` |
| **Queries**       | `accountant.get_epsilon(delta)` | `acc.get_epsilon(state, delta)`                    |
| **Immutability**  | ❌ Modifies object               | ✅ Returns new state                                |
| **Concurrency**   | ⚠️ Not thread-safe              | ✅ Thread-safe                                      |
| **Composability** | ⚠️ Limited                      | ✅ Highly composable                                |

## See Also

- **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)**: Interactive accounting tutorial
- **[Tutorial 03](../tutorials/03_complete_dp_sgd_training.ipynb)**: Complete DP-SGD with accounting
- **[API Reference](../api/accounting.md)**: Detailed function documentation
- **[Noise Addition](noise.md)**: How noise and accounting work together

---

**Next**: Explore [Optimizers & Adaptive Clipping](optimizers.md) for better utility
