# Privacy Accounting

Privacy accounting is how we **track and query the privacy budget** consumed during DP training. Opaque provides a composable, functional accounting API backed by Privacy Loss Distributions (PLD) for the tightest known bounds.

## Why Accounting Matters

Every time you train on data with DP-SGD, you "spend" some privacy budget. Once you've spent your budget (epsilon, delta), you cannot train more without weakening your privacy guarantee.

**Privacy accounting answers**:

- How much privacy have I spent so far?
- How many more training steps can I afford?
- What noise level do I need for my target privacy?

## Quick Start

```python
import opaque_dp_accounting as dp

# Build a DP-SGD process: Poisson-subsampled Gaussian, 1000 steps
step = dp.poisson(noise_multiplier=1.1, sample_rate=0.01)
training = step * 1000

# Query privacy
eps = training.epsilon_at(delta=1e-5)
print(f"Privacy spent: epsilon={eps:.2f}")
```

## Core Concepts

### Everything is a `DpProcess`

Every mechanism constructor returns a `DpProcess`, and composition operators produce new `DpProcess` instances. Privacy metrics are computed on demand from the underlying PLD.

```python
import opaque_dp_accounting as dp

# Mechanisms
g = dp.gaussian(1.1)                     # Gaussian mechanism
p = dp.poisson(1.1, 0.01)               # Poisson-subsampled Gaussian
tp = dp.truncated_poisson(1.1, 0.01,    # Truncated Poisson (production DP-SGD)
         batch_size_cap=100, dataset_size=10000)

# Composition
training = p * 1000                      # repeat 1000 times
combined = g | dp.eps_delta(0.5)         # compose two different processes

# Query privacy (all derived from the same PLD)
eps = training.epsilon_at(1e-5)          # (epsilon, delta)-DP
delta = training.delta_at(1.0)           # inverse direction
adv = training.advantage()               # f-DP advantage
beta = training.beta_at(0.05)            # Type-II error
risk = training.risk_at(0.5)             # Bayes risk
```

### Key Principles

1. **Composable**: Mechanisms compose via operators (`*` for repetition, `|` for heterogeneous composition)
2. **Functional**: No mutable state — each operation produces a new process
3. **Tight bounds**: PLD-based accounting with FFT convolution gives optimal bounds
4. **Multiple metrics**: All privacy metrics derived from a single PLD computation

## Mechanisms

### `dp.poisson()` — Standard DP-SGD

**Use when**: Training with Poisson sampling (batch size ~ sample_rate x dataset_size)

```python
step = dp.poisson(
    noise_multiplier=1.2,           # noise stddev / clip_norm
    sample_rate=32 / 10000,         # batch_size / dataset_size
)
training = step * 1000              # 1000 training steps
eps = training.epsilon_at(1e-5)
```

**Why Poisson?** Each example is sampled independently with probability `sample_rate`, providing **privacy amplification through subsampling**. With `sample_rate=0.01` the effective epsilon can be 50-100x smaller than the un-subsampled Gaussian.

### `dp.truncated_poisson()` — Production DP-SGD

**Use when**: You want tight privacy bounds with bounded batch sizes

```python
step = dp.truncated_poisson(
    noise_multiplier=1.2,
    sample_rate=32 / 10000,
    batch_size_cap=32,              # maximum batch size
    dataset_size=10000,
)
training = step * 1000
eps = training.epsilon_at(1e-5)
```

**Advantage**: Tighter privacy bounds than standard Poisson (up to 20% improvement). This matches what production DP-SGD frameworks (Opacus, JAX Privacy, TF Privacy) actually do.

**When to use**: Always, unless you have a specific reason not to.

### `dp.gaussian()` — No Sampling

**Use when**: Processing entire dataset (no subsampling)

```python
proc = dp.gaussian(noise_multiplier=1.2)
eps = proc.epsilon_at(1e-5)
```

**Rarely used** in practice since DP-SGD almost always uses sampling.

### `dp.accumulate()` — Gradient Accumulation

**Use when**: Memory-limited training with microbatching

```python
step = dp.accumulate(
    noise_multiplier=1.1,
    sample_rate=0.01,
    microbatches=4,                 # 4 microbatches per noise step
)
training = step * 500
eps = training.epsilon_at(1e-5)
```

### `dp.adaclip()` — Adaptive Clipping

**Use when**: Automatically adjusting clipping threshold (Andrew et al. 2021)

```python
step = dp.adaclip(
    noise_multiplier=1.1,
    quantile_noise_std=50.0,        # noise for quantile estimation
)
eps = step.epsilon_at(1e-5)

# With Poisson subsampling
step = dp.poisson_adaclip(1.1, quantile_noise_std=50.0, sample_rate=0.01)
```

### `dp.eps_delta()` — Fixed Guarantee

**Use when**: Composing a non-Gaussian mechanism with known (epsilon, delta)

```python
proc = dp.eps_delta(epsilon=1.0, delta=1e-5)
combined = dp.gaussian(1.1) | proc  # compose with a Gaussian
```

### `dp.identity()` — Zero Privacy Loss

**Use when**: Representing a no-op in composition chains

```python
proc = dp.identity()
assert proc.epsilon_at(1e-5) < 1e-10
```

## Composition

### Homogeneous Composition (Repetition)

Use `*` or `dp.repeat()` for k-fold composition of the same mechanism:

```python
step = dp.poisson(1.1, 0.01)

# These are equivalent
training = step * 1000
training = dp.repeat(step, 1000)
```

### Heterogeneous Composition

Use `|` or `dp.compose()` to compose different mechanisms:

```python
a = dp.gaussian(1.0)
b = dp.gaussian(0.8)

# These are equivalent
combined = a | b
combined = dp.compose(a, b)

assert combined.epsilon_at(1e-5) > a.epsilon_at(1e-5)
```

## Privacy Metrics

All metrics are computed from the same Privacy Loss Distribution — no redundant computation.

### (epsilon, delta)-Differential Privacy

The **standard metric** used in most DP papers:

```python
proc = dp.poisson(1.1, 0.01) * 1000
eps = proc.epsilon_at(delta=1e-5)
delta = proc.delta_at(epsilon=1.0)
print(f"Privacy: (epsilon={eps:.2f}, delta=1e-5)")
```

**Interpretation**: Adding/removing any single person changes the model output by at most a factor of exp(epsilon), except with probability delta.

**Typical values**:

- Strong privacy: epsilon <= 1
- Moderate privacy: epsilon in [1, 3]
- Weak privacy: epsilon > 10

### f-DP Advantage

A **tighter bound** than (epsilon, delta)-DP, from [Dong et al. 2019](https://arxiv.org/abs/1905.02383):

```python
adv = proc.advantage()
print(f"f-DP advantage: {adv:.4f}")
```

**Interpretation**: Maximum probability of distinguishing neighboring datasets. Equal to `delta_at(0)`.

### (alpha, beta) Error Rates

**Hypothesis testing** interpretation:

```python
beta = proc.beta_at(alpha=0.01)
print(f"Error rates: (alpha=0.01, beta={beta:.3f})")
```

**Interpretation**:

- alpha: Probability of false positive (detecting person when not present)
- beta: Probability of false negative (missing person when present)

### Bayes Risk

```python
risk = proc.risk_at(prior=0.5)
print(f"Bayes risk: {risk:.4f}")
```

**Interpretation**: Minimum expected loss of any decision rule, weighted by prior.

## Calibration

Instead of guessing noise levels, **calibrate** to find the minimum noise for your target privacy:

```python
nm = dp.calibrate_noise(
    target_epsilon=8.0,
    target_delta=1e-5,
    sample_rate=0.01,
    num_steps=1000,
)
print(f"Use noise_multiplier={nm:.4f}")

# Verify
actual_eps = dp.compute_epsilon(nm, 0.01, 1000, delta=1e-5)
assert abs(actual_eps - 8.0) < 0.1
```

### Quick Epsilon Computation

For one-off privacy checks:

```python
eps = dp.compute_epsilon(
    noise_multiplier=1.1,
    sample_rate=0.01,
    num_steps=1000,
    delta=1e-5,
)
```

## Debugging and Introspection

### Quick Display

```python
proc = dp.poisson(1.1, 0.01) * 1000
print(proc)
# Repeat(Poisson(noise_multiplier=1.1, sample_rate=0.01), k=1000) | eps(delta=1e-5)=3.73
```

### Constructor Parameters

```python
proc = dp.poisson(1.1, 0.01)
proc.describe()
# {'type': 'Poisson(...)', 'noise_multiplier': 1.1, 'sample_rate': 0.01}
```

### PLD Grid Diagnostics

Inspect the internal PLD representation for debugging numerical precision:

```python
info = proc.pld_info()
print(f"Grid size: {info['grid_size']} bins")
print(f"Discretization: {info['discretization']}")
print(f"Total mass: {info['total_mass']}")    # should be ~1.0
print(f"Inf mass: {info['infinity_mass']:.2e}")
print(f"Computed in {info['elapsed_ms']:.1f} ms")
```

### Full Summary

```python
print(proc.summary(delta=1e-5))
# --- Poisson(noise_multiplier=1.1, sample_rate=0.01) ---
# epsilon(delta=1e-5)  = 3.73
# delta(epsilon=1)     = 2.1e-02
# advantage            = 4.5e-01
# beta(alpha=0.05)     = 0.12
# risk(prior=0.5)      = 0.38
# ---
# PLD grid: 84001 bins, disc=0.0001, inf_mass=1.2e-10
# PLD computed in 42.3 ms
```

## Custom Precision

Override default PLD discretization for faster or more precise computation:

```python
# Faster (coarser grid)
cfg = dp.DiscretizationConfig(discretization=1e-3)

# More precise (finer grid, wider tails)
cfg = dp.DiscretizationConfig(
    discretization=1e-5,
    log_mass_truncation_bound=-50.0,
)

# Use with any mechanism
proc = dp.gaussian(1.1, config=cfg)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discretization` | 1e-4 | Grid spacing. Error scales as O(disc^2) per step |
| `log_mass_truncation_bound` | -32.0 | Tails below 2^bound are truncated |
| `pessimistic_estimate` | True | Upper-bound rounding (safe for privacy) |
| `max_grid_size` | 10,000,000 | Auto-coarsen if grid exceeds this |

## Complete Training Example

Here's a full DP-SGD training loop with accounting:

```python
import torch
import opaque_dp_accounting as dp
from opaque import clipped_grad, add_gaussian_noise

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
noise_multiplier = dp.calibrate_noise(
    target_epsilon=target_epsilon,
    target_delta=target_delta,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# Create DP gradient function
dp_grad_fn = clipped_grad(loss_fn, l2_clip_norm=clip_norm, ...)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        grads = dp_grad_fn(params, batch)
        noisy_grads = add_gaussian_noise(grads, stddev=noise_multiplier * clip_norm)
        params = update(params, noisy_grads)

    # Check privacy at end of each epoch
    steps_so_far = (epoch + 1) * steps_per_epoch
    current_eps = dp.compute_epsilon(noise_multiplier, sample_rate, steps_so_far, target_delta)
    print(f"Epoch {epoch+1}: epsilon={current_eps:.2f}/{target_epsilon:.2f}")

# Verify final privacy
final_eps = dp.compute_epsilon(noise_multiplier, sample_rate, num_steps, target_delta)
assert final_eps <= target_epsilon + 0.1, "Privacy budget exceeded!"
```

## Privacy Composition Basics

Privacy degrades as you train more:

```python
# Compare epsilon at different training durations
for steps in [100, 1000, 10000]:
    eps = dp.compute_epsilon(1.2, 0.01, steps, delta=1e-5)
    print(f"After {steps:>5} steps: epsilon={eps:.2f}")
```

!!! warning "Privacy degrades with training"
    More training steps -> higher epsilon -> weaker privacy. Plan your training budget carefully!

## Privacy Amplification Through Sampling

Subsampling **amplifies privacy** -- you get stronger guarantees for the same noise:

```python
# No sampling (full batch)
eps_full = dp.gaussian(1.0).epsilon_at(1e-5)

# With Poisson sampling (sample_rate=0.01)
eps_sampled = dp.poisson(1.0, 0.01).epsilon_at(1e-5)

print(f"Full batch:  epsilon={eps_full:.2f}")
print(f"Sampled:     epsilon={eps_sampled:.4f}")  # Much smaller!
```

**Key insight**: Larger batches (higher sample rate) provide less amplification but enable more stable training.

## Understanding delta

delta is the **failure probability** -- the probability that the privacy guarantee fails:

**Typical values**:

- delta = 1/n (inverse of dataset size)
- delta = 1/n^2 (more conservative)
- delta = 1e-5 or 1e-6 (fixed small value)

**Guideline**: Set delta much smaller than 1/dataset_size.

## Best Practices

### 1. Always Calibrate Noise

!!! success "Use calibration"
    Don't guess noise multipliers! Use `dp.calibrate_noise()`.

### 2. Use Truncated Poisson When Possible

```python
# Tighter bounds (preferred)
step = dp.truncated_poisson(nm, rate, batch_size_cap=B, dataset_size=n)

# vs standard Poisson
step = dp.poisson(nm, rate)
```

### 3. Query Multiple Metrics

```python
proc = dp.poisson(1.1, 0.01) * 1000
print(f"epsilon(delta=1e-5) = {proc.epsilon_at(1e-5):.2f}")
print(f"advantage           = {proc.advantage():.4f}")
print(f"beta(alpha=0.05)    = {proc.beta_at(0.05):.4f}")
```

### 4. Use Introspection for Debugging

```python
# Check PLD grid for numerical issues
info = proc.pld_info()
assert abs(info['total_mass'] - 1.0) < 1e-8, "PLD mass not conserved!"
```

## See Also

- **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)**: Interactive accounting tutorial
- **[Tutorial 03](../tutorials/03_complete_dp_sgd_training.ipynb)**: Complete DP-SGD with accounting
- **[API Reference](../api/accounting.md)**: Detailed function documentation
- **[Noise Addition](noise.md)**: How noise and accounting work together

---

**Next**: Explore [Optimizers & Adaptive Clipping](optimizers.md) for better utility
