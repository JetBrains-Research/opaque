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
import opaque.accounting as acc

# Build a DP-SGD process: Poisson-subsampled Gaussian, 1000 steps
step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
training = step * 1000

# Query privacy
eps = training.epsilon_at(delta=1e-5)
print(f"Privacy spent: epsilon={eps:.2f}")
```

## Core Concepts

### Everything is a `DpProcess`

Every mechanism constructor returns a `DpProcess`, and composition operators produce new `DpProcess` instances. Privacy metrics are computed on demand from the underlying PLD.

```python
import opaque.accounting as acc

# Mechanisms
g = acc.gaussian(1.1)                                # Gaussian mechanism
p = acc.poisson(acc.gaussian(1.1), 0.01)              # Poisson-subsampled Gaussian
tp = acc.truncated_poisson(acc.gaussian(1.1), 0.01,   # Truncated Poisson (production DP-SGD)
         batch_size_cap=100, dataset_size=10000)

# Composition
training = p * 1000                      # repeat 1000 times
combined = g | acc.eps_delta(0.5)        # compose two different processes

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

### `acc.poisson()` — Standard DP-SGD

**Use when**: Training with Poisson sampling (batch size ~ sample_rate x dataset_size)

```python
step = acc.poisson(
    acc.gaussian(1.2),
    sample_rate=32 / 10000,         # batch_size / dataset_size
)
training = step * 1000              # 1000 training steps
eps = training.epsilon_at(1e-5)
```

**Why Poisson?** Each example is sampled independently with probability `sample_rate`, providing **privacy amplification through subsampling**. With `sample_rate=0.01` the effective epsilon can be 50-100x smaller than the un-subsampled Gaussian.

### `acc.truncated_poisson()` — Production DP-SGD

**Use when**: You want tight privacy bounds with bounded batch sizes

```python
step = acc.truncated_poisson(
    acc.gaussian(1.2),
    sample_rate=32 / 10000,
    batch_size_cap=32,              # maximum batch size
    dataset_size=10000,
)
training = step * 1000
eps = training.epsilon_at(1e-5)
```

**Advantage**: Tighter privacy bounds than standard Poisson (up to 20% improvement). This matches what production DP-SGD frameworks (Opacus, JAX Privacy, TF Privacy) actually do.

**When to use**: Always, unless you have a specific reason not to.

### `acc.gaussian()` — No Sampling

**Use when**: Processing entire dataset (no subsampling)

```python
proc = acc.gaussian(noise_multiplier=1.2)
eps = proc.epsilon_at(1e-5)
```

**Rarely used** in practice since DP-SGD almost always uses sampling.

### `acc.accumulate()` — Gradient Accumulation

**Use when**: Memory-limited training with microbatching

```python
step = acc.accumulate(
    acc.poisson(acc.gaussian(1.1), 0.01),
    microbatches=4,                 # 4 microbatches per noise step
)
training = step * 500
eps = training.epsilon_at(1e-5)
```

### `acc.adaclip()` — Adaptive Clipping

**Use when**: Automatically adjusting clipping threshold (Andrew et al. 2021)

```python
step = acc.adaclip(
    acc.gaussian(1.1),
    quantile_noise_std=50.0,        # noise for quantile estimation
)
eps = step.epsilon_at(1e-5)
```

### `acc.eps_delta()` — Fixed Guarantee

**Use when**: Composing a non-Gaussian mechanism with known (epsilon, delta)

```python
proc = acc.eps_delta(epsilon=1.0, delta=1e-5)
combined = acc.gaussian(1.1) | proc  # compose with a Gaussian
```

### `acc.identity()` — Zero Privacy Loss

**Use when**: Representing a no-op in composition chains

```python
proc = acc.identity()
assert proc.epsilon_at(1e-5) < 1e-10
```

## Composition

### Homogeneous Composition (Repetition)

Use `*` or `acc.repeat()` for k-fold composition of the same mechanism:

```python
step = acc.poisson(acc.gaussian(1.1), 0.01)

# These are equivalent
training = step * 1000
training = acc.repeat(step, 1000)
```

### Heterogeneous Composition

Use `|` or `acc.compose()` to compose different mechanisms:

```python
a = acc.gaussian(1.0)
b = acc.gaussian(0.8)

# These are equivalent
combined = a | b
combined = acc.compose(a, b)

assert combined.epsilon_at(1e-5) > a.epsilon_at(1e-5)
```

## Privacy Metrics

All metrics are computed from the same Privacy Loss Distribution — no redundant computation.

### (epsilon, delta)-Differential Privacy

The **standard metric** used in most DP papers:

```python
proc = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
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
import opaque.accounting as acc

sample_rate = 0.01
num_steps = 1000

# Define how training depends on the noise multiplier
def build(nm):
    return acc.poisson(acc.gaussian(nm), sample_rate) * num_steps

# Calibrate for target (epsilon, delta)
result = acc.calibrate(
    acc.epsilon(8.0, delta=1e-5),  # Target: ε=8.0 at δ=1e-5
    build,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
print(f"Use noise_multiplier={noise_multiplier:.4f}")
print(f"Achieved epsilon: {result.achieved:.4f}")

# Verify
actual_eps = build(noise_multiplier).epsilon_at(1e-5)
assert abs(actual_eps - 8.0) < 0.1
```

### Quick Epsilon Computation

For one-off privacy checks:

```python
training = acc.poisson(
    acc.gaussian(1.1),
    sample_rate=0.01,
) * 1000

eps = training.epsilon_at(1e-5)
print(f"epsilon = {eps:.4f}")
```

## Debugging and Introspection

### Quick Display

```python
proc = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
print(proc)
# Repeat(Poisson(Gaussian(noise_multiplier=1.1), sample_rate=0.01), count=1000)
```

### Process Structure

```python
proc = acc.poisson(acc.gaussian(1.1), 0.01)
print(repr(proc))
# Poisson(Gaussian(noise_multiplier=1.1), sample_rate=0.01)
```

## Custom Precision

Override default PLD discretization for faster or more precise computation:

```python
import opaque.accounting as acc

# Faster (coarser grid)
cfg = acc.DiscretizationConfig(discretization=1e-3)

# More precise (finer grid, wider tails)
cfg = acc.DiscretizationConfig(
    discretization=1e-5,
    log_mass_truncation_bound=-50.0,
)

# Use with any mechanism
proc = acc.gaussian(1.1, discretization=cfg)
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
def build(nm):
    return acc.poisson(acc.gaussian(nm), sample_rate) * num_steps

result = acc.calibrate(acc.epsilon(target_epsilon, delta=target_delta), build, 0.1, 10.0)
noise_multiplier = result.param

# Create DP gradient function and noise function
dp_grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=clip_norm, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_norm)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        grads, clip_state = dp_grad_fn(params, batch, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        params = update(params, noisy_grads)

    # Check privacy at end of each epoch
    steps_so_far = (epoch + 1) * steps_per_epoch
    current_eps = (acc.poisson(acc.gaussian(noise_multiplier), sample_rate) * steps_so_far).epsilon_at(target_delta)
    print(f"Epoch {epoch+1}: epsilon={current_eps:.2f}/{target_epsilon:.2f}")

# Verify final privacy
final_eps = build(noise_multiplier).epsilon_at(target_delta)
assert final_eps <= target_epsilon + 0.1, "Privacy budget exceeded!"
```

## Privacy Composition Basics

Privacy degrades as you train more:

```python
# Compare epsilon at different training durations
for steps in [100, 1000, 10000]:
    eps = (acc.poisson(acc.gaussian(1.2), 0.01) * steps).epsilon_at(1e-5)
    print(f"After {steps:>5} steps: epsilon={eps:.2f}")
```

!!! warning "Privacy degrades with training"
    More training steps -> higher epsilon -> weaker privacy. Plan your training budget carefully!

## Privacy Amplification Through Sampling

Subsampling **amplifies privacy** -- you get stronger guarantees for the same noise:

```python
# No sampling (full batch)
eps_full = acc.gaussian(1.0).epsilon_at(1e-5)

# With Poisson sampling (sample_rate=0.01)
eps_sampled = acc.poisson(acc.gaussian(1.0), 0.01).epsilon_at(1e-5)

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
    Don't guess noise multipliers! Use `acc.calibrate()` to find the right noise level.

### 2. Use Truncated Poisson When Possible

```python
# Tighter bounds (preferred)
step = acc.truncated_poisson(acc.gaussian(nm), rate, batch_size_cap=B, dataset_size=n)

# vs standard Poisson
step = acc.poisson(acc.gaussian(nm), rate)
```

### 3. Query Multiple Metrics

```python
proc = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
print(f"epsilon(delta=1e-5) = {proc.epsilon_at(1e-5):.2f}")
print(f"advantage           = {proc.advantage():.4f}")
print(f"beta(alpha=0.05)    = {proc.beta_at(0.05):.4f}")
```

### 4. Query Multiple Metrics

```python
proc = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
print(f"epsilon(delta=1e-5) = {proc.epsilon_at(1e-5):.2f}")
print(f"advantage           = {proc.advantage():.4f}")
print(f"beta(alpha=0.05)    = {proc.beta_at(0.05):.4f}")
```

## See Also

- **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)**: Interactive accounting tutorial
- **[Tutorial 03](../tutorials/03_complete_dp_sgd_training.ipynb)**: Complete DP-SGD with accounting
- **[API Reference](../api/accounting.md)**: Detailed function documentation
- **[Noise Addition](noise.md)**: How noise and accounting work together

---

**Next**: Explore [Optimizers & Adaptive Clipping](optimizers.md) for better utility
