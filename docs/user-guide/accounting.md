# Privacy Accounting

Privacy accounting tracks how much privacy budget is consumed during training.
Opaque uses Privacy Loss Distributions (PLD) computed by a Rust engine for
numerically tight composition bounds. The API is built around composable
`DpProcess` objects that represent privacy mechanisms.

For mathematical details, supported amplifications, and parameter guidance
for each mechanism, see the [Mechanisms](../mechanisms/index.md) reference.

## Core concepts

### DpProcess

Every mechanism constructor returns a `DpProcess`. Composition operators
produce new `DpProcess` instances. Privacy metrics are computed on demand from
the underlying PLD.

```python
import opaque.accounting as acc

# Mechanism constructors return DpProcess instances
g = acc.gaussian(0.8)
p = acc.poisson(acc.gaussian(0.8), sample_rate=0.01)

# Composition produces new DpProcess instances
training = p * 1000

# Privacy metrics (all derived from the same PLD)
eps = training.epsilon_at(delta=1e-5)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
```

### Composition operators

| Operator | Description | Example |
|----------|-------------|---------|
| `proc * k` | Repeat k times (homogeneous) | `step * 1000` |
| `a \| b` | Compose two processes (heterogeneous) | `warmup \| main` |

Homogeneous composition (`*`) is used when the same mechanism is applied
repeatedly (e.g., 1000 identical DP-SGD steps). Heterogeneous composition
(`|`) is used when different mechanisms are applied in sequence (e.g., a
warmup phase with different noise followed by a main phase).

```python
# Homogeneous: same step repeated
step = acc.poisson(acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000

# Heterogeneous: different phases
warmup = acc.poisson(acc.gaussian(0.3), sample_rate=0.01) * 100
main = acc.poisson(acc.gaussian(0.5), sample_rate=0.01) * 900
total = warmup | main

eps = total.epsilon_at(delta=1e-5)
```

Composition is automatically optimized: identical processes are merged into
repeated nodes (2 FFTs instead of N), and identity processes are elided.

## Mechanisms

### `acc.gaussian(noise_multiplier)`

Gaussian mechanism without subsampling. Rarely used directly since DP-SGD
typically uses Poisson sampling.

```python
g = acc.gaussian(0.8)
eps = g.epsilon_at(delta=1e-5)
```

### `acc.poisson(inner, sample_rate)`

Standard Poisson-subsampled mechanism. Each example is included independently
with probability `sample_rate`. This provides privacy amplification through
subsampling. Accepts `gaussian()`,
`truncated_gaussian()`, or `adaclip()` as the inner mechanism.

```python
step = acc.poisson(acc.gaussian(0.8), sample_rate=256 / 50_000)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

### `acc.truncated_poisson(inner, sample_rate, batch_size_cap, dataset_size)`

Caps the maximum batch size to limit memory consumption, at the cost of
slightly worse privacy bounds compared to standard Poisson (the truncation
introduces additional privacy cost).

```python
n = 50_000
batch = 256
step = acc.truncated_poisson(
    acc.gaussian(0.8), batch / n,
    batch_size_cap=batch, dataset_size=n,
)
```

### `acc.parallel_poisson(inner, sample_rate, num_workers)`

Accounts for Poisson sampling under parallel worker execution. Like
`poisson()` and `truncated_poisson()`, this is a full wrapper: pass the
inner Gaussian mechanism and sample rate directly.

```python
step = acc.parallel_poisson(
    acc.gaussian(0.8), sample_rate=0.01, num_workers=4,
)
```

### `acc.truncated_gaussian(noise_multiplier, radius)`

Bounded Gaussian mechanism — truncated variant. The density is renormalized
over `[-R*sigma, R*sigma]` (no point masses at boundaries). Tighter than
`acc.gaussian()` because the bounded support limits worst-case hockey-stick
divergence.

Use this when adding noise via `truncated_gaussian_noise()`.
Composable with `poisson()` for subsampled accounting.

```python
step = acc.poisson(acc.truncated_gaussian(1.1, radius=5.0), sample_rate=0.01)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)  # tighter than acc.gaussian(1.1)
```

The truncated variant gives tighter ε than `acc.gaussian()`. For most
workloads, prefer truncated when bounded noise is desired.

### `acc.adaclip(inner, *, fraction_noise_std, batch_size)`

Accounts for the additional privacy cost of adaptive clipping (the noisy
quantile query). Use this when using `adaptive_clipped_grad`.

```python
step = acc.poisson(
    acc.adaclip(acc.gaussian(0.8),
                fraction_noise_std=0.05,
                batch_size=256),
    sample_rate=0.01,
)
```

### `acc.eps_delta(epsilon, delta=0.0)`

A fixed (epsilon, delta)-DP mechanism. Useful for composing external privacy
costs (e.g., a hyperparameter tuning step with known privacy cost).

```python
external_cost = acc.eps_delta(1.0, delta=1e-6)
total = external_cost | (step * 1000)
```

### `acc.identity()`

Zero privacy cost. Identity element for composition. Useful as an initial
value when building up a process programmatically.

```python
process = acc.identity()
for step_proc in step_list:
    process = process | step_proc
eps = process.epsilon_at(delta=1e-5)
```

## Matrix factorization mechanisms

MF noise introduces correlations between training steps, reducing
effective noise on cumulative updates. The accounting must handle the
modified sensitivity of the strategy matrix. These mechanisms optimize
the strategy internally and compute the correct sensitivity — do not use
`acc.gaussian()` for MF noise.

### `acc.band_mf(noise_multiplier, n_steps, bands)`

BandMF mechanism with banded Toeplitz strategy. Computes single-participation
sensitivity from the optimized encoder.

```python
proc = acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10)
eps = proc.epsilon_at(delta=1e-5)
```

For subsampling amplification, wrap with `cyclic_poisson` (see below).

### `acc.blt_mf(noise_multiplier, n_steps, *, min_sep, max_participations)`

BLT mechanism with Buffered Linear Toeplitz strategy. Supports
multi-epoch training via `min_sep` and `max_participations`.

```python
# Single participation
proc = acc.blt_mf(noise_multiplier=1.0, n_steps=5000)
eps = proc.epsilon_at(delta=1e-5)

# Multi-epoch: each user participates up to 5 times, at least 100 steps apart
proc = acc.blt_mf(1.0, 5000, min_sep=100, max_participations=5)
eps = proc.epsilon_at(delta=1e-5)
```

### `acc.dense_mf(noise_multiplier, n_steps, *, epochs)`

Dense MF with optimal strategy matrix. Materializes the full n x n matrix,
so use only for short training runs (n < 100).

```python
proc = acc.dense_mf(noise_multiplier=1.0, n_steps=50, epochs=2)
eps = proc.epsilon_at(delta=1e-5)
```

### `acc.cyclic_poisson(inner, sample_rate)`

Cyclic Poisson amplification for BandMF. Decomposes the training run
into `ceil(n_steps / bands)` independent groups, each analyzed as a
Poisson-subsampled Gaussian mechanism. Only accepts `band_mf` processes.

```python
proc = acc.cyclic_poisson(
    acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10),
    sample_rate=0.01,
)
eps = proc.epsilon_at(delta=1e-5)
```

### Calibrating MF noise

Calibration works the same way — pass a lambda that builds the MF process:

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.cyclic_poisson(
        acc.band_mf(nm, n_steps=1000, bands=10),
        sample_rate=0.01,
    ),
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

## Privacy metrics

All metrics are methods on `DpProcess`. They compute the underlying PLD on
demand and cache the result.

| Method | Returns | Interpretation |
|--------|---------|----------------|
| `.epsilon_at(delta)` | float | Smallest epsilon for (epsilon, delta)-DP |
| `.delta_at(epsilon)` | float | Smallest delta for (epsilon, delta)-DP |
| `.advantage()` | float | f-DP total-variation advantage (0 = perfect privacy) |
| `.beta_at(alpha)` | float | Type-II error at Type-I error alpha (higher = more private) |
| `.risk_at(prior)` | float | Bayes risk under optimal adversary (higher = more private) |

```python
training = acc.poisson(acc.gaussian(1.1), sample_rate=0.01) * 1000

eps = training.epsilon_at(delta=1e-5)
delta = training.delta_at(epsilon=3.0)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
risk = training.risk_at(prior=0.5)
```

See [DP Concepts](dp-concepts.md#privacy-metrics) for the meaning of each
metric.

## Calibration

Calibration finds the parameter value (typically noise multiplier) that
achieves a target privacy budget. It uses binary search over the parameter
space.

### Basic calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

The `process` argument is a lambda that takes the parameter being calibrated
and returns a `DpProcess`. `calibrate` binary-searches for the parameter
value where the budget metric equals the target.

### Budget types

| Function | Target metric | Direction |
|----------|---------------|-----------|
| `acc.epsilon_budget(eps, delta)` | epsilon | decreasing (more noise = lower epsilon) |
| `acc.delta_budget(delta, epsilon)` | delta | decreasing |
| `acc.advantage_budget(adv)` | advantage | decreasing |
| `acc.beta_budget(beta, alpha)` | beta | increasing (more noise = higher beta) |
| `acc.risk_budget(risk, prior)` | risk | increasing |

### Calibrating other parameters

`calibrate` works with any float parameter, not just noise multiplier. For
example, calibrating the sample rate:

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda sr: acc.poisson(acc.gaussian(1.0), sample_rate=sr) * 1000,
    param_min=0.001,
    param_max=0.1,
)
sample_rate = result.param
```

## Accountant

The `Accountant` class provides step-by-step privacy tracking during training.
It wraps a `DpProcess` and provides budget checking.

```python
from opaque.accounting.accountant import Accountant

acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)

for batch in dataloader:
    # ... train ...
    acct = acct | step
    if acct.budget_exceeded:
        print("Privacy budget exhausted")
        break

eps = acct.epsilon_at(delta=1e-5)
```

`Accountant` is functional: `acct | step` returns a new `Accountant` without
mutating the original. The `budget_exceeded` property checks whether the
accumulated process exceeds the budget.

### Serialization

```python
state = acct.state_dict()
# Save state to disk...

acct = Accountant.from_state_dict(state)
```

## Discretization

PLD computation uses a discretized grid. The default parameters are suitable
for most use cases. For tighter bounds at the cost of computation, adjust the
discretization:

```python
from opaque.accounting import set_discretization

# Tighter (slower)
set_discretization(discretization=1e-5, max_grid_size=50_000_000)

# Faster (looser)
set_discretization(discretization=1e-3)
```

Parameters can also be overridden per query:

```python
eps = training.epsilon_at(delta=1e-5, discretization=1e-5)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `discretization` | 1e-4 | Grid spacing. Smaller = tighter, slower. |
| `log_x_mass_truncation_bound` | -50.0 | Log tail mass cutoff. |
| `pessimistic_estimate` | True | Round upward for safe guarantees. |
| `max_grid_size` | 10,000,000 | Maximum grid bins before coarsening. |

## API reference

See [Accounting API Reference](../api/accounting.md) for complete function
signatures and return types.
