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
subsampling. Accepts `gaussian()` or `adaclip()` as the inner mechanism.

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

### `acc.adaclip(inner, *, fraction_noise_std, expected_batch_size)`

Accounts for the additional privacy cost of adaptive clipping (the noisy
quantile query). Use this when using `adaptive_clipped_grad`.

```python
expected_batch_size = sample_rate * dataset_size
step = acc.poisson(
    acc.adaclip(acc.gaussian(0.8),
                fraction_noise_std=0.05,
                expected_batch_size=expected_batch_size),
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

### Strategy-driven accounting

Each MF accounting constructor takes a `sensitivity` (and optionally a
`gram_matrix`) that describes the privacy cost of the strategy matrix.
These values are **computed by the noise strategy**, not set manually.

The workflow is:

1. Create a **noise strategy** (e.g. `band_mf_strategy()`,
   `lambda_cgd_strategy()`) with the training parameters (number of steps,
   bands, participation pattern, momentum, etc.).
2. The strategy computes `sensitivity` and `gram_matrix` internally from
   the strategy matrix C.
3. Pass `strategy.sensitivity` and `strategy.gram_matrix` to the accounting
   constructor.

This separation ensures that noise generation and privacy accounting always
agree on the mechanism parameters — the strategy is the single source of
truth.

```python
from opaque.dpftrl.noise import band_mf_strategy
import opaque.accounting as acc

# Strategy computes sensitivity and num_groups internally
strategy = band_mf_strategy(n_steps=1000, bands=10, momentum=0.95)

proc = acc.cyclic_poisson(
    acc.band_mf(1.0, sensitivity=strategy.sensitivity,
                num_groups=strategy.num_groups),
    sample_rate=0.01,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `acc.band_mf(noise_multiplier, sensitivity, num_groups=1)`

BandMF mechanism for cyclic Poisson amplification. Takes `sensitivity` and
`num_groups` from a `band_mf_strategy()`.

```python
strategy = band_mf_strategy(n_steps=1000, bands=10)
proc = acc.band_mf(1.0, sensitivity=strategy.sensitivity,
                   num_groups=strategy.num_groups)
eps = proc.epsilon_at(delta=1e-5)
```

For subsampling amplification, wrap with `cyclic_poisson` (see below).

### `acc.blt(noise_multiplier, sensitivity, gram_matrix=())`

BLT mechanism. Takes `sensitivity` and optional `gram_matrix` from a
`blt_strategy()`.

```python
strategy = blt_strategy(
    n_steps=10000, min_sep=1000, max_participations=5,
)

# Unamplified
proc = acc.blt(1.0, sensitivity=strategy.sensitivity)
eps = proc.epsilon_at(delta=1e-5)

# With Balls-in-Bins amplification
proc = acc.balls_in_bins(
    acc.blt(1.0, sensitivity=strategy.sensitivity,
            gram_matrix=strategy.gram_matrix),
    num_bins=1000, num_epochs=5,
)
```

### `acc.lambda_cgd(noise_multiplier, sensitivity, gram_matrix=())`

DP-λCGD mechanism (Kalinin et al., 2026). Takes `sensitivity` and
`gram_matrix` from a `lambda_cgd_strategy()`.

```python
strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = acc.balls_in_bins(
    acc.lambda_cgd(1.0, sensitivity=strategy.sensitivity,
                   gram_matrix=strategy.gram_matrix),
    num_bins=steps_per_epoch, num_epochs=num_epochs,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `acc.bisr(noise_multiplier, sensitivity, gram_matrix=())`

BISR mechanism (Kalinin et al., ICLR 2026). Generalises λCGD to
arbitrary bandwidth. Takes `sensitivity` and `gram_matrix` from a
`bisr_strategy()`.

```python
strategy = bisr_strategy(
    bandwidth=4, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = acc.balls_in_bins(
    acc.bisr(1.0, sensitivity=strategy.sensitivity,
             gram_matrix=strategy.gram_matrix),
    num_bins=steps_per_epoch, num_epochs=num_epochs,
)
```

### `acc.cyclic_poisson(inner, sample_rate)`

Cyclic Poisson amplification for BandMF. Decomposes the training run
into `ceil(n_steps / bands)` independent groups, each analyzed as a
Poisson-subsampled Gaussian mechanism. Only accepts `band_mf` processes.

```python
strategy = band_mf_strategy(n_steps=1000, bands=10)
proc = acc.cyclic_poisson(
    acc.band_mf(1.0, sensitivity=strategy.sensitivity,
                num_groups=strategy.num_groups),
    sample_rate=0.01,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `acc.balls_in_bins(inner, num_bins, num_epochs)`

Balls-in-Bins (random-partition) amplification. Returns the **total** privacy
cost across all epochs — do NOT compose further with `* num_epochs`.

Used with DP-λCGD, BISR, BLT (with Gram matrix), and Gaussian mechanisms.

```python
strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = acc.balls_in_bins(
    acc.lambda_cgd(1.0, sensitivity=strategy.sensitivity,
                   gram_matrix=strategy.gram_matrix),
    num_bins=steps_per_epoch, num_epochs=num_epochs,
)
eps = proc.epsilon_at(delta=1e-5)
```

### Calibrating MF noise

Calibration works the same way — create the strategy first, then build
the accounting mechanism from strategy-derived quantities. The strategy is
created once, and the calibration lambda varies only `noise_multiplier`:

```python
strategy = band_mf_strategy(n_steps=1000, bands=10)

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.cyclic_poisson(
        acc.band_mf(nm, sensitivity=strategy.sensitivity,
                    num_groups=strategy.num_groups),
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
from opaque.accounting import Accountant

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
