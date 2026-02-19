# Privacy Accounting

Privacy accounting tracks how much privacy budget is consumed during DP
training. Opaque uses Privacy Loss Distributions (PLD) for tight composition
bounds.

## Quick Start

```python
import opaque.accounting as acc

step = acc.poisson(acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

## Core Concepts

### DpProcess

Every mechanism constructor returns a `DpProcess`. Composition operators
produce new `DpProcess` instances. Privacy metrics are computed on demand
from the underlying PLD.

```python
import opaque.accounting as acc

# Mechanism constructors
g = acc.gaussian(0.8)
p = acc.poisson(acc.gaussian(0.8), 0.01)
tp = acc.truncated_poisson(
    acc.gaussian(0.8), 0.01, batch_size_cap=100, dataset_size=10000
)

# Composition
training = p * 1000
combined = g | acc.eps_delta(0.5)

# Privacy metrics (all derived from the same PLD)
eps = training.epsilon_at(1e-5)
delta = training.delta_at(1.0)
adv = training.advantage()
beta = training.beta_at(0.05)
risk = training.risk_at(0.5)
```

### Composition Operators

| Operator     | Description                     | Equivalent             |
|--------------|---------------------------------|------------------------|
| `proc * k`   | Repeat k times (homogeneous)   | `acc.repeat(proc, k)`  |
| `a \| b`     | Compose two processes           | `acc.compose(a, b)`    |

```python
step = acc.poisson(acc.gaussian(0.5), 0.01)

# Homogeneous
training = step * 1000

# Heterogeneous (multi-phase)
warmup = acc.poisson(acc.gaussian(0.3), 0.01) * 100
main = acc.poisson(acc.gaussian(0.5), 0.01) * 900
total = warmup | main
```

## Mechanisms

### `acc.poisson()` -- Standard DP-SGD

Poisson-subsampled Gaussian mechanism. Each example is included independently
with probability `sample_rate`. This provides privacy amplification through
subsampling.

```python
step = acc.poisson(acc.gaussian(0.8), sample_rate=256 / 50_000)
training = step * 1000
eps = training.epsilon_at(1e-5)
```

### `acc.truncated_poisson()` -- Bounded Batch Size

Tighter privacy bounds than standard Poisson by capping the batch size.
Gives up to 20% improvement in epsilon.

```python
n = 50_000
batch = 256
step = acc.truncated_poisson(
    acc.gaussian(0.8), batch / n, batch_size_cap=batch, dataset_size=n
)
training = step * 1000
```

### `acc.gaussian()` -- No Sampling

Gaussian mechanism without subsampling. Rarely used directly since DP-SGD
almost always uses Poisson sampling.

```python
proc = acc.gaussian(0.5)
eps = proc.epsilon_at(1e-5)
```

### `acc.accumulate()` -- Gradient Accumulation

Microbatching: process gradients in sub-batches, accumulate clipped gradients,
add noise once.

```python
step = acc.accumulate(
    acc.poisson(acc.gaussian(0.5), 0.01),
    microbatches=4,
)
```

### `acc.adaclip()` -- Adaptive Clipping

Accounts for the extra privacy cost of noisy quantile estimation (Andrew et
al. 2021). Returns a Gaussian with reduced effective noise multiplier.

```python
step = acc.poisson(
    acc.adaclip(acc.gaussian(0.5), quantile_noise_std=50.0),
    sample_rate=0.01,
)
```

### `acc.eps_delta()` -- Fixed Guarantee

Compose an external mechanism with known (epsilon, delta) into the privacy
process tree.

```python
external = acc.eps_delta(1.0, delta=1e-5)
total = external | (acc.poisson(acc.gaussian(0.5), 0.01) * 1000)
```

## Privacy Metrics

All metrics are computed from the same PLD. No redundant computation.

| Method              | Metric                        |
|---------------------|-------------------------------|
| `epsilon_at(delta)` | (epsilon, delta)-DP           |
| `delta_at(epsilon)` | delta for given epsilon        |
| `advantage()`       | f-DP total-variation advantage |
| `beta_at(alpha)`    | Type-II error at given alpha   |
| `risk_at(prior)`    | Bayes risk                     |

```python
proc = acc.poisson(acc.gaussian(0.5), 0.01) * 1000
eps = proc.epsilon_at(1e-5)
adv = proc.advantage()
beta = proc.beta_at(alpha=0.01)
```

## Accountant

The `Accountant` class tracks accumulated privacy loss across a training loop.
Each compose operation returns a new `Accountant` (the original is unchanged).

```python
import opaque.accounting as acc

step = acc.poisson(acc.gaussian(0.5), 0.01)
acct = acc.Accountant()

for i in range(num_steps):
    acct = acct | step
    if i % 100 == 0:
        eps = acct.epsilon_at(1e-5)
        print(f"Step {i}: eps={eps:.2f}")
```

### Budget Tracking

Pass a calibration target to enable automatic budget checking:

```python
from opaque.accounting import calibration as cal

budget = cal.epsilon_budget(3.0, delta=1e-5)
acct = acc.Accountant(budget=budget)
step = acc.poisson(acc.gaussian(0.5), 0.01)

for i in range(num_steps):
    acct = acct | step
    if acct.budget_exceeded:
        print(f"Budget exhausted at step {i}")
        break
```

### Multi-Phase Training

Compose different phases in a single accounting session:

```python
acct = acc.Accountant()

warmup_step = acc.poisson(acc.gaussian(0.3), 0.01)
for _ in range(100):
    acct = acct | warmup_step

main_step = acc.poisson(acc.gaussian(0.5), 0.01)
for _ in range(900):
    acct = acct | main_step

eps = acct.epsilon_at(1e-5)
```

## Calibration

Find the noise multiplier (or any other parameter) that achieves a target
privacy budget:

```python
from opaque.accounting import calibration as cal

result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
    param_min=0.1,
    param_max=5.0,
)
noise_multiplier = result.param
```

Calibrate sample rate instead:

```python
result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda q: acc.poisson(acc.gaussian(0.5), sample_rate=q) * 1000,
    param_min=1e-4,
    param_max=0.1,
)
```

### Target Types

| Factory                          | Metric                  | Decreasing with noise |
|----------------------------------|-------------------------|-----------------------|
| `cal.epsilon_budget(eps, delta)` | epsilon at given delta  | Yes                   |
| `cal.delta_budget(delta, eps)`   | delta at given epsilon  | Yes                   |
| `cal.advantage_budget(adv)`      | f-DP advantage          | Yes                   |
| `cal.beta_budget(beta, alpha)`   | Type-II error           | No                    |
| `cal.risk_budget(risk, prior)`   | Bayes risk              | No                    |

"Decreasing with noise" indicates whether the metric decreases as the
calibrated parameter (typically noise multiplier) increases. The binary search
direction adapts automatically.

## Privacy Amplification Through Sampling

Subsampling amplifies privacy -- the same noise gives stronger guarantees:

```python
eps_full = acc.gaussian(0.5).epsilon_at(1e-5)
eps_sampled = acc.poisson(acc.gaussian(0.5), 0.01).epsilon_at(1e-5)
print(f"Full batch: eps={eps_full:.2f}")
print(f"Sampled:    eps={eps_sampled:.4f}")
```

Lower sample rates provide more amplification but may require more training
steps.

## Custom Precision

Override default PLD discretization for faster or more precise computation:

```python
cfg = acc.DiscretizationConfig(discretization=1e-3)  # faster, coarser
proc = acc.gaussian(0.5, discretization=cfg)
```

| Parameter                   | Default      | Description                          |
|-----------------------------|--------------|--------------------------------------|
| `discretization`            | `1e-4`       | Grid spacing. Error scales as O(d^2) |
| `log_mass_truncation_bound` | `-32.0`      | Tails below 2^bound are truncated    |
| `pessimistic_estimate`      | `True`       | Upper-bound rounding (safe)          |
| `max_grid_size`             | `10_000_000` | Auto-coarsen if grid exceeds this    |

## See Also

- [API Reference: Accounting](../api/accounting.md)
- [Tutorial 02: Noise and Accounting](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)
- [Tutorial 03: Complete DP-SGD](../tutorials/03_complete_dp_sgd_training.ipynb)
- [Noise Addition](noise.md)
