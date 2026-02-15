# Privacy Accounting

The `opaque_dp_accounting` module provides privacy accounting using Privacy Loss Distributions (PLD) with Connect-the-Dots discretization for the tightest known bounds.

**See also**: [Privacy Accounting User Guide](../user-guide/accounting.md)

```python
import opaque_dp_accounting as dp
```

---

## Classes

### `DpProcess`

Central class representing a differential privacy process. Constructed via module-level functions, composed via operators.

**Privacy metrics:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `epsilon_at` | `(delta: float) -> float` | Smallest epsilon for (epsilon, delta)-DP |
| `delta_at` | `(epsilon: float) -> float` | Smallest delta for (epsilon, delta)-DP |
| `advantage` | `() -> float` | f-DP advantage (= `delta_at(0)`) |
| `beta_at` | `(alpha: float) -> float` | Type-II error at given Type-I error |
| `risk_at` | `(prior: float) -> float` | Bayes risk under optimal adversary |

**Operators:**

| Operator | Description | Equivalent |
|----------|-------------|------------|
| `proc * k` | Homogeneous k-fold composition | `dp.repeat(proc, k)` |
| `k * proc` | Same (reflected multiply) | `dp.repeat(proc, k)` |
| `a \| b` | Heterogeneous composition | `dp.compose(a, b)` |

**Introspection:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `describe` | `() -> dict` | Constructor parameters |
| `pld_info` | `() -> dict` | PLD grid diagnostics + timing |
| `summary` | `(delta=1e-5, epsilon=1.0, alpha=0.05, prior=0.5) -> str` | Formatted privacy report |
| `__str__` | `() -> str` | One-line summary with epsilon |
| `__repr__` | `() -> str` | Reconstructible representation |

### `DiscretizationConfig`

Controls PLD numerical precision.

```python
cfg = dp.DiscretizationConfig(
    discretization=1e-4,              # Grid spacing (default 1e-4)
    log_mass_truncation_bound=-32.0,  # Tail truncation (default -32)
    pessimistic_estimate=True,        # Upper-bound rounding (default True)
    max_grid_size=10_000_000,         # Auto-coarsen limit (default 10M)
)
```

**Properties** (read-only): `discretization`, `log_mass_truncation_bound`, `pessimistic_estimate`, `max_grid_size`

**Operators**: `==`, `!=`, `repr()`

---

## Mechanism Functions

### `gaussian(noise_multiplier, config=None) -> DpProcess`

Gaussian mechanism with sensitivity 1. Building block for DP-SGD.

- `noise_multiplier` (float): Ratio of noise std to sensitivity
- `config` (DiscretizationConfig, optional): Custom precision

### `poisson(noise_multiplier, sample_rate, config=None) -> DpProcess`

Poisson-subsampled Gaussian. Standard DP-SGD step where `sample_rate = batch_size / dataset_size`.

- `noise_multiplier` (float): Gaussian noise std / sensitivity
- `sample_rate` (float): Poisson probability q in (0, 1]
- `config` (DiscretizationConfig, optional): Custom precision

### `truncated_poisson(noise_multiplier, sample_rate, batch_size_cap, dataset_size, config=None) -> DpProcess`

Truncated Poisson-subsampled Gaussian. Production DP-SGD with capped batch size. Provides tighter bounds than standard Poisson (up to 20% epsilon improvement).

- `noise_multiplier` (float): Gaussian noise std / sensitivity
- `sample_rate` (float): Expected sampling rate
- `batch_size_cap` (int): Maximum batch size
- `dataset_size` (int): Total dataset size
- `config` (DiscretizationConfig, optional): Custom precision

### `accumulate(noise_multiplier, sample_rate, microbatches, config=None) -> DpProcess`

Gradient accumulation with microbatching. Models multiple micro-batches accumulated before a single noise addition (Mixture-of-Gaussians framework).

- `noise_multiplier` (float): Gaussian noise std / sensitivity
- `sample_rate` (float): Per-microbatch Poisson rate
- `microbatches` (int): Number of micro-batches per step
- `config` (DiscretizationConfig, optional): Custom precision

### `adaclip(noise_multiplier, quantile_noise_std, config=None) -> DpProcess`

Adaptive clipping mechanism (Andrew et al. 2021). Accounts for both the main Gaussian mechanism and the quantile-estimation noise.

- `noise_multiplier` (float): Main mechanism noise
- `quantile_noise_std` (float): Quantile estimation noise
- `config` (DiscretizationConfig, optional): Custom precision

### `poisson_adaclip(noise_multiplier, quantile_noise_std, sample_rate, config=None) -> DpProcess`

Poisson-subsampled AdaClip Gaussian. Convenience wrapper combining `adaclip` and `poisson`.

- `noise_multiplier` (float): Main mechanism noise
- `quantile_noise_std` (float): Quantile estimation noise
- `sample_rate` (float): Poisson sampling rate
- `config` (DiscretizationConfig, optional): Custom precision

### `eps_delta(epsilon, delta=0, config=None) -> DpProcess`

Fixed (epsilon, delta)-DP mechanism. Useful for composing non-Gaussian mechanisms.

- `epsilon` (float): Privacy parameter (>= 0)
- `delta` (float): Failure probability (default 0)
- `config` (DiscretizationConfig, optional): Custom precision

### `identity(config=None) -> DpProcess`

Identity mechanism with zero privacy loss. Neutral element for composition.

---

## Composition Functions

### `repeat(process, count) -> DpProcess`

Homogeneous k-fold composition. Equivalent to `process * count`.

### `compose(left, right) -> DpProcess`

Heterogeneous two-process composition. Equivalent to `left | right`.

---

## Convenience Functions

### `compute_epsilon(noise_multiplier, sample_rate, num_steps, delta) -> float`

One-liner for DP-SGD epsilon. Equivalent to:

```python
(dp.poisson(noise_multiplier, sample_rate) * num_steps).epsilon_at(delta)
```

### `calibrate_noise(target_epsilon, target_delta, sample_rate, num_steps, ...) -> float`

Find the smallest noise multiplier achieving a target (epsilon, delta) guarantee via bisection search.

- `target_epsilon` (float): Desired maximum epsilon
- `target_delta` (float): Delta for the guarantee
- `sample_rate` (float): Poisson sampling rate
- `num_steps` (int): Number of DP-SGD steps
- `param_min` (float): Search lower bound (default 0.1)
- `param_max` (float): Search upper bound (default 1.2)
- `tolerance` (float): Bisection tolerance (default 1e-6)
- `max_iterations` (int): Maximum iterations (default 100)
