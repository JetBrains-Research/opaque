# Privacy Accounting

The `opaque.accounting` module provides privacy accounting using Privacy Loss Distributions (PLD) with Connect-the-Dots discretization for the tightest known bounds.

**See also**: [Privacy Accounting User Guide](../user-guide/accounting.md)

```python
import opaque.accounting as acc
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
| `proc * k` | Homogeneous k-fold composition | `acc.repeat(proc, k)` |
| `k * proc` | Same (reflected multiply) | `acc.repeat(proc, k)` |
| `a \| b` | Heterogeneous composition | `acc.compose(a, b)` |

**Introspection:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `pld_info` | `() -> dict` | PLD grid diagnostics + timing |
| `summary` | `(delta=1e-5, epsilon=1.0, alpha=0.05, prior=0.5) -> str` | Formatted privacy report |
| `__str__` | `() -> str` | One-line summary with epsilon |
| `__repr__` | `() -> str` | Reconstructible representation |

### `DiscretizationConfig`

Controls PLD numerical precision.

```python
cfg = acc.DiscretizationConfig(
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

### `gaussian(noise_multiplier, discretization=None) -> Gaussian`

Gaussian mechanism with sensitivity 1. Building block for DP-SGD.

- `noise_multiplier` (float): Ratio of noise std to sensitivity
- `discretization` (DiscretizationConfig, optional): Custom precision

### `poisson(inner, sample_rate) -> Poisson`

Poisson-subsampled mechanism. Standard DP-SGD step where `sample_rate = batch_size / dataset_size`.

- `inner` (Gaussian | AdaClip): Base mechanism (from `gaussian()` or `adaclip()`)
- `sample_rate` (float): Poisson probability q in (0, 1]

### `truncated_poisson(inner, sample_rate, batch_size_cap, dataset_size) -> TruncatedPoisson`

Truncated Poisson-subsampled mechanism. Production DP-SGD with capped batch size. Provides tighter bounds than standard Poisson (up to 20% epsilon improvement).

- `inner` (Gaussian | AdaClip): Base mechanism (from `gaussian()` or `adaclip()`)
- `sample_rate` (float): Expected sampling rate
- `batch_size_cap` (int): Maximum batch size
- `dataset_size` (int): Total dataset size

### `accumulate(inner, microbatches) -> Accumulated`

Gradient accumulation with microbatching. Models multiple micro-batches accumulated before a single noise addition.

- `inner` (Poisson): Poisson-subsampled process (from `poisson()`)
- `microbatches` (int): Number of micro-batches per step

### `adaclip(inner, quantile_noise_std) -> AdaClip`

Adaptive clipping mechanism (Andrew et al. 2021). Accounts for both the main Gaussian mechanism and the quantile-estimation noise.

- `inner` (Gaussian): Base Gaussian mechanism (from `gaussian()`)
- `quantile_noise_std` (float): Quantile estimation noise

### `eps_delta(epsilon, delta=0, discretization=None) -> EpsDelta`

Fixed (epsilon, delta)-DP mechanism. Useful for composing non-Gaussian mechanisms.

- `epsilon` (float): Privacy parameter (>= 0)
- `delta` (float): Failure probability (default 0)
- `discretization` (DiscretizationConfig, optional): Custom precision

### `identity(discretization=None) -> Identity`

Identity mechanism with zero privacy loss. Neutral element for composition.

---

## Composition Functions

### `repeat(process, count) -> DpProcess`

Homogeneous k-fold composition. Equivalent to `process * count`.

### `compose(left, right) -> DpProcess`

Heterogeneous two-process composition. Equivalent to `left | right`.

---

## Convenience Functions

### `calibrate(target, build, param_min, param_max, ...) -> CalibrateResult`

Find the smallest noise multiplier achieving a target privacy guarantee via bisection search.

- `target`: Calibration target (e.g., `acc.epsilon(3.0, delta=1e-5)`)
- `build` (callable): Maps noise_multiplier → `DpProcess`
- `param_min` (float): Search lower bound
- `param_max` (float): Search upper bound
- `tolerance` (float): Bisection tolerance (default 1e-6)
- `max_iterations` (int): Maximum iterations (default 100)

Returns `CalibrateResult` with `.param` (noise multiplier) and `.achieved` (actual metric value).

Example:

```python\ndef build(nm):\n    return acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000", "oldString": "```python\ndef build(nm):\n    return acc.poisson(nm, sample_rate=0.01) * 1000

result = acc.calibrate(acc.epsilon(3.0, delta=1e-5), build, 0.1, 10.0)
print(f"Noise: {result.param:.3f}, ε: {result.achieved:.2f}")
```
