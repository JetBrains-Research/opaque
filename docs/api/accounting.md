# opaque.accounting

Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees.
Mechanism constructors return `DpProcess` objects that compose with `*` (repeat)
and `|` (heterogeneous compose). Privacy metrics are queried directly on the
resulting process.

```python
import opaque.accounting as acc

step = acc.poisson(acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000
epsilon = training.epsilon_at(1e-5)
```

The underlying implementation uses Google's PLD accounting via the
`opaque-accounting` Rust crate (PyO3 bindings).

**See also**: [Privacy Accounting User Guide](../user-guide/accounting.md)

---

## Classes

### `DpProcess`

Abstract base class for all privacy processes. Subclasses implement `pld()` to
compute the Privacy Loss Distribution on demand. Results are automatically
cached via `@lru_cache` (maxsize=8). Use `cached()` for larger cache
size (16) or as an opaque merge barrier.

**Privacy metrics:**

| Method               | Returns                                         |
|----------------------|-------------------------------------------------|
| `epsilon_at(delta)`  | Smallest epsilon achieving (epsilon, delta)-DP   |
| `delta_at(epsilon)`  | Smallest delta achieving (epsilon, delta)-DP     |
| `advantage()`        | Total-variation advantage (f-DP)                 |
| `beta_at(alpha)`     | Type-II error at given Type-I error alpha        |
| `risk_at(prior)`     | Bayes risk under optimal adversary               |

**Composition operators:**

| Operator     | Description                                  | Equivalent             |
|--------------|----------------------------------------------|------------------------|
| `proc * k`   | Homogeneous k-fold composition (repeat)      | `acc.repeat(proc, k)`  |
| `k * proc`   | Same (reflected multiply)                    | `acc.repeat(proc, k)`  |
| `a \| b`     | Heterogeneous composition                    | `acc.compose(a, b)`    |

Composition is optimized at construction time: identical steps are collapsed
via structural equality, nested repeats are flattened, and identity processes
are elided. Composing the same step in a loop produces a single `Repeated` node
with one `self_compose` call (2 FFTs), not `n` heterogeneous composes.

```python
step = acc.poisson(acc.gaussian(0.5), 0.01)

# Homogeneous composition
training = step * 1000

# Heterogeneous composition (multi-phase)
phase1 = acc.poisson(acc.gaussian(0.5), 0.01) * 500
phase2 = acc.poisson(acc.gaussian(0.3), 0.01) * 500
total = phase1 | phase2

eps = total.epsilon_at(1e-5)
```

### `DiscretizationConfig`

Controls PLD discretization precision. Configuration is applied at **query time**
when computing privacy metrics via `pld()`, not stored in process structure.

| Parameter                      | Default      | Description                                      |
|--------------------------------|--------------|--------------------------------------------------|
| `discretization`               | `1e-4`       | Grid spacing for PLD PMF. Error scales as O(d^2) |
| `log_x_mass_truncation_bound`  | `-50`        | Tails below exp(bound) are truncated             |
| `pessimistic_estimate`         | `True`       | Round upward for safe upper bounds               |
| `max_grid_size`                | `10_000_000` | Coarsen grid if it exceeds this many bins        |

**Query-time configuration (recommended):**

```python
# Create processes without config
proc = acc.poisson(acc.gaussian(0.8), 0.01)

# Apply config at query time
eps_coarse = proc.epsilon_at(1e-5, discretization=1e-3)  # faster, less accurate
eps_fine = proc.epsilon_at(1e-5, discretization=1e-5)    # slower, more accurate
```

**Module-level discretization defaults:**

Set default config for all queries when not overridden:

```python
acc.set_discretization(discretization=1e-4)  # Apply to all queries
proc = acc.poisson(acc.gaussian(0.8), 0.01)
eps = proc.epsilon_at(1e-5)  # Uses 1e-4 default
```

- `acc.set_discretization(discretization=1e-4, ...)` -- Set global default
- `acc.get_discretization()` -- Return current `DiscretizationConfig`

---

## Mechanism Functions

All mechanism constructors return a `DpProcess`. Discretization is configured
at query time via `epsilon_at(..., discretization=...)` or module-level via
`set_discretization()`.

### `gaussian(noise_multiplier) -> DpProcess`

Gaussian mechanism with noise multiplier sigma. Adds noise N(0, sigma^2) to
sensitivity-1 queries. Base mechanism for DP-SGD.

- `noise_multiplier` (float): Ratio of noise std to sensitivity. Larger = more private.

### `poisson(inner, sample_rate) -> DpProcess`

Poisson-subsampled mechanism (standard DP-SGD step). `sample_rate` is
`batch_size / dataset_size`.

- `inner` (Gaussian | TruncatedGaussian | AdaClip): Base mechanism
- `sample_rate` (float): Probability of including each example, in (0, 1]

```python
step = acc.poisson(acc.gaussian(0.5), sample_rate=256 / 50_000)
```

### `truncated_poisson(inner, sample_rate, batch_size_cap, dataset_size) -> DpProcess`

Truncated Poisson sampling with capped batch size. Gives tighter privacy bounds
than standard Poisson subsampling. Use this for production DP-SGD with a fixed
batch size limit.

- `inner` (Gaussian | AdaClip): Base mechanism (from `gaussian()` or `adaclip()`)
- `sample_rate` (float): Expected sampling rate
- `batch_size_cap` (int): Maximum batch size
- `dataset_size` (int): Total dataset size

```python
n = 50_000
batch = 256
step = acc.truncated_poisson(acc.gaussian(0.8), batch / n, batch, n)
```

### `parallel_poisson(inner, sample_rate, num_workers) -> DpProcess`

Parallel Poisson subsampling. Models independent Poisson sampling on
multiple workers, where the same example can appear on multiple devices.
Like `poisson()` and `truncated_poisson()`, this is a full wrapper.

- `inner` (Gaussian | AdaClip): Base mechanism (from `gaussian()` or `adaclip()`)
- `sample_rate` (float): Probability of including each example, in (0, 1]
- `num_workers` (int): Number of parallel workers sampling independently

```python
step = acc.parallel_poisson(
    acc.gaussian(0.5), sample_rate=0.01, num_workers=4,
)
```

### `truncated_gaussian(noise_multiplier, radius) -> DpProcess`

Bounded Gaussian mechanism — truncated variant. The density is renormalized
over `[-R*sigma, R*sigma]` (no point masses at boundaries). Tighter than
the standard Gaussian.

- `noise_multiplier` (float): Ratio of noise std to sensitivity.
- `radius` (float): Bound radius in units of sigma (R ≥ 1).

Composable with `poisson()` for subsampled accounting.

### `adaclip(inner, *, fraction_noise_std, batch_size) -> DpProcess`

Adaptive clipping (Andrew et al. 2021). Accounts for the extra privacy cost of
noisy quantile estimation using the combined sensitivity formula. Returns an
`AdaClip` process with the effective noise multiplier, composable with
`poisson()` or `truncated_poisson()`.

- `inner` (Gaussian | TruncatedGaussian): Base mechanism (from `gaussian()` or `truncated_gaussian()`)
- `fraction_noise_std` (float): Noise std on the clipping fraction. Default: 0.05.
- `batch_size` (float): Expected batch size, used to compute the absolute noise std for the quantile query.

```python
step = acc.poisson(acc.adaclip(acc.gaussian(0.5), fraction_noise_std=0.05, batch_size=256), 0.01)
```

### `eps_delta(epsilon, delta=0.0) -> DpProcess`

Fixed (epsilon, delta)-DP guarantee. Useful for composing an external mechanism
with known privacy parameters into tracked processes.

- `epsilon` (float): Privacy parameter (>= 0)
- `delta` (float): Failure probability (default 0.0)

```python
external = acc.eps_delta(3.0, 1e-5)
total = external | (acc.poisson(acc.gaussian(0.5), 0.01) * 1000)
```

### `identity() -> DpProcess`

Identity mechanism (zero privacy loss). Acts as the identity element in
composition: `identity() | a` returns `a`.

---

## Matrix Factorization Mechanisms

MF mechanisms compute the correct sensitivity of the correlated noise strategy
internally. They return a `DpProcess` that composes with all standard operators.

### `band_mf(noise_multiplier, n_steps, bands) -> DpProcess`

BandMF mechanism with banded Toeplitz strategy. Single-participation sensitivity
is computed from the optimized encoder matrix.

- `noise_multiplier` (float): Raw noise standard deviation sigma.
- `n_steps` (int): Number of training iterations.
- `bands` (int): Number of bands in the Toeplitz matrix (1 to `n_steps`).

```python
proc = acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10)
eps = proc.epsilon_at(1e-5)
```

### `blt_mf(noise_multiplier, n_steps, *, min_sep=1, max_participations=1, error="max", max_buffers=10) -> DpProcess`

BLT (Buffered Linear Toeplitz) mechanism. Supports multi-epoch participation
patterns via `min_sep` and `max_participations`.

- `noise_multiplier` (float): Raw noise standard deviation sigma.
- `n_steps` (int): Number of training iterations.
- `min_sep` (int): Minimum steps between participations (default 1).
- `max_participations` (int | None): Maximum participations per user (default 1).
- `error` (str): Error metric to optimize: `"max"` or `"mean"`.
- `max_buffers` (int): Maximum number of BLT buffers (default 10).

```python
proc = acc.blt_mf(1.0, 5000, min_sep=100, max_participations=5)
eps = proc.epsilon_at(1e-5)
```

### `dense_mf(noise_multiplier, n_steps, *, epochs=1, bands=None, equal_norm=False) -> DpProcess`

Dense MF with optimal strategy matrix. Materializes the full n x n matrix.

- `noise_multiplier` (float): Raw noise standard deviation sigma.
- `n_steps` (int): Number of training iterations.
- `epochs` (int): Number of epochs; must divide `n_steps`.
- `bands` (int | None): Optional banded constraint.
- `equal_norm` (bool): If True, optimize with equal column norm constraint.

```python
proc = acc.dense_mf(noise_multiplier=1.0, n_steps=50, epochs=2)
eps = proc.epsilon_at(1e-5)
```

### `cyclic_poisson(inner, sample_rate) -> DpProcess`

Cyclic Poisson amplification for BandMF. Decomposes the training run into
`ceil(n_steps / bands)` independent groups, each analyzed as a
Poisson-subsampled Gaussian. Only accepts `BandMf` inner processes.

- `inner` (BandMf): A BandMf process (from `band_mf()`).
- `sample_rate` (float): Poisson sampling probability per group.

```python
proc = acc.cyclic_poisson(
    acc.band_mf(1.0, 1000, 10), sample_rate=0.01,
)
eps = proc.epsilon_at(1e-5)
```

---

## Composition Functions

Functional equivalents of the `*` and `|` operators. Most users should prefer
the operator syntax.

### `repeat(process, count) -> DpProcess`

Homogeneous k-fold composition. Equivalent to `process * count`.

### `compose(left, right) -> DpProcess`

Heterogeneous two-process composition. Equivalent to `left | right`.

### `cached(process) -> DpProcess`

Increases the LRU cache size from 8 to 16 entries and acts as an opaque merge
barrier: the composition optimizer will not look through a cached node.

**Note**: All `pld()` methods are automatically cached with `maxsize=8` via
`@lru_cache`. Use `cached()` when you need:
- A larger cache (16 entries instead of 8)
- An explicit merge barrier to prevent composition optimizations

```python
# All queries automatically cached (maxsize=8)
step = acc.poisson(acc.gaussian(0.5), 0.01)
eps = step.epsilon_at(1e-5)   # Cached automatically
adv = step.advantage()         # Cache hit

# Use cached() for merge barrier or larger cache
training = acc.cached(step * 1000)
eps = training.epsilon_at(1e-5)   # Cached with maxsize=16
```

---

## Serialization

All processes implement `state_dict()` for JSON-friendly serialization.

```python
step = acc.poisson(acc.gaussian(0.5), 0.01)
state = step.state_dict()
```

---

## Accountant

The `Accountant` class tracks accumulated privacy loss across a training loop.
It provides a functional API: composing a new process returns a fresh
`Accountant` (the original is not modified).

Merge optimization is automatic. Composing the same `step` repeatedly in a loop
produces a single `Repeated` node internally.

```python
from opaque.accounting.accountant import Accountant

acct = Accountant()
step = acc.poisson(acc.gaussian(0.5), 0.01)

for i in range(num_steps):
    acct = acct | step

    if i % 100 == 0:
        eps = acct.epsilon_at(1e-5)
        print(f"Step {i}: eps={eps:.2f}")
```

### Budget tracking

Pass an optional `Budget` from the calibration module to enable budget checking:

```python
from opaque.accounting import calibration as cal
from opaque.accounting.accountant import Accountant

budget = cal.epsilon_budget(3.0, delta=1e-5)
acct = Accountant(budget=budget)
step = acc.poisson(acc.gaussian(0.5), 0.01)

for i in range(num_steps):
    acct = acct | step
    if acct.budget_exceeded:
        print("Privacy budget exhausted.")
        break
```

**Methods:** `epsilon_at(delta)`, `delta_at(epsilon)`, `advantage()`,
`beta_at(alpha)`, `risk_at(prior)`, `budget_exceeded` (property).

### Serialization

```python
state = acct.state_dict()
# Save state to disk (JSON-serializable dict)...

acct = Accountant.from_state_dict(state)
# Or equivalently (torch-style alias):
acct = Accountant.load_state_dict(state)
```

`from_state_dict` restores the accumulated process tree but not the budget.
Reattach a budget after loading if needed.

---

## Calibration

Submodule: `opaque.accounting.calibration`

```python
from opaque.accounting import calibration as cal
```

Binary search for finding parameter values that achieve a target privacy budget.

### `calibrate(budget, process, param_min, param_max, tolerance=1e-6, max_iterations=100) -> CalibrateResult`

Binary search for a parameter value such that `process(param)` produces a
`DpProcess` achieving the given privacy budget.

| Parameter        | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `budget`         |         | A `Budget` object from a budget factory (see below)      |
| `process`        |         | Callable: `float -> DpProcess`                           |
| `param_min`      |         | Lower bound for search                                   |
| `param_max`      |         | Upper bound for search                                   |
| `tolerance`      | `1e-6`  | Convergence threshold on `abs(achieved - target)`        |
| `max_iterations` | `100`   | Maximum binary search iterations                         |

The `process` callable takes a single float parameter and returns a `DpProcess`.
The default parameter range is tuned for noise_multiplier search, but
`calibrate()` is general: it can calibrate any float parameter in a process
against any budget.

```python
import opaque.accounting as acc
from opaque.accounting import calibration as cal

budget = cal.epsilon_budget(3.0, delta=1e-5)
result = cal.calibrate(
    budget,
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=5.0,
)
print(f"noise_multiplier = {result.param:.4f}, epsilon = {result.achieved:.6f}")
```

Calibrating a different parameter (e.g., sample rate):

```python
result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda q: acc.poisson(acc.gaussian(0.5), sample_rate=q) * 1000,
    param_min=1e-4,
    param_max=0.1,
)
```

Multi-phase training:

```python
result = cal.calibrate(
    cal.epsilon_budget(5.0, delta=1e-5),
    lambda nm: (
        acc.poisson(acc.gaussian(nm), 0.01) * 500
        | acc.poisson(acc.gaussian(nm * 0.8), 0.01) * 500
        | acc.poisson(acc.gaussian(nm * 0.5), 0.01) * 500
    ),
    param_min=0.2,
    param_max=3.0,
    tolerance=0.01,
)
```

### `CalibrateResult`

Returned by `calibrate()`.

| Attribute   | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `param`     | `float` | Found parameter value                            |
| `achieved`  | `float` | Achieved metric value at `param`                 |
| `target`    | `float` | Target metric value                              |
| `iterations`| `int`   | Number of binary search iterations               |
| `converged` | `bool`  | Whether convergence was reached within tolerance |

### Budget Factories

Budget factories create `Budget` objects that define what privacy metric to
optimize and what value to achieve.

| Factory                             | Metric being calibrated                 | Decreasing with noise |
|-------------------------------------|-----------------------------------------|-----------------------|
| `cal.epsilon_budget(eps, delta)`    | epsilon at given delta                  | Yes                   |
| `cal.delta_budget(delta, epsilon)`  | delta at given epsilon                  | Yes                   |
| `cal.advantage_budget(advantage)`   | f-DP total-variation advantage          | Yes                   |
| `cal.beta_budget(beta, alpha)`      | Type-II error at given Type-I error     | No                    |
| `cal.risk_budget(risk, prior)`      | Bayes risk under optimal adversary      | No                    |

"Decreasing with noise" indicates whether the metric decreases as the
calibrated parameter (typically noise_multiplier) increases. The binary search
adapts direction automatically based on the budget's `decreasing` property.

```python
# (epsilon, delta)-DP
result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
    0.1, 5.0,
)

# f-DP advantage
result = cal.calibrate(
    cal.advantage_budget(0.1),
    lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)

# (alpha, beta) error rates
result = cal.calibrate(
    cal.beta_budget(0.05, alpha=0.01),
    lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)

# Bayes risk
result = cal.calibrate(
    cal.risk_budget(0.1, prior=0.5),
    lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)
```
