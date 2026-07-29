# Privacy Accounting

Privacy accounting tracks how much privacy budget is consumed during training.
Opaque uses Privacy Loss Distributions (PLD) computed by a Rust engine for
numerically tight composition bounds. The API is built around composable
`DpProcess` objects that represent privacy mechanisms.

For mathematical details, supported amplifications, and parameter guidance
for each mechanism, see the [Mechanisms](../mechanisms/index.md) reference.

## Namespace organization

The accounting surface is split across three modules so each algorithm's
factories live next to its runtime:

| Module | Provides | Ships with |
|--------|----------|------------|
| `opaque.accounting` | Cross-cutting primitives — composition (`compose`, `repeat`, `cached`), `calibrate`, generic mechanisms (`identity`, `nonprivate`, `eps_delta`), `Accountant`, and the shared PLD / discretization stack. | `opaque-accounting` |
| `opaque.dpsgd.accounting` | DP-SGD factories — `gaussian`, `adaclip`, `poisson` (plain or truncated via `truncated_batch_size` / `dataset_size`), `parallel_poisson`. | `opaque-dpsgd` |
| `opaque.dpftrl.accounting` | DP-FTRL factories — `band_mf`, `blt`, `bisr`, `bsr`, `lambda_cgd`, `identity_mf`, `poisson` (cyclic when `bands > 1`, plain when `bands == 1`, parameterized by `n_steps`), `b_min_sep`, `balls_in_bins`. | `opaque-dpftrl` |

Private second moments do **not** use a separate accounting wrapper: the joint gradient + squared-gradient release is handled in the runtime σ split (sensitivity-proportional Mahalanobis allocation), so calibration stays on the same underlying mechanism PLD as first-moment-only training. See [Noise API](../reference/noise.md#paired-second-moment-release).

Both algorithm-specific namespaces re-export from the shared `opaque-accounting`
implementation; the split is purely organisational. The `Accountant` interactive
container is on `opaque.accounting` directly (`from opaque.accounting import
Accountant`); calibration helpers (`calibrate`, `epsilon_budget`, etc.) live
there too.

The mechanism factories themselves (`gaussian`, `poisson`, `band_mf`, …) are
**only** on the algorithm-specific namespaces. Use the namespace that matches
your training run (`opaque.dpsgd.accounting` or `opaque.dpftrl.accounting`) —
the per-step (DP-SGD) vs whole-process (DP-FTRL) distinction is part of the
import path, on purpose. The algorithm-specific subpackages are
*lazy-imported* from `opaque.dpsgd` / `opaque.dpftrl`, so the Rust PLD
extension only loads when accounting is actually used.

## Core concepts

### DpProcess

Every mechanism constructor returns a `DpProcess`. Composition operators
produce new `DpProcess` instances. Privacy metrics are computed on demand from
the underlying PLD.

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

# Mechanism constructors return DpProcess instances.  Cross-cutting
# primitives (composition, calibration) live at acc; algorithm-specific
# factories live in the per-algorithm namespace.
g = dpsgd_acc.gaussian(0.8)
p = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)

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
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000

# Heterogeneous: different phases
warmup = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.3), sample_rate=0.01) * 100
main = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), sample_rate=0.01) * 900
total = warmup | main

eps = total.epsilon_at(delta=1e-5)
```

Composition is automatically optimized: identical processes are merged into
repeated nodes (2 FFTs instead of N), and identity processes are elided.

## Mechanisms

### `dpsgd_acc.gaussian(noise_multiplier)`

Gaussian mechanism without subsampling. Rarely used directly since DP-SGD
typically uses Poisson sampling.

```python
g = dpsgd_acc.gaussian(0.8)
eps = g.epsilon_at(delta=1e-5)
```

### `dpsgd_acc.poisson(inner, sample_rate)`

Standard Poisson-subsampled mechanism. Each example is included independently
with probability `sample_rate`. This provides privacy amplification through
subsampling. Accepts `gaussian()` or `adaclip()` as the inner mechanism.

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=256 / 50_000)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

### `dpsgd_acc.poisson(inner, sample_rate, *, truncated_batch_size, dataset_size)` (truncated form)

Setting both `truncated_batch_size` and `dataset_size` (must be both or
neither) selects the truncated-Poisson PLD for a per-step batch cap. That
matches production runs with capped batches but is **no stricter** than plain
Poisson at the same `sample_rate`—typically **weaker** (higher ε at the same
noise) because truncation changes the subsampling distribution.

```python
n = 50_000
batch = 256
step = dpsgd_acc.poisson(
    dpsgd_acc.gaussian(0.8),
    sample_rate=batch / n,
    truncated_batch_size=batch,
    dataset_size=n,
)
```

### `dpsgd_acc.parallel_poisson(inner, sample_rate, num_workers)`

Accounts for Poisson sampling under parallel worker execution. Like
`poisson()` (plain or truncated), this is a full wrapper: pass the
inner Gaussian mechanism and sample rate directly.

```python
step = dpsgd_acc.parallel_poisson(
    dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4,
)
```

### `dpsgd_acc.adaclip(inner, *, fraction_noise_std, expected_batch_size)`

Accounts for the additional privacy cost of adaptive clipping (the noisy
quantile query). Use this when using `adaptive_clipped_grad`.

```python
expected_batch_size = sample_rate * dataset_size
step = dpsgd_acc.poisson(
    dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8),
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
`dpsgd_acc.gaussian()` for MF noise.

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
import opaque.dpftrl.accounting as dpftrl_acc

# Strategy is a recipe; the amplifier supplies the horizon at PLD time.
strategy = band_mf_strategy(bands=10, momentum=0.95)

proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `dpftrl_acc.mf_gaussian(noise_multiplier, strategy)`

Single MF Gaussian mechanism wrapping a strategy recipe.  The strategy
carries only static workload knobs (e.g. `bands`, `momentum`); horizon
(`n_steps`, `min_sep`, `max_participations`) is supplied by the
surrounding amplification factory at PLD time.

```python
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.mf_gaussian(1.0, strategy, n_steps=1000)  # bare use
eps = proc.epsilon_at(delta=1e-5)
```

For subsampling amplification, wrap with `dpftrl_acc.poisson(..., n_steps=...)`
(see below).

### Correlated MF mechanisms (BLT, λCGD, BISR, BSR)

Correlated MF mechanisms use the same `dpftrl_acc.mf_gaussian(noise_multiplier,
strategy)` factory — the strategy carries the static workload knobs and
the amplifier supplies the participation context.  Wrap in
`dpftrl_acc.balls_in_bins(...)` (BnB) for the full PLD:

```python
strategy = blt_strategy(
    n_steps=10000, min_sep=1000, max_participations=5,
)

# Unamplified — single-Gaussian PLD
proc = ftrl_acc.mf_gaussian(1.0, strategy)
eps = proc.epsilon_at(delta=1e-5)

# With Balls-in-Bins amplification
proc = dpftrl_acc.balls_in_bins(
    ftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=1000, n_steps=5000,
)
```

The same pattern works for `lambda_cgd_strategy`, `bisr_strategy`, and
`bsr_strategy`:

```python
strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = dpftrl_acc.balls_in_bins(
    ftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch, n_steps=steps_per_epoch * num_epochs,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `dpftrl_acc.poisson(inner, sample_rate, *, n_steps)`

Poisson amplification for DP-FTRL. Whole-process accountant covering all
`n_steps` rounds (do **not** compose with `* num_steps` externally).
Cyclic when the inner is `BandMf` with `bands > 1` (decomposes into
`ceil(n_steps / bands)` independent groups), plain Poisson per round
when the inner is `IdentityMf` or `BandMf` with `bands == 1`.

```python
strategy = band_mf_strategy(n_steps=1000, bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `dpftrl_acc.balls_in_bins(inner, num_bins, n_steps)`

Balls-in-Bins (random-partition) amplification. Returns the **total**
privacy cost across all `n_steps` rounds (must be a positive multiple of
`num_bins`; per-bin participation count is `n_steps // num_bins`). Do NOT
compose further externally.

Used with DP-λCGD, BISR, BLT (with Gram matrix), `identity_mf`, and
Gaussian mechanisms.

```python
strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = dpftrl_acc.balls_in_bins(
    ftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch,
    n_steps=steps_per_epoch * num_epochs,
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
    lambda nm: dpftrl_acc.poisson(
        dpftrl_acc.mf_gaussian(nm, strategy),
        sample_rate=0.01,
        n_steps=1000,
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
training = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01) * 1000

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
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

The `process` argument is a lambda that takes the parameter being calibrated
and returns a `DpProcess`. `calibrate` binary-searches for a privacy-safe
parameter whose metric is within the requested relative tolerance. For
privacy-loss budgets (epsilon, delta, and advantage), the returned metric is
at or below the target. For privacy-gain budgets (beta and risk), it is at or
above the target. Every successfully returned result has `converged=True`.

The `tolerance` argument must be finite and positive, and `max_iterations`
must be positive. Invalid values raise `ValueError` before the process is
evaluated. If the search cannot find a safe endpoint satisfying
`math.isclose(achieved, target, rel_tol=tolerance, abs_tol=0.0)`, it raises
`RuntimeError` instead of returning an under-noised parameter.

### Budget types

| Function | Target metric | Direction |
|----------|---------------|-----------|
| `acc.epsilon_budget(eps, delta)` | epsilon | decreasing (more noise = lower epsilon) |
| `acc.delta_budget(delta, epsilon)` | delta | decreasing |
| `acc.advantage_budget(adv)` | advantage | decreasing |
| `acc.beta_budget(beta, alpha)` | beta | increasing (more noise = higher beta) |
| `acc.risk_budget(risk, prior)` | risk | increasing |

### Search direction

The current search contract supports parameters such as noise multiplier whose
increase improves privacy in the direction declared by the selected budget:
privacy-loss metrics decrease and privacy-gain metrics increase. The metric
produced by `process(param)` must follow that declared direction, with
`param_min` as the unsafe endpoint and `param_max` as the safe endpoint.

## Accountant

The `Accountant` class provides step-by-step privacy tracking during training.
It wraps a `DpProcess` and provides budget checking.

```python
from opaque.accounting import Accountant

acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), sample_rate)

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

### DP-FTRL: step-by-step accounting via `per_step`

DP-FTRL accountants are *whole-process* (`dpftrl_acc.poisson` / `b_min_sep` /
`balls_in_bins` already cover all `n_steps` rounds), so feeding the bare
process to `acct |= step` would over-count. Wrap the process with
`dpftrl_acc.per_step(...)` to get a step-shaped adapter whose
`per_step(proc) * K` materialises the strategy-aware K-prefix PLD (rather
than the K-fold composition of a single-step PLD, which would double-count
the workload coefficients):

```python
import opaque.accounting as acc
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.accounting import Accountant
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=64)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    sample_rate=0.01,
    n_steps=15_624,
)
step = dpftrl_acc.per_step(proc)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))

for batch in dataloader:
    # ... train ...
    acct |= step
    if acct.budget_exceeded:
        break

eps = acct.epsilon_at(delta=1e-5)
```

The K-step ε is bounded above by `proc.epsilon_at(delta)` and is monotone
non-decreasing in K — both follow from the post-processing inequality on the
deployed N-step mechanism. `K > proc.n_steps` raises `ValueError`.

### Serialization

```python
from opaque.accounting import Accountant
from opaque.serialization import from_state_dict, state_dict

flat = state_dict(acct)
# ... torch.save(flat, path) / flat = torch.load(path) ...
```

The flat mapping may include ``torch.Tensor`` and NumPy arrays. Persist it with
:func:`torch.save` / :func:`torch.load` (or another pickle-compatible format),
not JSON, unless every leaf is JSON-serialisable.

For a bare ``DpProcess``, use ``from_state_dict(identity(), flat)`` (or any
concrete process as the template).

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

See [Accounting API Reference](../reference/accounting.md) for complete function
signatures and return types.
