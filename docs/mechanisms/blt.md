# BLT Mechanism

BLT (Buffered Linear Toeplitz) is a correlated noise mechanism optimized
for **long training runs** and **multi-epoch training**. It uses a parametric
representation of the Toeplitz coefficients via exponential decay buffers,
giving $O(\text{buffers})$ memory regardless of training length. Unlike
[BandMF](band-mf.md), BLT natively supports multi-participation patterns
where each user's data is seen multiple times.

## Idea

BandMF stores explicit Toeplitz coefficients and is limited by the `bands`
parameter. For very long training runs ($n > 5000$), you would need many
bands for good noise reduction, consuming proportional memory.

BLT solves this by **parametrizing** the Toeplitz coefficients through a
small number of exponential decay buffers. Each buffer $i$ has a decay
factor $\theta_i$ and a scale factor $\omega_i$. The effective coefficient
at lag $k$ is:

$$c_k = \sum_i \omega_i \cdot \theta_i^k$$

This sum of exponentials can represent long-range correlations with just a
few buffers (typically 3–10), regardless of $n$.

**Multi-epoch support**: When training for multiple epochs, each example
participates multiple times. BLT's sensitivity computation accounts for
this via `min_sep` (minimum steps between participations) and
`max_participations`, giving tight privacy bounds for multi-epoch training.

**When to use**: Long training runs ($n > 5000$) where BandMF's banded
approach would require too many bands, or multi-epoch training where the
sensitivity analysis needs to account for repeated participation.

## Mathematics

### BLT parametrization

The encoder is a lower-triangular Toeplitz matrix with coefficients
derived from $B$ exponential buffers:

$$C_{t,s} = \sum_{i=1}^{B} \omega_i \cdot \theta_i^{t-s} \quad \text{for } t \geq s$$

The parameters $\{(\omega_i, \theta_i)\}_{i=1}^{B}$ are optimized to
minimize workload error (max or mean MSE on prefix sums).

### Sensitivity

The sensitivity depends on the participation pattern.

**Single participation** ($\text{min\_sep}=1$, $\text{max\_participations}=1$):

Sensitivity squared is computed from the BLT parameters directly:

$$S^2 = 1 + \sum_{i,j} \omega_i \cdot \omega_j \cdot G(\theta_i \cdot \theta_j, n-1)$$

where $G(\rho, n) = \sum_{k=1}^{n} \rho^k = \rho \cdot \frac{1 - \rho^n}{1 - \rho}$ is the geometric sum.

For $n \to \infty$:

$$S^2 = 1 + \sum_{i,j} \frac{\omega_i \cdot \omega_j}{1 - \theta_i \cdot \theta_j}$$

**Min-sep participation** ($\text{min\_sep} > 1$):

When examples can participate multiple times with at least `min_sep` steps
between participations, sensitivity uses the Toeplitz min-sep algorithm
(Theorem 2, BSR paper). The algorithm:

1. Pads coefficients to min-sep blocks
2. Computes cumulative sums within blocks
3. Uses sliding-window subtraction to find the worst-case participation

**Fixed-epoch participation** is not natively supported by BLT.

### Privacy analysis

The entire training run reduces to a single Gaussian mechanism with
effective noise multiplier:

$$\sigma_{\text{eff}} = \frac{\sigma}{S}$$

where $S$ is the sensitivity. The PLD is a single Gaussian PLD.

## Assumptions and limitations

- BLT targets long runs via a **buffered** Toeplitz parameterization; privacy is for the optimized strategy you instantiate.
- Optional **`lr_schedule`** is encoded like BandMF into a Toeplitz workload for the optimizer; see [BandMF — Assumptions](band-mf.md#assumptions-and-limitations) for the constant- versus variable-\(\eta\) caveat.
- **Subsampling**: BLT does not use `cyclic_poisson` the way BandMF does; combine with Balls-in-Bins when using correlated MF + epoch structure (see examples).
- Overview: [Correlated noise (DP-FTRL)](../user-guide/dp-ftrl.md).

## Supported amplifications

BLT handles multi-participation patterns **internally** via the
sensitivity computation. There is no external amplification wrapper.

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | No | Not applicable |
| `poisson()` (truncated) | No | Not applicable |
| `cyclic_poisson()` | No | For BandMF only |

If you need subsampling amplification with correlated noise, use
[BandMF](band-mf.md) with `cyclic_poisson()` instead.

!!! note "Multi-epoch vs subsampling"
    BLT and Poisson subsampling solve different problems. Poisson subsampling
    amplifies privacy by using a random subset at each step. BLT handles the
    privacy cost of the same example appearing in multiple steps via its
    multi-participation (min-sep / max-participations) sensitivity computation.
    BLT itself does **not** model subsampling amplification and has no
    `sample_rate` parameter. If you need subsampling with correlated noise,
    use [BandMF](band-mf.md) with `cyclic_poisson()` instead.

## Code examples

### Noise injection

```python
from opaque.dpftrl.noise import mf_noise, blt_strategy
from opaque.random import key

# Single participation
strategy = blt_strategy(n_steps=10000, min_sep=1, max_buffers=10)
noise_fn, noise_state = mf_noise(
    grad_template=params,
    strategy=strategy,
    noise_multiplier=noise_multiplier,
    key=key(42),
)

for step in range(10000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
```

```python
# Multi-epoch: each user participates up to 5 times, ≥100 steps apart
strategy = blt_strategy(
    n_steps=5000, min_sep=100, max_participations=5,
)
noise_fn, noise_state = mf_noise(
    grad_template=params,
    strategy=strategy,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### Privacy accounting

The accounting constructor receives `sensitivity` and `gram_matrix` from the
same `blt_strategy` used for noise generation:

```python
import opaque.accounting as acc           # cross-cutting balls_in_bins
import opaque.dpftrl.accounting as ftrl_acc  # DP-FTRL factories
from opaque.dpftrl.noise import blt_strategy

strategy = blt_strategy(
    n_steps=5000, min_sep=100, max_participations=5,
)

# Unamplified BLT
proc = ftrl_acc.blt(1.0, sensitivity=strategy.sensitivity)
eps = proc.epsilon_at(delta=1e-5)

# With Balls-in-Bins amplification (recommended)
proc = ftrl_acc.balls_in_bins(
    ftrl_acc.blt(1.0, sensitivity=strategy.sensitivity,
                 gram_matrix=strategy.gram_matrix),
    num_bins=100, num_epochs=5,
)
eps = proc.epsilon_at(delta=1e-5)
```

!!! note
    Always use `strategy.sensitivity` and `strategy.gram_matrix` rather than
    hardcoded values. The strategy computes these from the optimized BLT
    parameters and the participation pattern.

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `n_steps` | Must be known in advance | Total training iterations. |
| `min_sep` | 1 – `n_steps` (default 1) | Minimum steps between participations. Higher = lower sensitivity. |
| `max_participations` | 1 – $\lceil n / \text{min\_sep} \rceil$ (default 1) | Number of times each user's data is used. |
| `error` | `"max"` or `"mean"` | Error metric to optimize. `"max"` is conservative. |
| `max_buffers` | 1 – 20 (default 10) | Number of exponential decay buffers. More = better noise reduction. |

**Tips**:

- **`max_buffers` = 10** is a good default. Going above 10 rarely helps.
- **`min_sep`** should reflect the actual minimum number of steps between
  a given example's appearances. For standard epoch training with batch
  size $B$ and dataset size $N$: `min_sep = N // B` (steps per epoch).
- **`max_participations`** is the number of epochs. Set it to the actual
  number of epochs in your training run.
- **`error = "max"`** is the safe choice for worst-case guarantees.
  `"mean"` optimizes average error, which may be better for practical
  accuracy but gives worse worst-case behavior.
- BLT requires knowing `n_steps`, `min_sep`, and `max_participations`
  before training starts.
- For single-participation training (1 epoch), BLT and BandMF give
  similar results. BLT's advantage is primarily in multi-epoch settings.

## References

- **Choquette-Choo et al. (2024)** — [Optimal Matrix-Factorization Mechanisms with Applications to Group Privacy](https://arxiv.org/abs/2404.16706).
  BLT mechanism with buffered linear Toeplitz optimization.
- **Choquette-Choo et al. (2024)** — [Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning](https://arxiv.org/abs/2408.08868).
  Multi-epoch sensitivity analysis for BLT.
- **Dvijotham et al. (2024)** — [Efficient and Near-Optimal Noise Generation for Streaming Differential Privacy](https://arxiv.org/abs/2405.13763).
  BSR paper with Toeplitz min-sep sensitivity theorem.
- **Kairouz et al. (2021)** — [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039).
  DP-FTRL framework.
