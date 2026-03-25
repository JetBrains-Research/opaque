# Dense MF Mechanism

Dense MF computes the **optimal** strategy matrix for correlated noise by
materializing the full $n \times n$ lower-triangular matrix. This gives
the best possible noise reduction but at $O(n^2)$ memory and compute cost,
making it practical only for short training runs ($n < 100$).

## Idea

[BandMF](band-mf.md) restricts the strategy to a banded Toeplitz structure
for efficiency. [BLT](blt.md) uses a parametric sum of exponentials. Dense
MF removes all structural constraints: it optimizes over **all**
lower-triangular matrices to find the one that minimizes the workload error
(MSE on prefix sums) subject to a sensitivity bound.

The result is provably optimal — no other matrix factorization can achieve
lower error for the same privacy budget. The trade-off is that the full
$n \times n$ matrix must be stored and inverted, so memory grows
quadratically with the number of steps.

Dense MF natively supports **fixed-epoch** participation patterns, where
each user's data appears at regular epoch boundaries.

**When to use**: Short training runs ($n < 100$) where you want the absolute
best privacy-utility trade-off and are willing to pay the $O(n^2)$ cost.
For longer runs, use [BandMF](band-mf.md) or [BLT](blt.md).

## Mathematics

### Strategy matrix

The strategy matrix $C$ is an $n \times n$ lower-triangular matrix
optimized to minimize:

$$\text{error}(C) = \|A C^{-1}\|_F^2$$

where $A$ is the workload matrix (prefix sums: $A_{t,s} = 1$ if $s \leq t$,
else 0), subject to a sensitivity constraint.

The optimization finds $C$ using convex optimization over the cone of
lower-triangular matrices with bounded column norms.

### Noise generation

At each step $t$, the mechanism:

1. Draws fresh noise $z_t \sim \mathcal{N}(0, \sigma^2 I)$
2. Computes $n_t = \sum_{s \leq t} C^{-1}_{t,s} \cdot z_s$
3. Adds $n_t$ to the clipped gradient sum

Since $C^{-1}$ is dense, this requires storing all previous noise vectors —
$O(n \cdot d)$ total memory where $d$ is the parameter dimension.

### Sensitivity under fixed-epoch participation

For fixed-epoch participation with $e$ epochs and $n$ steps (where $e$
divides $n$), each user contributes at steps $\{t, t + n/e, t + 2n/e, \ldots\}$.

The sensitivity is computed via the Gram matrix $X = C^\top C$:

$$S = \max_u \sqrt{u^\top X u}$$

where the maximization is over valid participation vectors $u \in \{0, 1\}^n$
with the fixed-epoch structure. This is computed by grouping indices by epoch
and summing the absolute values of epoch-aligned submatrices.

### Privacy analysis

As with all MF mechanisms, the training run reduces to a single Gaussian
with effective noise multiplier:

$$\sigma_{\text{eff}} = \frac{\sigma}{S}$$

## Supported amplifications

Dense MF handles multi-epoch participation **internally** through the
fixed-epoch sensitivity computation. No external amplification is needed.

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | No | Not applicable |
| `truncated_poisson()` | No | Not applicable |
| `cyclic_poisson()` | No | For BandMF only |

!!! note "No subsampling amplification"
    Dense MF does not support external subsampling wrappers. The privacy
    analysis accounts for the full participation pattern through the
    sensitivity computation. If you need subsampling amplification, use
    [BandMF](band-mf.md) with `cyclic_poisson()`.

## Code examples

### Noise injection

```python
from opaque import dense_mf_noise
from opaque.random import key

noise_fn, noise_state = dense_mf_noise(
    grad_template=params,
    n_steps=50,
    stddev=noise_multiplier * clip_state.sensitivity(),
    key=key(42),
    epochs=2,
)

for step in range(50):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### Privacy accounting

```python
import opaque.accounting as acc

# Single epoch
proc = acc.dense_mf(noise_multiplier=1.0, n_steps=50)
eps = proc.cgf().epsilon_at(delta=1e-5)

# Two epochs (each user participates twice)
proc = acc.dense_mf(noise_multiplier=1.0, n_steps=50, epochs=2)
eps = proc.cgf().epsilon_at(delta=1e-5)
```

### Calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.dense_mf(nm, n_steps=50, epochs=2),
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

### Comparing strategies

```python
import opaque.accounting as acc

n = 50
nm = 1.0

# Dense (optimal) vs BandMF vs independent
dense = acc.dense_mf(nm, n_steps=n, epochs=1)
band  = acc.band_mf(nm, n_steps=n, bands=10)
gauss = acc.gaussian(nm) * n  # independent noise, n compositions

for name, proc in [("Dense MF", dense), ("BandMF", band), ("Gaussian×n", gauss)]:
    print(f"{name:12s}  ε = {proc.cgf().epsilon_at(1e-5):.4f}")
```

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `n_steps` | 1 – ~100 (practical limit) | Total training iterations. Memory is $O(n^2)$. |
| `epochs` | 1 – `n_steps` | Number of epochs. Must divide `n_steps`. |
| `bands` | `None` or 1 – `n_steps` | Optional banding constraint on the strategy. |
| `equal_norm` | `False` / `True` | Optimize with equal column norm constraint. |

**Tips**:

- **$n \leq 100$** is the practical limit. At $n = 100$, the strategy
  matrix has 10,000 entries — still manageable. At $n = 1000$, it has
  1,000,000 entries and optimization becomes slow.
- **`epochs` must divide `n_steps`**. For 2 epochs over 50 steps, each
  epoch has 25 steps.
- **`equal_norm = True`** adds an extra constraint that all columns of $C$
  have equal norm. This can improve robustness but slightly increases error.
- **`bands`** optionally restricts the dense matrix to a banded structure.
  This combines the optimality of dense optimization with the memory
  efficiency of banding — useful when $n$ is moderate (50–200) and you
  want better-than-BandMF but less-than-full-dense.
- For $n > 100$, prefer [BandMF](band-mf.md) or [BLT](blt.md).
- The dense strategy is the **gold standard** for benchmarking: if BandMF
  or BLT approach the dense $\varepsilon$ on a short run, they are
  near-optimal.

## References

- **Denisov et al. (2022)** — [Improved Differential Privacy for SGD via Optimal Private Linear Operators](https://arxiv.org/abs/2202.08312).
  Dense matrix factorization mechanism and optimality theory.
- **Choquette-Choo et al. (2022)** — [Privacy Amplification for Matrix Mechanisms](https://arxiv.org/abs/2211.06530).
  Fixed-epoch sensitivity analysis for dense strategies.
- **Kairouz et al. (2021)** — [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039).
  DP-FTRL framework that Dense MF implements optimally.
