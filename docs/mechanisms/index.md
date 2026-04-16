# Mechanisms

A **mechanism** is a randomized algorithm that adds noise to a query result to
provide differential privacy. The mechanism determines the noise distribution,
its support, and how privacy loss is computed. Opaque implements several mechanisms
across two families.

## Independent noise (DP-SGD)

Each training step adds fresh, independent noise to the clipped gradient sum.
Simple and broadly applicable.

| Mechanism | Noise distribution | Support |
|-----------|--------------------|----------|
| [Gaussian](gaussian.md) | $\mathcal{N}(0, \sigma^2)$ | $(-\infty, +\infty)$ |

For bounded noise support, use `truncated_gaussian_noise()` for noise injection
while accounting with `acc.gaussian()`. See [Gaussian — Bounded noise variant](gaussian.md#bounded-noise-variant).

## Correlated noise (DP-FTRL)

Instead of independent noise at each step, matrix-factorization (MF) mechanisms
add *correlated* noise designed to partially cancel over the training run. This
reduces effective noise on cumulative updates, improving accuracy for the same
privacy budget — at the cost of knowing the total number of steps in advance.

For assumptions (workload vs DP correctness, LR schedules, JME caveats), see the
[Matrix factorization user guide](../user-guide/matrix-factorization.md).

| Mechanism | Strategy | Memory | Best for |
|-----------|----------|--------|----------|
| [BandMF](band-mf.md) | Banded Toeplitz | $O(\text{bands})$ | General use, moderate runs |
| [BLT](blt.md) | Buffered Linear Toeplitz | $O(\text{buffers})$ | Long runs ($n > 5000$), multi-epoch |
| [DP-λCGD](lambda-cgd.md) | PRNG replay (exponential decay) | $O(1)$ | Zero extra memory, any run length |
| [BISR](bisr.md) | Banded inverse square root | $O(p)$ | Asymptotically optimal, generalises λCGD |
| [BSR](bsr.md) | Banded square root (closed form) | $O(p)$ | Paper `alpha`, `beta` kwargs; no optimizer at init |
| [LR-Aware](lr-aware.md) | Schedule-aware Toeplitz square root | $O(p)$ | Exponential LR decay; closed-form $C_\alpha$ |
| Identity | $I$ (no correlation) | $O(1)$ | Baseline / ablation |

## Which mechanism should I use?

```
Need correlated noise across steps (DP-FTRL)?
│
├─ No ─── Gaussian (standard DP-SGD)
│         Use truncated_gaussian_noise() for bounded support if desired;
│         accounting always uses acc.gaussian().
│
└─ Yes ── Constraints?
          ├─ Zero extra memory → DP-λCGD (PRNG replay)
          ├─ Asymptotically optimal → BISR (generalises λCGD)
          ├─ Closed-form workload (α>β) → BSR (NeurIPS 2024)
          ├─ n < 5000 → BandMF + cyclic Poisson (good default)
          └─ n > 5000, multi-epoch → BLT (memory-efficient)
```

For most DP-SGD workloads, **Gaussian** is the right starting point.
Use an MF mechanism only when you need the privacy-utility improvement
of correlated noise and are willing to fix the training length in advance.

## Amplification compatibility

Subsampling amplification reduces per-step privacy cost. Not all mechanisms
support all amplification types:

| Mechanism | `poisson()` | `truncated_poisson()` | `cyclic_poisson()` | `balls_in_bins()` |
|-----------|:-----------:|:---------------------:|:-------------------:|:-----------------:|
| Gaussian | Yes | Yes | — | Yes |
| BandMF | — | — | Yes | — |
| BLT | *internal* | — | — | Yes |
| DP-λCGD | — | — | — | Yes |
| BISR | — | — | — | Yes |
| BSR | — | — | — | Yes |
| LR-Aware | — | — | — | Yes |

- **`poisson()`**: Standard Poisson subsampling. Each example included
  independently with probability $q$.
- **`truncated_poisson()`**: Poisson with a batch-size cap.
- **`cyclic_poisson()`**: Cyclic decomposition specific to BandMF. Decomposes
  $n$ steps into $\lceil n/b \rceil$ independent groups.
- **`balls_in_bins()`**: Random-partition amplification. Each epoch, examples
  are randomly assigned to bins. Used with BLT, DP-λCGD, and BISR.
- **internal**: BLT handles multi-participation patterns (min-sep)
  within its own sensitivity computation — no external amplification
  wrapper needed. BLT also supports `balls_in_bins()` with a pre-computed
  Gram matrix.

## Quick comparison

```python
import opaque_accounting as acc
from opaque.noise.mf import band_mf_strategy, lambda_cgd_strategy

# --- Independent noise ---
gauss     = acc.poisson(acc.gaussian(1.0), sample_rate=0.01) * 1000

# --- Correlated noise ---
# BandMF: strategy computes sensitivity and num_groups
band_s = band_mf_strategy(n_steps=1000, bands=10)
band   = acc.cyclic_poisson(
    acc.band_mf(1.0, sensitivity=band_s.sensitivity,
                num_groups=band_s.num_groups),
    sample_rate=0.01,
)

# DP-λCGD: strategy computes sensitivity and gram_matrix
lcgd_s = lambda_cgd_strategy(
    lambda_=0.9, n_steps=1000, min_sep=100, max_participations=5,
)
lcgd   = acc.balls_in_bins(
    acc.lambda_cgd(1.0, sensitivity=lcgd_s.sensitivity,
                   gram_matrix=lcgd_s.gram_matrix),
    num_bins=100, num_epochs=5,
)

for name, proc in [("Gaussian", gauss), ("BandMF", band), ("λCGD", lcgd)]:
    print(f"{name:12s}  ε = {proc.epsilon_at(1e-5):.4f}")
```
