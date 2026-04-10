# Mechanisms

A **mechanism** is a randomized algorithm that adds noise to a query result to
provide differential privacy. The mechanism determines the noise distribution,
its support, and how privacy loss is computed. Opaque implements six mechanisms
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

| Mechanism | Strategy | Memory | Best for |
|-----------|----------|--------|----------|
| [BandMF](band-mf.md) | Banded Toeplitz | $O(\text{bands})$ | General use, moderate runs |
| [BLT](blt.md) | Buffered Linear Toeplitz | $O(\text{buffers})$ | Long runs ($n > 5000$), multi-epoch |

## Which mechanism should I use?

```
Need correlated noise across steps (DP-FTRL)?
│
├─ No ─── Gaussian (standard DP-SGD)
│         Use truncated_gaussian_noise() for bounded support if desired;
│         accounting always uses acc.gaussian().
│
└─ Yes ── How many training steps?
          ├─ n < 5000 → BandMF + cyclic Poisson (good default)
          └─ n > 5000 → BLT (memory-efficient, supports multi-epoch)
```

For most DP-SGD workloads, **Gaussian** is the right starting point.
Use an MF mechanism only when you need the privacy-utility improvement
of correlated noise and are willing to fix the training length in advance.

## Amplification compatibility

Subsampling amplification reduces per-step privacy cost. Not all mechanisms
support all amplification types:

| Mechanism | `poisson()` | `truncated_poisson()` | `cyclic_poisson()` |
|-----------|:-----------:|:---------------------:|:-------------------:|
| Gaussian | Yes | Yes | — |
| BandMF | — | — | Yes |
| BLT | *internal* | — | — |

- **`poisson()`**: Standard Poisson subsampling. Each example included
  independently with probability $q$.
- **`truncated_poisson()`**: Poisson with a batch-size cap.
- **`cyclic_poisson()`**: Cyclic decomposition specific to BandMF. Decomposes
  $n$ steps into $\lceil n/b \rceil$ independent groups.
- **internal**: BLT handles multi-participation patterns (min-sep)
  within its own sensitivity computation — no external amplification
  wrapper needed.

## Quick comparison

```python
import opaque.accounting as acc

# --- Independent noise ---
gauss     = acc.poisson(acc.gaussian(1.0), sample_rate=0.01) * 1000

# --- Correlated noise ---
# Note: cyclic_poisson's sample_rate is a per-group Poisson probability
# (typically ≈ bands * q when q is the usual DP-SGD sampling rate).
band      = acc.cyclic_poisson(acc.band_mf(1.0, 1000, bands=10), sample_rate=0.01)
blt       = acc.blt_mf(1.0, 1000)

for name, proc in [("Gaussian", gauss), ("BandMF", band), ("BLT", blt)]:
    print(f"{name:12s}  ε = {proc.epsilon_at(1e-5):.4f}")
```
