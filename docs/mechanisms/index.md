# Mechanisms

A **mechanism** is a randomized algorithm that adds noise to a query result to
provide differential privacy. The mechanism determines the noise distribution,
its support, and how privacy loss is computed. Opaque implements six mechanisms
across two families.

## Independent noise (DP-SGD)

Each training step adds fresh, independent noise to the clipped gradient sum.
Simple and broadly applicable.

| Mechanism | Noise distribution | Support | Privacy ordering |
|-----------|--------------------|---------|-----------------|
| [Gaussian](gaussian.md) | $\mathcal{N}(0, \sigma^2)$ | $(-\infty, +\infty)$ | Baseline |
| [Rectified Gaussian](rectified-gaussian.md) | $\mathcal{N}(0, \sigma^2)$ clamped | $[-R\sigma, R\sigma]$ | $\leq$ Gaussian |
| [Truncated Gaussian](truncated-gaussian.md) | $\mathcal{N}(0, \sigma^2)$ renormalized | $[-R\sigma, R\sigma]$ | $\leq$ Rectified $\leq$ Gaussian |

**Privacy ordering** means: for the same noise multiplier $\sigma$ and radius $R$,

$$\varepsilon_{\text{truncated}} \leq \varepsilon_{\text{rectified}} \leq \varepsilon_{\text{Gaussian}}$$

at every $\delta$. Truncated gives the tightest accounting; Gaussian the loosest.
All three converge as $R \to \infty$.

## Correlated noise (DP-FTRL)

Instead of independent noise at each step, matrix-factorization (MF) mechanisms
add *correlated* noise designed to partially cancel over the training run. This
reduces effective noise on cumulative updates, improving accuracy for the same
privacy budget — at the cost of knowing the total number of steps in advance.

| Mechanism | Strategy | Memory | Best for |
|-----------|----------|--------|----------|
| [BandMF](band-mf.md) | Banded Toeplitz | $O(\text{bands})$ | General use, moderate runs |
| [BLT](blt.md) | Buffered Linear Toeplitz | $O(\text{buffers})$ | Long runs ($n > 5000$), multi-epoch |
| [Dense MF](dense-mf.md) | Full $n \times n$ matrix | $O(n^2)$ | Short runs ($n < 100$), optimal noise |

## Which mechanism should I use?

```
Need correlated noise across steps (DP-FTRL)?
│
├─ No ─── Want tighter bounds from bounded noise?
│         ├─ Yes → Truncated Gaussian (tightest independent bounds)
│         └─ No  → Gaussian (standard DP-SGD, simplest)
│
└─ Yes ── How many training steps?
          ├─ n < 100  → Dense MF (optimal but O(n²) memory)
          ├─ n < 5000 → BandMF + cyclic Poisson (good default)
          └─ n > 5000 → BLT (memory-efficient, supports multi-epoch)
```

For most DP-SGD workloads, **Gaussian** is the right starting point.
Switch to **Truncated Gaussian** if you want ~5-15% tighter $\varepsilon$
for free (same noise level, better accounting). Use an MF mechanism only
when you need the privacy-utility improvement of correlated noise and are
willing to fix the training length in advance.

## Amplification compatibility

Subsampling amplification reduces per-step privacy cost. Not all mechanisms
support all amplification types:

| Mechanism | `poisson()` | `truncated_poisson()` | `cyclic_poisson()` |
|-----------|:-----------:|:---------------------:|:-------------------:|
| Gaussian | Yes | Yes | — |
| Rectified Gaussian | Yes | — | — |
| Truncated Gaussian | Yes | — | — |
| BandMF | — | — | Yes |
| BLT | *internal* | — | — |
| Dense MF | *internal* | — | — |

- **`poisson()`**: Standard Poisson subsampling. Each example included
  independently with probability $q$. Works with all three Gaussian variants.
- **`truncated_poisson()`**: Poisson with a batch-size cap. Currently only
  supports standard Gaussian as the inner mechanism.
- **`cyclic_poisson()`**: Cyclic decomposition specific to BandMF. Decomposes
  $n$ steps into $\lceil n/b \rceil$ independent groups.
- **internal**: BLT and Dense MF handle multi-participation patterns (epochs,
  min-sep) within their own sensitivity computation — no external amplification
  wrapper needed.

## Quick comparison

```python
import opaque.accounting as acc

# --- Independent noise ---
gauss     = acc.poisson(acc.gaussian(1.0), sample_rate=0.01) * 1000
rect      = acc.poisson(acc.rectified_gaussian(1.0, 5.0), sample_rate=0.01) * 1000
trunc     = acc.poisson(acc.truncated_gaussian(1.0, 5.0), sample_rate=0.01) * 1000

# --- Correlated noise ---
# Note: cyclic_poisson's sample_rate is a per-group Poisson probability
# (typically ≈ bands * q when q is the usual DP-SGD sampling rate).
band      = acc.cyclic_poisson(acc.band_mf(1.0, 1000, bands=10), sample_rate=0.01)
blt       = acc.blt_mf(1.0, 1000)
dense     = acc.dense_mf(1.0, 50, epochs=2)

for name, proc in [("Gaussian", gauss), ("Rectified", rect),
                    ("Truncated", trunc), ("BandMF", band),
                    ("BLT", blt), ("Dense", dense)]:
    print(f"{name:12s}  ε = {proc.epsilon_at(1e-5):.4f}")
```
