# BISR (Banded Inverse Square Root)

**BISR** (Kalinin et al., ICLR 2026) generalises [DP-λCGD](lambda-cgd.md) from
bandwidth 2 to arbitrary bandwidth p ≥ 2. The inverse of the strategy matrix
C^{-1} is a banded Toeplitz matrix whose coefficients are analytically determined
from the inverse square root of the workload matrix.

- **Paper**: [Back to Square Roots: Banded Inverse Square Root for DP Matrix Factorization](https://arxiv.org/abs/2505.12128)
- **Strategy matrix**: Inverse is banded Toeplitz with bandwidth p
- **Memory**: p-1 noise vectors (via PRNG replay or small buffer)
- **Optimality**: Asymptotically optimal (matches upper and lower bounds)
- **Amplification**: Balls-in-Bins (BnB) with MC dominating pair accounting

## Accounting

```python
import opaque_accounting as acc

# BISR with bandwidth=4
training = acc.balls_in_bins(
    acc.bisr(noise_multiplier, n_steps=total_steps, bandwidth=4,
             min_sep=steps_per_epoch, max_participations=num_epochs),
    num_bins=steps_per_epoch,
    num_epochs=num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ |
| `n_steps` | Total training steps |
| `bandwidth` | Number of bands p (≥ 2). Higher = better utility, more PRNG replays. |
| `min_sep` | Steps per epoch |
| `max_participations` | Number of epochs |
| `momentum` | Optimizer momentum β. Enters coefficient computation (changes C). |

### BISR coefficients

The inverse coefficients c̃_k are computed from the inverse square root
formula (Lemma 1 of the paper):

- Base sequence: r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}
- With momentum β: c̃_k = Σ_{j=0}^{k} r̃_j · β^j · r̃_{k-j}

For β=0 (FTRL): c̃ = [1, -0.5, -0.125, -0.0625, ...]

Momentum enters the coefficient computation, changing the strategy matrix C
itself. The privacy analysis then uses raw C columns (momentum does not
enter sensitivity or Gram matrix computation).

## Noise generation

```python
from opaque.noise import bisr_noise

noise_fn, state = bisr_noise(
    grad_template, n_steps=total_steps,
    stddev=noise_multiplier * clip_sensitivity,
    key=key(seed), bandwidth=4, momentum=0.9,
)
```

The noise function regenerates p-1 previous noise vectors via PRNG replay
and computes the linear combination defined by the BISR coefficients.

## Bandwidth selection

| Bandwidth | Memory | Runtime overhead | Utility |
|-----------|--------|-----------------|---------|
| p=2 | Zero extra | <1% | Good (= λCGD) |
| p=4 | 3 vectors | ~4% | Better |
| p=8 | 7 vectors | ~8% | Near-optimal |
| p=16 | 15 vectors | ~15% | Optimal for most tasks |

The paper shows BISR matches BandMF at the same bandwidth while being
simpler to implement and asymptotically optimal.
