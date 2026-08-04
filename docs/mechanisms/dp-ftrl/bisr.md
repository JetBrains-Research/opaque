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

Like all MF mechanisms, the noise **strategy** computes `sensitivity` and
`gram_matrix` from the mechanism parameters; the accounting constructor
receives only these pre-computed values. This avoids duplicating mechanism
parameters and keeps the accounting API uniform across all MF mechanisms.

```python
from opaque.dpftrl.noise import bisr_strategy
import opaque.accounting as acc           # cross-cutting balls_in_bins
import opaque.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories

# 1. Create strategy — computes sensitivity and Gram matrix internally
strategy = bisr_strategy(
    bandwidth=4, n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    momentum=0.9,
)

# 2. Build accounting mechanism via strategy.as_mechanism
training = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    num_bins=steps_per_epoch,
    n_steps=steps_per_epoch * num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Strategy parameters

| Parameter | Description |
|-----------|-------------|
| `bandwidth` | Number of bands p (≥ 2). Higher = better utility, more PRNG replays. |
| `n_steps` | Total training steps |
| `min_sep` | Steps per epoch |
| `max_participations` | Number of epochs |
| `momentum` | Optimizer momentum β. Enters coefficient computation (changes C). |

### Accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ |
| `sensitivity` | From `strategy.sensitivity(n_steps=...)` — L2 sensitivity of the BISR strategy |
| `gram_matrix` | From `strategy.gram_matrix` — for BnB Monte Carlo accounting |

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
from opaque.dpftrl.noise import mf_gaussian_noise, bisr_strategy
from opaque.random import key

strategy = bisr_strategy(
    bandwidth=4,
    momentum=0.9,
)
noise_fn, state = mf_gaussian_noise(
    grad_template, strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
    key=key(seed),
)
```

The noise function regenerates p-1 previous noise vectors via PRNG replay
and computes the linear combination defined by the BISR coefficients.

## Assumptions and limitations

- **BnB sampling**: pair with a sampler and accounting consistent with Balls-in-Bins (fixed partition semantics where required).
- **Momentum** enters the **inverse** coefficient construction (Lemma 1); sensitivity and Gram use the resulting strategy matrix.
- **No `lr_schedule`**: BISR coefficients are analytically determined from the prefix-sum workload; schedule-aware BISR would require a different construction (see arXiv:2511.17994). Use BandMF/BLT with `lr_schedule` for schedule-shaped workloads.
- **Not BSR**: BISR bands the **inverse** square root construction (generalised λCGD). [BSR](bsr.md) uses the **forward** square-root closed form for SGD+momentum+weight decay.
- For a high-level comparison of MF mechanisms, see [Matrix factorization (MF)](../../user-guide/dp-ftrl.md).

## Bandwidth selection

Bandwidth `p` retains `p - 1` noise vectors. Measure the runtime and utility
trade-off on your workload.
