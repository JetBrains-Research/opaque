# DP-λCGD

**DP-λCGD** (Kalinin et al., 2026) is a correlated noise mechanism that uses a
single parameter λ ∈ [0, 1) to control noise correlation across training steps.
Its key advantage is **zero extra memory** — noise correlation is achieved via
PRNG seed replay instead of storing previous noise vectors.

- **Paper**: [DP-λCGD: Leveraging Correlated Gradients for Improved DP-SGD](https://arxiv.org/abs/2601.22334)
- **Strategy matrix**: Lower-triangular Toeplitz with entries C[i,j] = λ^{i-j}
- **Inverse**: Bidiagonal — 1 on diagonal, -λ on subdiagonal (bandwidth 2)
- **Memory**: Zero extra (PRNG replay regenerates previous noise from its seed)
- **Amplification**: Balls-in-Bins (BnB) with MC dominating pair accounting

## Accounting

```python
import opaque_accounting as acc

# Multi-epoch BnB accounting (returns TOTAL cost, no further composition needed)
training = acc.balls_in_bins(
    acc.lambda_cgd(noise_multiplier, lambda_=0.9,
                   n_steps=total_steps,
                   min_sep=steps_per_epoch,
                   max_participations=num_epochs),
    num_bins=steps_per_epoch,
    num_epochs=num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ (calibrated or fixed) |
| `lambda_` | Correlation coefficient in [0, 1). λ=0 is DP-SGD. |
| `n_steps` | Total training steps |
| `min_sep` | Steps per epoch (= bins per epoch) |
| `max_participations` | Number of epochs |
| `normalized` | Column-normalize C (default True, gives sensitivity=1 for k=1) |

### Sensitivity

The sensitivity depends only on the strategy matrix C, not on the optimizer
workload (momentum, LR schedule). This is a fundamental property of the MF
privacy framework. The sensitivity formula (Theorem 1, eq 15 of the paper)
has a closed-form expression in terms of λ, min_sep, and max_participations.

## Noise generation

```python
from opaque.noise import lambda_cgd_noise

noise_fn, state = lambda_cgd_noise(
    grad_template, n_steps=total_steps,
    stddev=noise_multiplier * clip_sensitivity,
    key=key(seed), lambda_=0.9,
)
```

At each step t, the noise function:
1. Generates z_t from step t's PRNG seed
2. Regenerates z_{t-1} from step t-1's PRNG seed (PRNG replay)
3. Computes correlated noise: n_t = z_t - λ · z_{t-1}
4. Optionally applies column-norm scaling: n_t *= d_t

## Relationship to BISR

DP-λCGD is the bandwidth-2 special case of [BISR](bisr.md). For bandwidth > 2,
use `acc.bisr()` which generalises the correlation structure.
