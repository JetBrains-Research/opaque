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

The noise **strategy** and the privacy **accounting** are connected through two
values that the strategy computes: `sensitivity` and `gram_matrix`. The strategy
knows the full structure of the mechanism (λ, participation pattern, number of
steps) and derives these quantities from the strategy matrix C. The accounting
constructor then takes only these pre-computed values — it does not need to know
how they were obtained.

This separation keeps accounting constructors simple and avoids duplicating
mechanism parameters in two places.

```python
from opaque.dpftrl.noise import lambda_cgd_strategy
import opaque.accounting as acc

# 1. Create strategy — computes sensitivity and Gram matrix internally
strategy = lambda_cgd_strategy(
    lambda_=0.9,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
)

# 2. Build accounting mechanism from strategy-derived quantities
training = acc.balls_in_bins(
    acc.lambda_cgd(noise_multiplier,
                   sensitivity=strategy.sensitivity,
                   gram_matrix=strategy.gram_matrix),
    num_bins=steps_per_epoch,
    num_epochs=num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Strategy parameters

| Parameter | Description |
|-----------|-------------|
| `lambda_` | Correlation coefficient in [0, 1). λ=0 is DP-SGD. |
| `n_steps` | Total training steps |
| `min_sep` | Steps per epoch (= bins per epoch) |
| `max_participations` | Number of epochs |
| `normalized` | Column-normalize C (default True, gives sensitivity=1 for k=1) |

### Accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ (calibrated or fixed) |
| `sensitivity` | From `strategy.sensitivity` — L2 sensitivity of C |
| `gram_matrix` | From `strategy.gram_matrix` — for BnB Monte Carlo accounting |

### Sensitivity

The sensitivity depends only on the strategy matrix C, not on the optimizer
workload (momentum, LR schedule). This is a fundamental property of the MF
privacy framework. The sensitivity formula (Theorem 1, eq 15 of the paper)
has a closed-form expression in terms of λ, min_sep, and max_participations.

## Assumptions and limitations

- Bandwidth is **fixed** (bidiagonal inverse); correlation is controlled by a single \(\lambda\). Does **not** accept `momentum` (use `bisr_strategy` with bandwidth > 2 for momentum-aware coefficients).
- Uses **Balls-in-Bins** amplification like other epoch-structured MF mechanisms; sampler semantics must match accounting.
- **Private second moments (DP-Adam)**: auto-deriving the second-moment strategy is **not supported** for λCGD. Pass `second_moment_strategy` explicitly.
- Broader MF context: [Correlated noise (DP-FTRL)](../user-guide/dp-ftrl.md).

## Noise generation

```python
from opaque.dpftrl.noise import mf_noise, lambda_cgd_strategy
from opaque.random import key

strategy = lambda_cgd_strategy(
    lambda_=0.9,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
)
noise_fn, state = mf_noise(
    grad_template, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(seed),
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
