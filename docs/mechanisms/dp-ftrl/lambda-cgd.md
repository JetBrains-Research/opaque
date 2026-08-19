# DP-λCGD

**DP-λCGD** (Kalinin et al., 2026) is a correlated noise mechanism that uses a
single parameter \(\lambda \in [0, 1)\) to control noise correlation across training steps.
Its key advantage is **zero extra memory** — noise correlation is achieved via
PRNG seed replay instead of storing previous noise vectors.

- **Paper**: [DP-λCGD: Efficient Noise Correlation for Differentially Private Model Training](https://arxiv.org/abs/2601.22334)
- **Strategy matrix**: Lower-triangular Toeplitz with entries \(C_{i,j} = \lambda^{i-j}\)
- **Inverse**: Bidiagonal — 1 on the diagonal, \(-\lambda\) on the subdiagonal (bandwidth 2)
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
import opaque.accounting as acc           # cross-cutting balls_in_bins
import opaque.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories

# 1. Create a provider-independent strategy recipe
strategy = lambda_cgd_strategy(
    lambda_=0.9,
)

# 2. Build accounting from the same strategy recipe
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
| `lambda_` | Correlation coefficient in \([0, 1)\). \(\lambda = 0\) is DP-SGD. |
| `normalized` | Column-normalize C (default True, gives sensitivity=1 for k=1) |
| `lr_schedule` | Optional per-step schedule for a step-weighted BnB Gram |

Pass `n_steps`, `min_sep`, and `max_participations` to
`mf_gaussian_noise` and the matching accounting amplifier. Strategy queries use
that participation context to derive sensitivity and Gram data.

### Accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ (calibrated or fixed) |
| `sensitivity` | From `strategy.sensitivity(n_steps=...)` — L2 sensitivity of C |
| `gram_matrix` | From `strategy.gram_matrix` — for BnB Monte Carlo accounting |

### Sensitivity

The sensitivity depends only on the strategy matrix C, not on the optimizer
workload. A supplied `lr_schedule` does not change sensitivity or noise; it
selects the corresponding step-weighted Gram for Balls-in-Bins accounting. The
sensitivity formula (Theorem 1, eq 15 of the paper) has a closed-form
expression in terms of λ, min_sep, and max_participations.

## Assumptions and limitations

- Bandwidth is **fixed** (bidiagonal inverse); correlation is controlled by a single \(\lambda\). Does **not** accept `momentum` (use `bisr_strategy` with bandwidth > 2 for momentum-aware coefficients).
- `lr_schedule` weights the BnB Gram on the training-step axis only; it does not add a momentum parameter or change the noise strategy. Use the identical schedule in the optimizer; the strategy cannot validate external updates.
- Uses **Balls-in-Bins** amplification like other epoch-structured MF mechanisms; sampler semantics must match accounting.
- **Private second moments (DP-Adam)**: auto-deriving the second-moment strategy is **not supported** for λCGD. Pass `second_moment_strategy` explicitly.
- Broader MF context: [Correlated noise (DP-FTRL)](../../user-guide/dp-ftrl.md).

## Noise generation

λ-CGD coefficients are NumPy host data, while eager noise sampling and outputs
remain native Torch arrays. Replaying the previous keyed sample is
deterministic within the active provider; it is not a cross-provider bitstream
guarantee.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, lambda_cgd_strategy
from opaque.random import key

strategy = lambda_cgd_strategy(lambda_=0.9)
noise_fn, state = mf_gaussian_noise(
    grad_template, strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
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
use `bisr_strategy(...)`, which generalizes the correlation structure.
