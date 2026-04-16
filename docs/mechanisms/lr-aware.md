# LR-Aware (Schedule-Aware Factorization)

**LR-Aware** (Kalinin & Andersson, 2025) is a matrix factorization mechanism that constructs the strategy matrix $C_\alpha$ as the **Toeplitz square root** of the LR-schedule workload, rather than the prefix-sum workload.

For **exponential** LR decay $\chi_t = \beta^{(t-1)/(n-1)}$, the coefficients are closed-form:

$$c_j = \alpha^j \cdot r_j, \quad \alpha = \beta^{1/(n-1)}, \quad r_j = \left|\binom{-1/2}{j}\right|$$

This is the same $r_j$ sequence used by BSR but scaled by powers of $\alpha$ instead of involving momentum $\beta$.

- **Paper**: [Learning Rate Scheduling with Matrix Factorization](https://arxiv.org/abs/2511.17994)
- **Strategy matrix**: Banded lower-triangular Toeplitz $C_\alpha^{|p|}$
- **Sensitivity**: Toeplitz min-sep (same Rust path as BSR/BLT)
- **Amplification**: Balls-in-Bins with Gram matrix

## When to use

Use `lr_aware_strategy` when you train with **exponential LR decay** and want the noise structure to be **optimized for that schedule**. The paper shows this achieves the **optimal MaxSE rate** for exponential decay and improves over prefix-sum factorizations.

For constant LR or non-exponential schedules, use BandMF, BLT, or BISR.

## Accounting

```python
from opaque.noise.mf import lr_aware_strategy
import opaque_accounting as acc

strategy = lr_aware_strategy(
    bandwidth=64,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    lr_decay_beta=0.25,  # LR decays to 25% of initial
)

training = acc.balls_in_bins(
    acc.lr_aware(
        noise_multiplier,
        sensitivity=strategy.sensitivity,
        gram_matrix=strategy.gram_matrix,
    ),
    num_bins=steps_per_epoch,
    num_epochs=num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Strategy parameters

| Parameter | Description |
|-----------|-------------|
| `bandwidth` | Bandwidth $p$ (>= 1). |
| `n_steps` | Total training steps (>= 2). |
| `min_sep` | Minimum separation (steps per epoch). |
| `max_participations` | Epochs. |
| `lr_decay_beta` | Final-to-initial LR ratio $\beta \in (0, 1)$. |

## Noise generation

```python
from opaque.noise.mf import mf_noise, lr_aware_strategy
from opaque.random import key

strategy = lr_aware_strategy(
    bandwidth=64, n_steps=1000,
    min_sep=100, max_participations=10,
    lr_decay_beta=0.25,
)
noise_fn, state = mf_noise(
    grad_template=params,
    strategy=strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(seed),
)
```

## Assumptions and limitations

- **Exponential decay only** in the closed-form path. $\chi_t = \beta^{(t-1)/(n-1)}$.
- **Not compatible with JME auto-derivation** (no principled second-moment mapping for schedule-aware workloads). Provide `second_moment_strategy` explicitly if using Adam.
- For non-exponential schedules, use BandMF/BLT with `lr_schedule` (Toeplitz-surrogate optimization).
- See [Matrix factorization (MF)](../user-guide/matrix-factorization.md) for the broader context.

## References

- Kalinin & Andersson (2025) — [arXiv:2511.17994](https://arxiv.org/abs/2511.17994)
- Related: BSR [arXiv:2405.13763](https://arxiv.org/abs/2405.13763) (same coefficient family)
