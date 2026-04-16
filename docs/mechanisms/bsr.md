# BSR (Banded Square Root)

**BSR** (Kalinin & Lampert, NeurIPS 2024) is a matrix factorization mechanism where the **strategy matrix** is a **banded lower-triangular Toeplitz** matrix obtained from the **matrix square root** of the workload for **SGD with Polyak momentum and multiplicative weight decay**.

Opaque exposes **closed-form coefficients** (Theorem 1 in the paper): no L-BFGS optimization step at initialization.

- **Paper**: [Banded Square Root Matrix Factorization for Differentially Private Model Training](https://arxiv.org/abs/2405.13763)
- **Strategy matrix**: Banded lower-triangular Toeplitz \(C^{|p|}_{\alpha,\beta}\)
- **Sensitivity**: Closed form under min-separation participation (Theorem 2; Rust `toeplitz_minsep_sensitivity_squared`)
- **Amplification**: Balls-in-Bins with Monte Carlo PLD (same Gram path as BLT/BISR Toeplitz Gram)

## Accounting

```python
from opaque.noise.mf import bsr_strategy
import opaque_accounting as acc

strategy = bsr_strategy(
    bandwidth=8,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    momentum=0.95,
    weight_decay=1.0,
)

training = acc.balls_in_bins(
    acc.bsr(
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
| `bandwidth` | Bandwidth \(p\) (≥ 1). Coefficients \(c_j\) for \(j \ge p\) are zero. |
| `n_steps` | Total training steps |
| `min_sep` | Minimum separation between participations (typically steps per epoch) |
| `max_participations` | Maximum participations per user (epochs) |
| `momentum` | Polyak momentum \(\beta \in [0, 1)\) |
| `weight_decay` | Multiplicative decay \(\alpha \in (0, 1]\); **must satisfy** \(\alpha > \beta\) (paper regime) |

### Accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise \(\sigma\) |
| `sensitivity` | From `strategy.sensitivity` |
| `gram_matrix` | From `strategy.gram_matrix` for BnB Monte Carlo |

## Noise generation

```python
from opaque.noise.mf import mf_noise, bsr_strategy
from opaque.random import key

strategy = bsr_strategy(
    bandwidth=8,
    n_steps=1000,
    min_sep=195,
    max_participations=8,
    momentum=0.95,
    weight_decay=1.0,
)
noise_fn, state = mf_noise(
    grad_template=params,
    strategy=strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(seed),
)
```

## Assumptions and limitations

- **Closed-form regime only**: \(\beta \in [0,1)\), \(\alpha \in (0,1]\), \(\alpha > \beta\). Other hyperparameters raise `ValueError` with guidance to use `band_mf_strategy`.
- **`examples/train_dp_ftrl.py`**: workload \(\alpha\) for BSR is **`--bsr-alpha`** (default `1.0`). Optimizer weight decay is **`--weight-decay`** (default `0.0` for both SGD and JME AdamW), so you are not forced to use \(\alpha=1\) in the optimizer when using BSR noise accounting.
- **No learning-rate schedule in v1**: BSR coefficients assume the paper’s workload; use BandMF/BLT with `lr_schedule` if you need schedule-shaped workloads in the optimizer.
- **vs BISR**: BISR bands the **inverse** square root of the workload (different coefficient family). BSR bands the **forward** square root factors from Theorem 1.
- **`normalized=True`**: Not supported in v1.

## References

- Kalinin, Lampert (2024) — [arXiv:2405.13763](https://arxiv.org/abs/2405.13763)
