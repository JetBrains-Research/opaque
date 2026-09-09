# DP-FTRL mechanisms

DP-FTRL releases **correlated** noise across the whole training run.
Each mechanism is a matrix-factorization strategy whose Gram matrix
captures the correlation structure; the privacy accountant treats the
whole training run (not per-step composition) so the noise is
calibrated against `n_steps` once.

## Mechanisms

- **[Band-MF](band-mf.md)** — banded matrix factorization (Choquette-Choo
  et al. 2023). Standard DP-FTRL baseline.
- **[BLT](blt.md)** — buffered linear Toeplitz (Dvijotham et al. 2024).
  Multi-epoch BLT covers iterating over the dataset multiple times.
- **[BiSR](bisr.md)** — banded inverse square root.
- **[BSR](bsr.md)** — banded square root.
- **[λ-CGD](lambda-cgd.md)** — DP-λCGD (PRNG-replay noise; zero extra
  memory at inference time).

The simplest case (sensitivity 1, no correlation) is the
**identity strategy** — exposed via
`opaque.dpftrl.noise.identity_strategy()` paired with
`opaque.dpftrl.accounting.mf_gaussian(nm, identity_strategy())`. Useful
as a sanity check or when comparing against vanilla DP-SGD on equal
footing.

## Pairing with sampling

DP-FTRL pairs the noise mechanism with one of three amplification
factories — **all** parameterised by `n_steps`:

- `opaque.dpftrl.accounting.poisson(...)` — Poisson subsampling
  (cyclic-Poisson under banded MF).
- `opaque.dpftrl.accounting.b_min_sep(...)` — b-min-separation
  participation pattern.
- `opaque.dpftrl.accounting.balls_in_bins(...)` — fixed-partition
  participation.

Each amplification factory wraps a mechanism and produces a single
`DpProcess` representing the full training run.

## Router-load release under matrix mechanisms

Mixture-of-experts load balancing under DP-FTRL uses the same
[router-load release](../dp-sgd/moe-load-balancing.md) as DP-SGD: the
zero probe parameter's group joins the `PerGroup` bound that
`mf_gaussian_noise` latches on its first call, the correlated noise
$C^{-1} Z$ is applied to the probe leaf like every other leaf, and the
whole-run accountant `mf_gaussian(nm, strategy)` is unchanged (Denisov
et al. 2022, Theorem 2.1, applied to the per-group-whitened stream).
The participation pattern is shared across groups (same example, same
step), so the gradient's `min_sep` / `max_participations` apply to the
probe group as well.

What differs from the per-step case is the post-processing: the
smoothing filter's noise factors are computed at setup from the
instantiated strategy's streaming Toeplitz inverse (never a dense
solve), and the anti-correlated noise makes the smoothed load estimate
two to three and a half times more accurate than under DP-SGD at the
same multiplier. A strategy whose noise operator is not the Toeplitz
inverse of its coefficients (DP-λCGD) is rejected at setup. The price
is the same $\sqrt{1+\rho}$ gradient-noise inflation; the un-amplified
band-MF multiplier bounds the cost from above (see the
[cost table](../dp-sgd/moe-load-balancing.md#cost-table)).

## See also

- [DP-FTRL end-to-end guide](../../user-guide/dp-ftrl.md) — full
  training pipeline.
- [DP-SGD mechanisms](../dp-sgd/index.md) — the per-step Gaussian
  family.
