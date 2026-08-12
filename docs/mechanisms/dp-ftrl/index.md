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

## See also

- [DP-FTRL end-to-end guide](../../user-guide/dp-ftrl.md) — full
  training pipeline.
- [DP-SGD mechanisms](../dp-sgd/index.md) — the per-step Gaussian
  family.
