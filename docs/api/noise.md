# Noise Injection

Opaque's noise mechanisms live next to the training paradigm they support:
`opaque.dpsgd.noise` for independent Gaussian-family noise, and
`opaque.dpftrl.noise` for matrix-factorization (correlated) noise. The base
`NoiseState` type that both build on lives in `opaque.types`.

## Overview

After clipping gradients, DP-SGD requires adding noise proportional to the clip norm and noise multiplier. The
noise obscures individual contributions, providing the actual privacy guarantee.

Opaque provides several noise mechanisms:

### Independent Noise (DP-SGD)

- **`gaussian_noise()`** — Standard (unbounded) Gaussian noise. The default for most DP-SGD workflows.
- **`truncated_gaussian_noise()`** — Bounded Gaussian noise (renormalized density). The tail mass
  is redistributed over the bounded interval — no point masses at the boundaries.

### Correlated Noise (DP-FTRL / Matrix Factorization)

- **`mf_noise()`** — Unified correlated noise dispatcher. Takes a strategy object and creates the
  corresponding noise mechanism.

Strategy factories (passed to `mf_noise()`):

- **`band_mf_strategy()`** — BandMF banded Toeplitz correlated noise
- **`blt_strategy()`** — Buffered Linear Toeplitz (BLT) correlated noise
- **`lambda_cgd_strategy()`** — DP-λCGD correlated noise (PRNG replay, zero extra memory)
- **`bisr_strategy()`** — BISR (Banded Inverse Square Root) correlated noise
- **`identity_strategy()`** — Identity (DP-SGD via MF API, easy to swap)

All noise functions return `(noise_fn, state)` where `noise_fn(grads, state) -> (noisy_grads, new_state)`.
When private second moments are enabled, the noisy value is a
`SecondMomentNoiseOutput` carrying both `noisy_grads` and
`noisy_squared_grads`.

### State Classes

- **`NoiseState`** — Abstract base class for all noise state types. Defines `_step_counter` and `_rng_key`.
- **`SecondMomentNoiseOutput`** — Paired first/squared-gradient output for private second moments.
- **`GaussianNoiseState`** — State for `gaussian_noise()`. Holds step counter and RNG key.
- **`MFNoiseState`** — State for `mf_noise()`. Holds internal correlation state, step counter, and RNG key.
- **`SecondMomentMFNoiseState`** — Paired-stream MF state used by `mf_noise(..., second_moment_strategy=...)`.

### Distributed Sync Helpers

Use `sync()` from `opaque.distributed` to validate noise state consistency
across ranks. It auto-dispatches based on type:

- **`sync(GaussianNoiseState)`** — Validate RNG key and step counter match across ranks.
  Rectified and truncated Gaussian noise also return `GaussianNoiseState`,
  so `sync()` handles them automatically.
- **`sync(MFNoiseState)`** — Validate MF noise state matches across ranks.

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## Per-group paired second moments (DP-SGD)

When `clipped_grad(..., second_moment=True)` uses a `PerGroup`
`clipping_norm`, the paired output carries per-group contribution bounds on
**both** the gradient stream and the squared-gradient stream. For group \(g\),
write \(\Delta^{(1)}_g\) and \(\Delta^{(2)}_g\) for those per-record bounds
(after the same normalization as scalar clipping—typically \(\Delta^{(2)}_g\)
tracks \(C_g^2 / n\) when \(\Delta^{(1)}_g\) tracks \(C_g / n\)).

`gaussian_noise` and `truncated_gaussian_noise` then draw **independent**
Gaussian noise in each coordinate, with standard deviations
\((\sigma^{(1)}_g,\sigma^{(2)}_g)\) constant within group \(g\), chosen
**MSE-optimally** subject to one joint analytic Gaussian privacy constraint.
With noise multiplier \(\text{nm}\), defining

\[
  S := \sum_h \bigl(\Delta^{(1)}_h + \Delta^{(2)}_h\bigr),\qquad
  \sigma^{(1)}_g := \text{nm}\sqrt{\Delta^{(1)}_g \, S},\qquad
  \sigma^{(2)}_g := \text{nm}\sqrt{\Delta^{(2)}_g \, S}
\]

gives \(\sum_g \bigl[(\Delta^{(1)}_g/\sigma^{(1)}_g)^2 +
(\Delta^{(2)}_g/\sigma^{(2)}_g)^2\bigr] = 1/\text{nm}^2\) — the same
Mahalanobis-style budget as \(K\)-group single-stream `per_group_noise_stddev`,
extended to \(2K\) privacy streams (first + second per group). **Privacy
accounting remains `gaussian(nm)`** at the same multiplier: no extra penalty
across groups or streams, and no scalar `first_moment_overhead` \(\rho\) on
this path (\(\rho\) applies only when both streams use a scalar `max_norm`).

`mf_noise` does **not** support `PerGroup` bounds; use DP-SGD Gaussian-family
mechanisms for per-group paired second moments.

## Standard Gaussian

::: opaque.dpsgd.noise.gaussian_noise

## Bounded Gaussian — Truncated (renormalized)

`truncated_gaussian_noise` accepts the same `SecondMomentClippingOutput`
input as `gaussian_noise`: when gradients and squared-gradients are passed
together, it allocates the noise budget between the two streams. For **scalar**
`max_norm`, allocation follows `second_moment_stddevs`; the `first_moment_overhead`
parameter (default `sqrt(3/2)`) must match the value used in `acc.second_moment()`.
For **per-group** `max_norm` on both streams, allocation is the joint
Mahalanobis form above and `first_moment_overhead` is unused; accounting stays
`gaussian(nm)` without `acc.second_moment`.

::: opaque.dpsgd.noise.truncated_gaussian_noise

## Matrix Factorization Noise

### Dispatcher

::: opaque.dpftrl.noise.mf_noise

### Strategies

::: opaque.dpftrl.noise.band_mf_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.blt_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.lambda_cgd_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.bisr_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.identity_strategy
    options:
      heading_level: 4

## State Classes

::: opaque.types.NoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.types.SecondMomentNoiseOutput
    options:
      show_source: true
      heading_level: 3

::: opaque.dpsgd.noise.types.GaussianNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.noise.types.MFNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.noise.types.SecondMomentMFNoiseState
    options:
      show_source: true
      heading_level: 3

## Distributed Synchronization

Use `opaque.distributed.sync(state)` — it auto-dispatches on the state's
type to the right sync function. `GaussianNoiseState` and `MFNoiseState`
both register handlers at import time, so no named sync call is required.
