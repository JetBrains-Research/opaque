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

## Paired second-moment release

When `clipped_grad(..., second_moment=True)` produces a
`SecondMomentClippingOutput`, both `gaussian_noise` /
`truncated_gaussian_noise` (DP-SGD) and `mf_noise(..., second_moment_strategy=...)`
(DP-FTRL) consume it and emit a `SecondMomentNoiseOutput` with paired noise
on both streams.

The runtime σ allocation is **sensitivity-proportional** and works
identically for scalar `max_norm` and per-group `PerGroup` `max_norm`. For
each group \(g\) (single-group `K=1` for scalar clipping), let
\(\Delta^{(1)}_g\) be the first-stream per-record bound (`C_g / n` for
averaged clipping, or `ζ · ‖C₁‖` strategy-amplified for DP-FTRL) and
\(\Delta^{(2)}_g\) the second-stream per-record bound (`C_g² / n` derived
from the same clip; `ζ² · ‖C₂‖` for DP-FTRL). Then with noise multiplier
\(\text{nm}\), set

\[
  S := \sum_h \bigl(\Delta^{(1)}_h + \Delta^{(2)}_h\bigr),\qquad
  \sigma^{(1)}_g := \text{nm}\sqrt{\Delta^{(1)}_g\,S},\qquad
  \sigma^{(2)}_g := \text{nm}\sqrt{\Delta^{(2)}_g\,S}.
\]

This satisfies the joint Mahalanobis budget with equality:

\[
  \sum_{g,i} \Bigl(\frac{\Delta^{(i)}_g}{\sigma^{(i)}_g}\Bigr)^2
  = \frac{1}{\text{nm}^2}.
\]

So the paired release has **the same PLD as a single sensitivity-1
Gaussian release at multiplier `nm`** — i.e. **the same first-moment-only
mechanism at the same noise multiplier**. Accounting is plain `gaussian(nm)`
for DP-SGD and the underlying `mf_gaussian(nm, …)` for DP-FTRL; there is
no separate transformation wrapper and no `ρ` knob.

`mf_noise` accepts scalar `max_norm` only; per-group + DP-FTRL is not
implemented.

## Standard Gaussian

::: opaque.dpsgd.noise.gaussian_noise

## Bounded Gaussian — Truncated (renormalized)

`truncated_gaussian_noise` consumes the same `SecondMomentClippingOutput`
inputs as `gaussian_noise` and uses the same sensitivity-proportional joint
allocation; the only difference is that the per-coordinate noise sample is
drawn from a truncated normal of half-width `radius·σ`.

::: opaque.dpsgd.noise.truncated_gaussian_noise

## Joint-allocation helper

::: opaque.dpsgd.noise.paired_noise_stddevs

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
