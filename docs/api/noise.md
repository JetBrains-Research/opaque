# Noise Injection

The `opaque.noise` module provides functions for adding calibrated noise to gradients for differential privacy.

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

### State Classes

- **`NoiseState`** — Abstract base class for all noise state types. Defines `_step_counter` and `_rng_key`.
- **`GaussianNoiseState`** — State for `gaussian_noise()`. Holds step counter and RNG key.
- **`MFNoiseState`** — State for `mf_noise()`. Holds internal correlation state, step counter, and RNG key.

### Distributed Sync Helpers

Use `sync()` from `opaque.distributed` to validate noise state consistency
across ranks. It auto-dispatches based on type:

- **`sync(GaussianNoiseState)`** — Validate RNG key and step counter match across ranks.
  Rectified and truncated Gaussian noise also return `GaussianNoiseState`,
  so `sync()` handles them automatically.
- **`sync(MFNoiseState)`** — Validate MF noise state matches across ranks.

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## Standard Gaussian

::: opaque.noise.gaussian_noise

## Bounded Gaussian — Truncated (renormalized)

::: opaque.noise.truncated_gaussian_noise

## Matrix Factorization Noise

### Dispatcher

::: opaque.mf.noise.mf_noise

### Strategies

::: opaque.mf.noise.band_mf_strategy
    options:
      heading_level: 4

::: opaque.mf.noise.blt_strategy
    options:
      heading_level: 4

::: opaque.mf.noise.lambda_cgd_strategy
    options:
      heading_level: 4

::: opaque.mf.noise.bisr_strategy
    options:
      heading_level: 4

::: opaque.mf.noise.identity_strategy
    options:
      heading_level: 4

## State Classes

::: opaque.noise.types.NoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.noise.gaussian.GaussianNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.mf.noise.MFNoiseState
    options:
      show_source: true
      heading_level: 3

## Distributed Synchronization

::: opaque.noise.distributed.sync_gaussian_noise_state
    options:
      show_source: true
      heading_level: 3

::: opaque.noise.distributed.sync_mf_noise_state
    options:
      show_source: true
      heading_level: 3
