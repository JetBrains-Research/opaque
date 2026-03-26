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

- **`band_mf_noise()`** — BandMF banded Toeplitz correlated noise
- **`blt_mf_noise()`** — Buffered Linear Toeplitz (BLT) correlated noise
- **`dense_mf_noise()`** — Dense optimal strategy (small n)
- **`identity_mf_noise()`** — Identity (DP-SGD via MF API, easy to swap)
- **`custom_mf_noise()`** — Bring-your-own noising matrix

All noise functions return `(noise_fn, state)` where `noise_fn(grads, state) -> (noisy_grads, new_state)`.

### State Classes

- **`GaussianNoiseState`** — State for `gaussian_noise()`. Holds step counter and RNG key.
- **`MFNoiseState`** — State for all MF noise functions. Holds noise buffers, step counter, and correlation state.

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

::: opaque.noise.band_mf_noise

::: opaque.noise.blt_mf_noise

::: opaque.noise.dense_mf_noise

::: opaque.noise.identity_mf_noise

::: opaque.noise.custom_mf_noise

## State Classes

::: opaque.noise.gaussian_noise.GaussianNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.noise.matrix_factorization.MFNoiseState
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
