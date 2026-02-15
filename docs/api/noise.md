# Noise Injection

The `opaque.noise` module provides functions for adding calibrated noise to gradients for differential privacy.

## Overview

After clipping gradients, DP-SGD requires adding noise proportional to the clip norm and noise multiplier. The
noise obscures individual contributions, providing the actual privacy guarantee.

Opaque provides two noise mechanisms:

- **`gaussian_noise()`** / **`gaussian_noise_stateful()`** — Standard (unbounded) Gaussian noise. The default for most DP-SGD
  workflows.
- **`bounded_gaussian_noise()`** / **`bounded_gaussian_noise_stateful()`** — Bounded Gaussian noise using a truncated normal
  distribution ([Chen & Hale, 2024](https://arxiv.org/abs/2211.17230)). Guarantees all noisy outputs lie within a
  specified domain — useful when gradient values must stay in a valid range.

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## Standard Gaussian

::: opaque.noise.gaussian_noise

## Bounded Gaussian

::: opaque.noise.bounded_gaussian_noise
