# Gradient denoising (`opaque.denoising`)

Post-processing denoisers for noisy gradients. The public entry point for DiSK
is **`disk_denoiser`** in **`opaque.denoising`**.

## Overview

- **`disk_denoiser()`** — Build a DiSK denoiser for a gradient-shaped PyTree.
- **`DenoiserState`** — Abstract base class for denoiser state.
- **`DiskDenoiserState`** — Immutable state for `disk_denoiser`.

Public API is exposed from `opaque.denoising` and `opaque.denoising.disk`.
Underscore-prefixed modules (for example `opaque.denoising._kalman`) are internal
implementation details and may change without notice.

**See also**: [Gradient denoising user guide](../user-guide/denoising.md)

## API documentation

::: opaque.denoising.types.DenoiserState
    options:
      show_source: true
      heading_level: 2

::: opaque.denoising.disk.DiskDenoiserState
    options:
      show_source: true
      heading_level: 2

::: opaque.denoising.disk.disk_denoiser
    options:
      show_source: true
      heading_level: 2
