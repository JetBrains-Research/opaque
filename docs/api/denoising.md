# Gradient denoising (`opaque.denoising`)

Post-processing denoisers for noisy gradients. The public entry point for DiSK
is **`disk_denoiser`** in **`opaque.denoising`**.

## Overview

- **`disk_denoiser()`** — Build a DiSK denoiser for a gradient-shaped PyTree.
- **`DenoiserState`** — Abstract base class for denoiser state (in `opaque.denoising.types`).
- **`DiskDenoiserState`** — Immutable state for `disk_denoiser` (in `opaque.denoising.types`).

Public API is exposed from `opaque.denoising` (`disk_denoiser`) and
`opaque.denoising.types` (state types). Underscore-prefixed modules (for
example `opaque.denoising._disk`) are internal
implementation details and may change without notice.

**See also**: [Gradient denoising user guide](../user-guide/denoising.md)

## API documentation

::: opaque.denoising.types.DenoiserState
    options:
      show_source: true
      heading_level: 2

::: opaque.denoising.types.DiskDenoiserState
    options:
      show_source: true
      heading_level: 2

::: opaque.denoising.disk_denoiser
    options:
      show_source: true
      heading_level: 2
