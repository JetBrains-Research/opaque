# Gradient Clipping

The `opaque.clipping` module provides algorithm-agnostic per-example
gradient clipping primitives. DP-SGD-specific variants (adaptive, AUTO-S)
live in `opaque.dpsgd.clipping`.

## Overview

Per-sample gradient clipping bounds the influence any single training example can have on the model, enabling
differential privacy.

### Clipping Functions

1. **`clipped_grad()`** — High-level API: differentiates, clips, and sums gradients.
2. **`clipped_fun()`** — Clip and sum arbitrary function outputs (PyTrees).
3. **`adaptive_clipped_grad()`** — Adaptive clipping (Andrew et al. 2021) with automatic threshold tuning.
4. **`auto_clipped_grad()`** — AUTO-S (Bu et al. NeurIPS 2023) automatic per-example gradient scaling.
5. **`auto_clipped_fun()`** — AUTO-S for arbitrary function outputs.
6. **`clip_pytree()`** — Low-level: clip an existing PyTree of gradients.
7. **`auto_scale_pytree()`** — Low-level: AUTO-S scale an existing PyTree.

### State Types

- **`ClipState`** — Base class for clipping state.
- **`FixedClipState`** — Marker state for fixed `clipped_grad` / `clipped_fun`.
- **`AdaptiveClipState`** — Internal execution state for `adaptive_clipped_grad`.
- **`AutoClipState`** — Marker state for `auto_clipped_grad` / `auto_clipped_fun`.

### Auxiliary Output Types

- **`ClippedGradAux`** — Per-example `loss_values`, `grad_norms`, `clipped_grad_norms`, `loss_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_grad`).
- **`ClippedFunAux`** — Per-example `values`, `norms`, `clipped_norms`, `value_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_fun`).
- **`AdaptiveClippedGradAux`** — Inherits all fields from `ClippedGradAux` (from `adaptive_clipped_grad`).
- **`AutoClippedGradAux`** — Inherits `ClippedGradAux` fields with AUTO-S semantics: `grad_norms` (pre-scale), `clipped_grad_norms` (post-scale, bounded by R), `clipping_rate` (fraction with ‖g‖ > R), `loss_values`, `loss_aux`, `group_norms`, `batch_size`.
- **`AutoClippedFunAux`** — Inherits `ClippedFunAux` fields with AUTO-S semantics: `norms` (pre-scale), `clipped_norms` (post-scale, bounded by R), `clipping_rate` (fraction with ‖v‖ > R), `values`, `value_aux`, `group_norms`, `batch_size`.

### Distributed Sync Helpers

Use `sync()` from `opaque.distributed` to synchronize any clipping state or aux
object. It auto-dispatches to the right function based on type:

- **`sync(FixedClipState)`** → marker-state passthrough.
- **`sync(AdaptiveClipState)`** → aggregates counts and recomputes the internal adaptive threshold.
- **`sync(ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux)`** → gathers aux across ranks.

**See also**: [Per-Sample Gradient Clipping User Guide](../user-guide/clipping.md)

## API Documentation

::: opaque.clipping
    options:
      show_source: true
      heading_level: 2

::: opaque.dpsgd.clipping
    options:
      show_source: true
      heading_level: 2
