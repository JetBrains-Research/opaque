# Gradient Clipping

The `opaque.clipping` module provides the core functionality for per-sample gradient clipping, the foundation of DP-SGD.

## Overview

Per-sample gradient clipping bounds the influence any single training example can have on the model, enabling
differential privacy.

### Clipping Functions

1. **`clipped_grad()`** — High-level API: differentiates, clips, and sums gradients.
2. **`clipped_fun()`** — Clip and sum arbitrary function outputs (PyTrees).
3. **`adaptive_clipped_grad()`** — Adaptive clipping (Andrew et al. 2021) with automatic threshold tuning.
4. **`clip_pytree()`** — Low-level: clip an existing PyTree of gradients.

### State Types

- **`ClipState`** — Base class for clipping state.
- **`FixedClipState`** — State for `clipped_grad` / `clipped_fun` (fixed threshold).
- **`AdaptiveClipState`** — State for `adaptive_clipped_grad` (adapting threshold).

### Auxiliary Output Types

- **`ClippedGradAux`** — Per-example `loss_values`, `grad_norms`, `clipped_grad_norms`, `loss_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_grad`).
- **`ClippedFunAux`** — Per-example `values`, `norms`, `clipped_norms`, `value_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_fun`).
- **`AdaptiveClippedGradAux`** — Inherits all fields from `ClippedGradAux` (from `adaptive_clipped_grad`).

### Distributed Sync Helpers

Use `sync()` from `opaque.distributed` to synchronize any clipping state or aux
object. It auto-dispatches to the right function based on type:

- **`sync(FixedClipState)`** → asserts `clipping_norm` matches across ranks.
- **`sync(AdaptiveClipState)`** → aggregates counts and recomputes global adaptive clip norm.
- **`sync(ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux)`** → gathers aux across ranks.

**See also**: [Per-Sample Gradient Clipping User Guide](../user-guide/clipping.md)

## API Documentation

::: opaque.clipping
    options:
      show_source: true
      heading_level: 2
