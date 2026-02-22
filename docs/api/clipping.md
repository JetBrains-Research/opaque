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
- **`NeighboringRelation`** — Enum for DP neighboring relation (ADD_OR_REMOVE_ONE, REPLACE_ONE, REPLACE_SPECIAL).

### Auxiliary Output Types

- **`ClippedGradAux`** — Per-example loss values, gradient norms, clipped norms (from `clipped_grad`).
- **`ClippedFunAux`** — Per-example values, norms, clipped norms (from `clipped_fun`).
- **`AdaptiveClippedGradAux`** — Extends `ClippedGradAux` with clipping rate (from `adaptive_clipped_grad`).

### Distributed Sync Helpers

- **`sync_clip_state(state)`** — Assert `FixedClipState.l2_norm_bound` matches across ranks.
- **`sync_adaptive_clip_state(state)`** — Aggregate counts and recompute global adaptive clip norm.
- **`sync_aux(aux)`** — Gather any clipping aux (``ClippedFunAux``, ``ClippedGradAux``, ``AdaptiveClippedGradAux``) across ranks.

**See also**: [Per-Sample Gradient Clipping User Guide](../../user-guide/clipping.md)

## API Documentation

::: opaque.clipping
    options:
      show_source: true
      heading_level: 2
