# Gradient Clipping

The `opaque.clipping` module provides the core functionality for per-sample gradient clipping, the foundation of DP-SGD.

## Overview

Per-sample gradient clipping bounds the influence any single training example can have on the model, enabling
differential privacy. Opaque provides three clipping functions:

1. **`clipped_grad()`** - High-level API for gradient clipping
  - Automatically differentiates loss function
  - Clips per-example gradients
  - Sums clipped gradients

2. **`clipped_fun()`** - Clip and sum arbitrary function outputs
  - Works with any function returning PyTrees
  - Primary building block

3. **`clip_pytree()`** - Low-level PyTree clipping
  - Clips existing gradients
  - Used internally by other functions

**Key concept**: All functions clip to maximum L2 norm, ensuring bounded sensitivity for DP.

**See also**: [Per-Sample Gradient Clipping User Guide](../../user-guide/clipping.md)

## API Documentation

::: opaque.clipping
    options:
      show_source: true
      heading_level: 2
