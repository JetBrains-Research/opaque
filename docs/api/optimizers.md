# DP Optimizers

The `opaque.optimizers` module provides DP-aware optimizer wrappers that integrate with TorchOpt functional optimizers.

## Overview

DP optimization requires:

1. Computing clipped per-example gradients
2. Adding calibrated noise
3. Updating parameters

Opaque provides **adaptive clipping** that automatically tunes the clip norm during training, improving the
privacy-utility tradeoff without weakening privacy guarantees.

**Key function**: `adaptive_clipping()` - Wrap any TorchOpt optimizer with adaptive clipping

**Features**:

- **Automatic clip norm tuning**: Tracks gradient statistics to adjust clipping
- **Works with any TorchOpt optimizer**: SGD, Adam, AdamW, RMSprop, etc.
- **Optional LR scaling**: Compensate for heavy clipping
- **Same privacy**: No weakening of guarantees

**See also**: [Optimizers & Adaptive Clipping User Guide](../user-guide/optimizers.md)

## Adaptive Clipping

::: opaque.optimizers.adaptive

## Supporting Classes

::: opaque.optimizers.adaptive.clip_buffer
options:
members:
- ClipBuffer

::: opaque.optimizers.adaptive.lr_scheduler
options:
members:
- LRScheduler
