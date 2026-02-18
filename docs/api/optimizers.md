# Adaptive Clipping

The `opaque.clipping` module provides `adaptive_clipped_grad()`, which computes per-example clipped gradients with an automatically tuned clip norm.

## Overview

Adaptive clipping automatically adjusts the clip norm `C` so that a target fraction of gradients are clipped each step (Andrew et al. 2021). This removes the need to manually tune `C`.

**Key function**: `adaptive_clipped_grad()` - Returns `(grad_fn, clip_state)` with auto-tuning clip norm

**Features**:

- **Automatic clip norm tuning**: Targets a quantile of gradient norms
- **State-based API**: Clip norm adapts via `clip_state` across steps
- **Configurable quantile**: `target_quantile=0.5` clips at the median
- **Privacy-accounted**: Use `acc.adaclip()` to account for quantile estimation cost

**See also**: [Adaptive Clipping User Guide](../user-guide/optimizers.md)

## API

```python
from opaque.clipping import adaptive_clipped_grad

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.5,
    learning_rate=0.2,
    batch_argnums=1,
)

# Training step
grads, clip_state = grad_fn(params, batch, state=clip_state)
print(f"Current clip norm: {clip_state.clip_norm:.4f}")
```
