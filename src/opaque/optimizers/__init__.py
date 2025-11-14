"""Functional DP Optimizers using TorchOpt.

This package provides differentially private optimizers built on TorchOpt's
functional optimization framework. All optimizers follow the same pattern:

    init_fn, step_fn = dp_optimizer(...)
    state = init_fn(params)
    params, state, metrics = step_fn(params, grads, grad_norms, state)

Available Optimizers:
    - dp_optimizer_ac: DP optimizer with adaptive clipping (works with any TorchOpt
      optimizer as base, defaults to AdamW)

All optimizers integrate:
    - Gradient clipping (per-example, from Stage 1)
    - Noise injection (Gaussian mechanism, from Stage 2)
    - Adaptive clipping and learning rate scheduling
    - EMA smoothing for better generalization

Note: Privacy accounting is EXTERNAL - users must track privacy budget separately
using opaque.accounting module.
"""

from opaque.optimizers.dp_optimizer_ac import adaptive_clipping
from opaque.optimizers.types import DPAdaptiveClipState

__all__ = [
    # Optimizers
    "adaptive_clipping",
    # States
    "DPAdaptiveClipState",
]
