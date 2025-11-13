"""Functional DP Optimizers using TorchOpt.

This package provides differentially private optimizers built on TorchOpt's
functional optimization framework. All optimizers follow the same pattern:

    init_fn, step_fn = dp_optimizer(...)
    state = init_fn(params)
    params, state, metrics = step_fn(params, grads, state)

Available Optimizers:
    - dp_sgd: Basic DP-SGD with momentum
    - dp_adam: DP-Adam with adaptive learning rates (no weight decay)
    - dp_adamw: DP-AdamW with decoupled weight decay (recommended for LLMs)
    - dp_adam_ac: DP-Adam with adaptive clipping (state-of-the-art, Oct 2024)

All optimizers integrate:
    - Gradient clipping (per-example, from Stage 1)
    - Noise injection (Gaussian mechanism, from Stage 2)
    - Privacy accounting (RDP/PLD, from Stage 2)
"""

from opaque.optimizers.base import DPOptimizerState, make_dp_optimizer
from opaque.optimizers.dp_adam import dp_adam
from opaque.optimizers.dp_adam_ac import AdaptiveClipState, dp_adam_ac
from opaque.optimizers.dp_adamw import dp_adamw
from opaque.optimizers.dp_sgd import dp_sgd

__all__ = [
    # Base
    "DPOptimizerState",
    "make_dp_optimizer",
    # Optimizers
    "dp_sgd",
    "dp_adam",
    "dp_adamw",
    "dp_adam_ac",
    # States
    "AdaptiveClipState",
]
