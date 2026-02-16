"""Type definitions for functional DP optimizers.

This module defines immutable state types for differential privacy optimizers
following functional programming principles.
"""

from typing import Any, NamedTuple

import torch


class DPAdaptiveClipState(NamedTuple):
    """Immutable state for DP optimizer with adaptive clipping.

    This state type replaces mutable components with immutable equivalents:
    - ClipNormBuffer object → tuple[Tensor, int] (norms, size)
    - accountant object → removed (external accounting)
    - RNG state → removed (external noise injection)
    - EMA parameters → removed (external smoothing wrapper)

    All fields are immutable. Updates create new state instances.

    Attributes:
        opt_state: Internal optimizer state from TorchOpt (already immutable NamedTuple)
        clip_buffer_state: Immutable clip buffer state as (norms_tensor, size) tuple
        current_clip_norm: Current adaptive clipping threshold C
        lr_multiplier: Current learning rate multiplier γ (always tracked, used conditionally)
        step: Training step counter

    Example:
        >>> # Initialize state
        >>> state = DPAdaptiveClipState(
        ...     opt_state=torchopt_state,
        ...     clip_buffer_state=(torch.zeros(1000), 0),
        ...     current_clip_norm=1.0,
        ...     lr_multiplier=1.0,
        ...     step=0,
        ... )
        >>>
        >>> # Update creates new state (immutable)
        >>> new_state = state._replace(step=state.step + 1)
        >>> assert state.step == 0  # Original unchanged
        >>> assert new_state.step == 1
    """

    opt_state: Any  # TorchOpt optimizer state (NamedTuple)
    clip_buffer_state: tuple[torch.Tensor, int]  # (norms_tensor, size)
    current_clip_norm: float  # Adaptive threshold C
    lr_multiplier: float  # Learning rate multiplier γ
    step: int  # Step counter


__all__ = ["DPAdaptiveClipState"]
