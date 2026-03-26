"""Type definitions for clipping operations."""

from abc import ABC
from dataclasses import dataclass


class ClipState(ABC):
    """Base class for clipping state with clip norm access.

    All clipping operations (fixed and adaptive) return a state object that
    inherits from this class, providing a unified ``clip_norm`` attribute
    for differential privacy noise calibration.

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, ClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Use clip_norm for noise calibration
        >>> from opaque import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1 * clip_state.clip_norm)
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """

    clip_norm: float
    """The effective per-example clip norm used at the current step.

    For fixed clipping this is the constant L2 norm bound.
    For adaptive clipping this is the threshold that was actually applied
    to clip the gradients, **not** the updated threshold for the next step.

    This is the value you multiply by ``noise_multiplier`` to get the
    required noise standard deviation.
    """


@dataclass(frozen=True)
class FixedClipState(ClipState):
    """Clipping state for fixed (non-adaptive) gradient clipping.

    This state is returned by `clipped_grad` and `clipped_fun` for fixed clipping,
    where the clip norm remains constant throughout training.

    Attributes:
        clip_norm: The L2 norm bound after clipping.

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>> grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.5, batch_argnums=(1, 2))
        >>>
        >>> # State is fixed throughout training
        >>> assert clip_state.clip_norm == 1.5
        >>>
        >>> # After gradient computation, state is unchanged
        >>> grads, new_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>> assert new_state.clip_norm == 1.5  # Still the same
    """

    clip_norm: float

    def __post_init__(self):
        """Validate state parameters."""
        if self.clip_norm <= 0:
            raise ValueError(
                f"clip_norm must be positive, got {self.clip_norm}"
            )


__all__ = [
    "ClipState",
    "FixedClipState",
]
