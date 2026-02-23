"""Type definitions for clipping operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ClipState(ABC):
    """Base class for clipping state with L2 sensitivity computation.

    This abstract base class defines the interface for clipping state objects
    that provide L2 sensitivity information for differential privacy noise calibration.

    All clipping operations (fixed and adaptive) return a state object that
    implements this interface, providing a unified API for computing sensitivity.

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, ClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Compute sensitivity for noise calibration
        >>> sensitivity = clip_state.sensitivity()  # 1.0
        >>> from opaque import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1 * sensitivity)
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """

    @abstractmethod
    def sensitivity(self) -> float:
        """Compute L2 sensitivity for differential privacy noise calibration.

        The L2 sensitivity is the maximum change in L2 norm of the function output
        when one record is added or removed from the dataset.

        This is the critical value for calibrating DP noise:
            noise_stddev = noise_multiplier * sensitivity

        For replace-one neighboring, double this value when calibrating noise.

        Returns:
            The L2 sensitivity (float). This is what you multiply by noise_multiplier
            to get the required noise standard deviation.
        """
        pass


@dataclass(frozen=True)
class FixedClipState(ClipState):
    """Clipping state for fixed (non-adaptive) gradient clipping.

    This state is returned by `clipped_grad` and `clipped_fun` for fixed clipping,
    where the clip norm and sensitivity remain constant throughout training.

    Attributes:
        l2_norm_bound: The L2 norm bound after clipping

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>> grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.5, batch_argnums=(1, 2))
        >>>
        >>> # State is fixed throughout training
        >>> assert clip_state.l2_norm_bound == 1.5
        >>> assert clip_state.sensitivity() == 1.5
        >>>
        >>> # After gradient computation, state is unchanged
        >>> grads, new_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>> assert new_state.l2_norm_bound == 1.5  # Still the same
    """

    l2_norm_bound: float

    def __post_init__(self):
        """Validate state parameters."""
        if self.l2_norm_bound <= 0:
            raise ValueError(
                f"l2_norm_bound must be positive, got {self.l2_norm_bound}"
            )

    def sensitivity(self) -> float:
        """Compute L2 sensitivity for DP noise calibration.

        For fixed clipping, sensitivity is always the l2_norm_bound.
        """
        return self.l2_norm_bound


__all__ = [
    "ClipState",
    "FixedClipState",
]
