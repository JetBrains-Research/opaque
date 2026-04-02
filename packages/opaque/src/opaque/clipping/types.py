"""Type definitions for clipping operations."""

from abc import ABC
from dataclasses import dataclass


class ClipState(ABC):
    """Base class for clipping state with clipping norm and sensitivity.

    All clipping operations (fixed and adaptive) return a state object that
    inherits from this class, providing a unified ``clipping_norm`` attribute
    (the raw clipping threshold), ``normalize_by`` divisor, and a
    ``sensitivity`` property (the L2 sensitivity of the query).

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, ClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Use sensitivity for noise calibration
        >>> from opaque import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1 * clip_state.sensitivity)
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """

    clipping_norm: float
    """The raw per-example clipping norm used at the current step.

    For fixed clipping this is the constant L2 norm bound.
    For adaptive clipping this is the threshold that was actually applied
    to clip the gradients, **not** the updated threshold for the next step.
    """

    normalize_by: float
    """Divisor applied to the clipped gradient sum.

    When ``normalize_by > 1`` the clipped sum is divided by this value,
    reducing the L2 sensitivity accordingly.  Defaults to ``1.0``
    (no averaging).
    """

    @property
    def sensitivity(self) -> float:
        """L2 sensitivity of the clipped query.

        Equal to ``clipping_norm / normalize_by`` — the maximum L2 change
        in the output when one record is added or removed.

        This is the value you multiply by ``noise_multiplier`` to get
        the required noise standard deviation.
        """
        return self.clipping_norm / self.normalize_by


@dataclass(frozen=True)
class FixedClipState(ClipState):
    """Clipping state for fixed (non-adaptive) gradient clipping.

    This state is returned by `clipped_grad` and `clipped_fun` for fixed clipping,
    where the clipping norm remains constant throughout training.

    Attributes:
        clipping_norm: The L2 norm bound after clipping.
        normalize_by: Divisor applied to the clipped sum (1.0 = no averaging).

    Example:
        >>> from opaque import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>> grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.5, batch_argnums=(1, 2))
        >>>
        >>> # State is fixed throughout training
        >>> assert clip_state.clipping_norm == 1.5
        >>> assert clip_state.sensitivity == 1.5  # normalize_by defaults to 1.0
        >>>
        >>> # After gradient computation, state is unchanged
        >>> grads, new_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>> assert new_state.clipping_norm == 1.5  # Still the same
    """

    clipping_norm: float
    normalize_by: float = 1.0

    def __post_init__(self):
        """Validate state parameters."""
        if self.clipping_norm <= 0:
            raise ValueError(
                f"clipping_norm must be positive, got {self.clipping_norm}"
            )
        if self.normalize_by <= 0:
            raise ValueError(f"normalize_by must be positive, got {self.normalize_by}")


__all__ = [
    "ClipState",
    "FixedClipState",
]
