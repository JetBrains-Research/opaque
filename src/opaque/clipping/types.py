"""Type definitions for clipping operations."""

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum


class NeighboringRelation(Enum):
    """Differential privacy neighboring relation definitions.

    These define what it means for two datasets to be "neighbors" in the
    differential privacy sense. The choice affects the sensitivity calculation.

    Attributes:
        ADD_OR_REMOVE_ONE: Datasets differ by adding or removing one record.
            This is the standard definition in DP literature. Sensitivity = S.
        REPLACE_ONE: Datasets differ by replacing one record with another.
            Sensitivity is doubled: 2*S (worst case: remove + add).
        REPLACE_SPECIAL: Datasets differ by replacing one record with a special
            "no-op" record. Used in some padding-based schemes. Sensitivity = S.
    """

    ADD_OR_REMOVE_ONE = "add_or_remove_one"
    REPLACE_ONE = "replace_one"
    REPLACE_SPECIAL = "replace_special"


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
        >>> from opaque import gaussian
        >>> noise_fn = gaussian(stddev=1.1 * sensitivity)
        >>> noisy_grads = noise_fn(grads)
    """

    @abstractmethod
    def sensitivity(
        self,
        neighboring_relation: NeighboringRelation = NeighboringRelation.REPLACE_SPECIAL,
    ) -> float:
        """Compute L2 sensitivity for differential privacy noise calibration.

        The L2 sensitivity is the maximum change in L2 norm of the function output
        when applied to neighboring datasets, as defined by the neighboring relation.

        This is the critical value for calibrating DP noise:
            noise_stddev = noise_multiplier * sensitivity

        Args:
            neighboring_relation: The neighboring relation to use. Default is
                REPLACE_SPECIAL, which is commonly used in DP-SGD with Poisson
                sampling.

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
        l2_norm_bound: The L2 norm bound after clipping (1.0 if rescaled, else clip_norm)
        rescale_to_unit_norm: Whether gradients were rescaled to unit norm

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
    rescale_to_unit_norm: bool = False

    def __post_init__(self):
        """Validate state parameters."""
        if self.l2_norm_bound <= 0:
            raise ValueError(
                f"l2_norm_bound must be positive, got {self.l2_norm_bound}"
            )

    def sensitivity(
        self,
        neighboring_relation: NeighboringRelation = NeighboringRelation.REPLACE_SPECIAL,
    ) -> float:
        """Compute L2 sensitivity for DP noise calibration.

        For fixed clipping, sensitivity is always the l2_norm_bound (which is
        1.0 if rescale_to_unit_norm=True, otherwise it's the clip_norm).
        """
        match neighboring_relation:
            case NeighboringRelation.ADD_OR_REMOVE_ONE:
                return self.l2_norm_bound
            case NeighboringRelation.REPLACE_ONE:
                return 2 * self.l2_norm_bound
            case NeighboringRelation.REPLACE_SPECIAL:
                return self.l2_norm_bound
            case _:
                raise ValueError(
                    f"Unsupported neighboring_relation={neighboring_relation}. "
                    f"Must be one of: {list(NeighboringRelation)}"
                )


ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
"""

ClippedFunAux = namedtuple("ClippedFunAux", ["user_aux", "norms"])
"""Auxiliary outputs from clipped_fun.

Fields:
    user_aux: Auxiliary data returned by the user's function (if has_aux=True), else None.
    norms: Per-example L2 norms before clipping (if return_norms=True), else None.
"""

ClippedGradAux = namedtuple("ClippedGradAux", ["loss_values", "grad_norms", "user_aux"])
"""Auxiliary outputs from clipped_grad and adaptive_clipped_grad.

Fields:
    loss_values: Per-example loss values (if return_values=True), else None.
    grad_norms: Per-example gradient L2 norms before clipping (if return_grad_norms=True), else None.
    user_aux: Auxiliary data returned by the user's loss function (if has_aux=True), else None.
"""

# Legacy alias for backward compatibility during migration
AuxiliaryOutput = ClippedGradAux

__all__ = [
    "ClipState",
    "FixedClipState",
    "NeighboringRelation",
    "ClipPytreeAux",
    "ClippedFunAux",
    "ClippedGradAux",
    "AuxiliaryOutput",  # Legacy alias
]
