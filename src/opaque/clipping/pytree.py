"""Core clipping operations for PyTrees."""

from collections import namedtuple

import torch

from opaque.utils.pytree import global_norm, tree_map

ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
"""


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clip_norm: float,
    return_zero: bool = False,
) -> tuple[dict[str, torch.Tensor], ClipPytreeAux]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    Raises ValueError if any tensor contains NaN or Inf values, since these
    would break the DP sensitivity bound.

    Args:
        pytree: Dictionary of tensors to clip
        clip_norm: Maximum L2 norm (non-negative, or inf for no clipping)
        return_zero: If True, the output PyTree is guaranteed to be zero no matter
            what the inputs are. Does not influence the formal guarantees but useful
            for privacy amplification via padding (see https://arxiv.org/pdf/2411.04205).

    Returns:
        Tuple of (clipped_pytree, aux) where aux contains:
            - norm: The L2 norm of the original (unclipped) pytree

    Raises:
        ValueError: If any tensor contains NaN or Inf values.

    Edge cases:
        - clip_norm=0: Returns zeros
        - clip_norm=inf: No clipping (passthrough)
        - pytree_norm=0: Returns unchanged
        - return_zero=True: Returns zeros regardless of other parameters
    """
    # Compute original norm
    orig_norm = global_norm(pytree)

    # Check for NaN/Inf — these break the sensitivity bound
    if not torch.isfinite(orig_norm):
        raise ValueError(
            "Per-example gradient contains NaN or Inf values. "
            "This breaks the DP sensitivity bound. "
            "Fix the loss function to avoid producing NaN/Inf gradients."
        )

    # Compute scale factor
    clip_norm_tensor = torch.tensor(
        clip_norm, dtype=orig_norm.dtype, device=orig_norm.device
    )
    clip_norm_tensor = torch.clamp(clip_norm_tensor, min=0.0)

    # Basic clipping: scale = min(1, clip_norm / orig_norm)
    scale = torch.minimum(torch.tensor(1.0), clip_norm_tensor / orig_norm)

    # Handle norm=0 or NaN: set scale to 0
    scale = torch.where(torch.isfinite(scale), scale, torch.tensor(0.0))

    # Apply scale
    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        return scale * t

    clipped = tree_map(
        lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree
    )

    # Apply return_zero if requested (for privacy amplification via padding)
    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t, clipped
        )

    return clipped, ClipPytreeAux(norm=orig_norm)


__all__ = ["clip_pytree", "ClipPytreeAux"]
