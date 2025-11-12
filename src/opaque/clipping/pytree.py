"""Core clipping operations for PyTrees."""

import torch

from opaque.utils.pytree import global_norm, tree_map


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
    nan_safe: bool = False,
    return_zero: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    Args:
        pytree: Dictionary of tensors to clip
        clip_norm: Maximum L2 norm (non-negative, or inf for no clipping)
        rescale_to_unit_norm: If True, additionally scale by 1/clip_norm
            so final norm is at most 1.0 regardless of clip_norm value
        nan_safe: If True, replace NaNs/Infs with zeros before clipping
        return_zero: If True, the output PyTree is guaranteed to be zero no matter
            what the inputs are. Does not influence the formal guarantees but useful
            for privacy amplification via padding (see https://arxiv.org/pdf/2411.04205).

    Returns:
        Tuple of (clipped_pytree, original_norm)

    Edge cases:
        - clip_norm=0, rescale=False: Returns zeros
        - clip_norm=0, rescale=True: Returns pytree/norm (unit norm)
        - clip_norm=inf, rescale=False: No clipping (passthrough)
        - clip_norm=inf, rescale=True: Returns zeros
        - pytree_norm=0: Returns unchanged
        - return_zero=True: Returns zeros regardless of other parameters
    """
    # Handle NaN/Inf
    if nan_safe:
        pytree = tree_map(lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), pytree)

    # Compute original norm
    orig_norm = global_norm(pytree)

    # Compute scale factor
    clip_norm_tensor = torch.tensor(clip_norm, dtype=orig_norm.dtype, device=orig_norm.device)
    clip_norm_tensor = torch.clamp(clip_norm_tensor, min=0.0)

    # Basic clipping: scale = min(1, clip_norm / orig_norm)
    scale = torch.minimum(torch.tensor(1.0), clip_norm_tensor / orig_norm)

    # Rescale to unit norm if requested
    if rescale_to_unit_norm:
        # If clip_norm > 0: scale / clip_norm
        # If clip_norm == 0: 1 / orig_norm
        scale = torch.where(clip_norm_tensor > 0, scale / clip_norm_tensor, 1.0 / orig_norm)

    # Handle norm=0 or NaN: set scale to 0
    scale = torch.where(torch.isfinite(scale), scale, torch.tensor(0.0))

    # Apply scale
    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        return scale * t

    clipped = tree_map(lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree)

    # Apply return_zero if requested (for privacy amplification via padding)
    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t, clipped
        )

    return clipped, orig_norm


__all__ = ["clip_pytree"]
