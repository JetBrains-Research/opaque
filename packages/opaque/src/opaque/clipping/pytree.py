"""Core clipping operations for PyTrees."""

from __future__ import annotations

from collections import namedtuple

import torch

from opaque.utils.per_group import PerGroup
from opaque.utils.pytree import global_norm, tree_map

ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm", "group_norms"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
    group_norms: Per-group L2 norms before clipping (dict[str, Tensor]),
        or None when global clipping is used.
"""


def _clip_pytree_per_group(
    pytree: dict[str, torch.Tensor],
    pg: PerGroup,
    return_zero: bool,
) -> tuple[dict[str, torch.Tensor], ClipPytreeAux]:
    """Per-group clipping: each group is clipped to its own L2 norm bound."""
    # 1. Accumulate squared norms per group
    group_sq_norms: dict[str, torch.Tensor] = {}
    for key, tensor in pytree.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        group_name = pg.groups[key]
        sq = (tensor.to(torch.float32) ** 2).sum()
        if group_name in group_sq_norms:
            group_sq_norms[group_name] = group_sq_norms[group_name] + sq
        else:
            group_sq_norms[group_name] = sq

    # 2. Compute per-group scale factors: min(1, norm_bound / norm)
    group_scales: dict[str, torch.Tensor] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = torch.sqrt(sq_norm)
        cn = torch.tensor(pg.values[group_name], dtype=norm.dtype, device=norm.device)
        cn = torch.clamp(cn, min=0.0)
        one = torch.tensor(1.0, device=norm.device)
        zero = torch.tensor(0.0, device=norm.device)
        scale = torch.minimum(one, cn / norm)
        scale = torch.where(torch.isfinite(scale), scale, zero)
        group_scales[group_name] = scale

    # 3. Apply per-group scales
    clipped: dict[str, torch.Tensor] = {}
    for key, val in pytree.items():
        if isinstance(val, torch.Tensor):
            group_name = pg.groups[key]
            scale = group_scales[group_name]
            clipped[key] = scale.to(dtype=val.dtype) * val
        else:
            clipped[key] = val

    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t,
            clipped,
        )

    orig_norm = global_norm(pytree)
    group_norms = {name: torch.sqrt(sq) for name, sq in group_sq_norms.items()}
    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=group_norms)


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clipping_norm: float | PerGroup,
    return_zero: bool = False,
) -> tuple[dict[str, torch.Tensor], ClipPytreeAux]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    NaN and Inf values in the input are replaced with zeros before clipping.
    This is vmap-compatible and DP-safe (the clipped output has norm <= clipping_norm).

    Args:
        pytree: Dictionary of tensors to clip
        clipping_norm: Maximum L2 norm (non-negative, or inf for no clipping).
            When ``PerGroup``, each group of parameters is clipped independently
            to its own norm bound.
        return_zero: If True, the output PyTree is guaranteed to be zero no matter
            what the inputs are. Does not influence the formal guarantees but useful
            for privacy amplification via padding (see https://arxiv.org/pdf/2411.04205).

    Returns:
        Tuple of (clipped_pytree, aux) where aux contains:
            - norm: The L2 norm of the original (unclipped) pytree

    Edge cases:
        - clipping_norm=0: Returns zeros
        - clipping_norm=inf: No clipping (passthrough)
        - pytree_norm=0: Returns unchanged
        - NaN/Inf values: Replaced with zeros before clipping
        - return_zero=True: Returns zeros regardless of other parameters
    """
    # Sanitize NaN/Inf → 0 before clipping.  This is both vmap-compatible
    # (no data-dependent control flow) and DP-safe (zeroed contributions
    # have norm 0, which is within the sensitivity bound).
    pytree = tree_map(
        lambda t: (
            torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(t, torch.Tensor)
            else t
        ),
        pytree,
    )

    # Per-group path: clip each group independently
    if isinstance(clipping_norm, PerGroup):
        return _clip_pytree_per_group(pytree, clipping_norm, return_zero)

    # --- Global (flat) clipping path ---

    # Compute original norm (always finite after sanitization)
    orig_norm = global_norm(pytree)

    # Compute scale factor
    clipping_norm_tensor = torch.tensor(
        clipping_norm, dtype=orig_norm.dtype, device=orig_norm.device
    )
    clipping_norm_tensor = torch.clamp(clipping_norm_tensor, min=0.0)

    # Basic clipping: scale = min(1, clipping_norm / orig_norm)
    scale = torch.minimum(torch.tensor(1.0), clipping_norm_tensor / orig_norm)

    # Handle norm=0 or NaN: set scale to 0
    scale = torch.where(torch.isfinite(scale), scale, torch.tensor(0.0))

    # Apply scale (cast to input dtype to avoid 0D-vs-0D promotion to float32)
    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        return scale.to(dtype=t.dtype) * t

    clipped = tree_map(
        lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree
    )

    # Apply return_zero if requested (for privacy amplification via padding)
    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t, clipped
        )

    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=None)


__all__ = ["clip_pytree", "ClipPytreeAux"]
