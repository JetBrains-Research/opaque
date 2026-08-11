"""Core clipping operations for PyTrees."""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.engine.backend import active_backend
from opaque.api.engine.pytree import (
    ParamPath,
    global_norm,
    param_path_display,
)
from opaque.api.engine.types import PerGroup

if TYPE_CHECKING:
    from opaque.api.engine.backend._protocol import Backend

ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm", "group_norms"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
    group_norms: Per-group L2 norms before clipping (dict[str, Tensor]),
        or None when global clipping is used.
"""


def _tensor_path_leaves(
    pytree: Any,
    backend: Backend,
) -> tuple[list[ParamPath], list[torch.Tensor], Any]:
    """Flatten tensor leaves with :data:`~opaque.pytree.ParamPath` keys."""
    paths, leaves, treedef = backend.tree_flatten_with_paths(pytree)
    tensor_leaves: list[torch.Tensor] = []
    for path, leaf in zip(paths, leaves, strict=True):
        if not backend.is_array(leaf):
            raise TypeError(
                f"Expected tensor leaves for per-group clipping; "
                f"got {type(leaf).__name__} at path {path!r}."
            )
        tensor_leaves.append(leaf)
    return paths, tensor_leaves, treedef


def _validate_per_group_paths(
    paths: list[ParamPath],
    pg: PerGroup,
) -> None:
    """Require a 1:1 match between leaf paths and ``pg.groups`` keys."""
    leaf_set = set(paths)
    group_set = set(pg.groups)
    if leaf_set == group_set:
        return
    missing_in_groups = sorted(leaf_set - group_set, key=param_path_display)
    missing_in_tree = sorted(group_set - leaf_set, key=param_path_display)
    parts: list[str] = [
        "PerGroup path keys must match the pytree tensor leaves exactly."
    ]
    if missing_in_groups:
        parts.append(f"Leaves with no group assignment: {missing_in_groups}.")
    if missing_in_tree:
        parts.append(f"Group paths with no matching leaf: {missing_in_tree}.")
    raise ValueError(" ".join(parts))


def _resolve_compute_dtype_for_reduction(
    leaves: list[torch.Tensor],
    compute_dtype: torch.dtype | None,
    backend: Backend,
) -> torch.dtype:
    """Pick the dtype for sum-of-squares reductions over ``leaves``."""
    if compute_dtype is not None:
        return compute_dtype
    acc = torch.float32
    for leaf in leaves:
        if not backend.is_floating(leaf):
            continue
        promoted = (
            torch.float32
            if leaf.dtype in (torch.float16, torch.bfloat16)
            else leaf.dtype
        )
        acc = backend.promote_dtype(acc, promoted)
    return acc


def _accumulate_group_sq_norms(
    paths: list[ParamPath],
    leaves: list[torch.Tensor],
    pg: PerGroup,
    acc_dtype: torch.dtype,
    backend: Backend,
) -> dict[str, torch.Tensor]:
    group_sq_norms: dict[str, torch.Tensor] = {}
    for path, tensor in zip(paths, leaves, strict=True):
        group_name = pg.groups[path]
        sq = backend.sum(backend.square(backend.astype(tensor, acc_dtype)))
        if group_name in group_sq_norms:
            group_sq_norms[group_name] = group_sq_norms[group_name] + sq
        else:
            group_sq_norms[group_name] = sq
    return group_sq_norms


def _scale_leaves_by_group(
    paths: list[ParamPath],
    leaves: list[torch.Tensor],
    pg: PerGroup,
    group_scales: dict[str, torch.Tensor],
    treedef: Any,
    backend: Backend,
) -> Any:
    scaled = [
        backend.astype(group_scales[pg.groups[path]], leaf.dtype) * leaf
        for path, leaf in zip(paths, leaves, strict=True)
    ]
    return backend.tree_unflatten(treedef, scaled)


def _auto_scale_per_group(
    pytree: Any,
    pg: PerGroup,
    gamma: float,
    compute_dtype: torch.dtype | None,
    backend: Backend,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group AUTO-S scaling: each group is scaled to sensitivity R_k."""
    paths, leaves, treedef = _tensor_path_leaves(pytree, backend)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype, backend)

    group_sq_norms = _accumulate_group_sq_norms(paths, leaves, pg, acc_dtype, backend)

    group_scales: dict[str, torch.Tensor] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = backend.sqrt(sq_norm)
        R = backend.scalar(pg.values[group_name], dtype=norm.dtype, like=norm)
        R = backend.clamp(R, lo=0.0)
        gamma_tensor = backend.scalar(gamma, dtype=norm.dtype, like=norm)
        scale = R / (norm + gamma_tensor)
        zero = backend.scalar(0.0, like=norm)
        scale = backend.where(backend.isfinite(scale), scale, zero)
        group_scales[group_name] = scale

    scaled = _scale_leaves_by_group(paths, leaves, pg, group_scales, treedef, backend)

    orig_norm = global_norm(pytree, compute_dtype=compute_dtype)
    group_norms = {name: backend.sqrt(sq) for name, sq in group_sq_norms.items()}
    return scaled, ClipPytreeAux(norm=orig_norm, group_norms=group_norms)


def auto_scale_pytree(
    pytree: Any,
    R: float | PerGroup = 1.0,
    gamma: float = 0.01,
    *,
    compute_dtype: torch.dtype | None = None,
) -> tuple[Any, ClipPytreeAux]:
    r"""AUTO-S automatic scaling of a PyTree (Bu et al., NeurIPS 2023).

    Scales the PyTree by ``R / (\|pytree\| + gamma)`` so the output L2
    norm is clipped by ``R`` for any input. Unlike :func:`clip_pytree`, there
    is no threshold to tune — every example contributes an approximately
    unit-length update, with the effective step size absorbed into the
    learning rate.

    Args:
        pytree: Tensor pytree to scale (flat or nested).
        R: Output sensitivity bound (non-negative). When ``PerGroup``, each
            group is scaled independently to its own bound; leaf paths must
            match ``R.groups`` exactly.
        gamma: Small positive denominator stabilizer :math:`\gamma` (default
            0.01). Must be strictly positive; at ``gamma=0`` this reduces to
            AUTO-V (undefined at zero gradient).
        compute_dtype: Internal accumulation dtype for the L2-norm
            reduction.  ``None`` (default) auto-promotes bf16/fp16 inputs to
            float32.

    Returns:
        Tuple of (scaled_pytree, aux) where ``aux.norm`` is the original L2
        norm and ``aux.group_norms`` carries per-group norms in per-group
        mode.

    Formal guarantee:
        For every input, the output has L2 norm at most ``R`` (scalar case)
        or :math:`\sqrt{\sum_k R_k^2}` (per-group case).

    NaN/Inf values are sanitized to zero before scaling, matching the
    behavior of :func:`clip_pytree`.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    backend = active_backend()

    pytree = backend.tree_map(
        lambda t: backend.nan_to_num(t) if backend.is_array(t) else t,
        pytree,
    )

    if isinstance(R, PerGroup):
        return _auto_scale_per_group(pytree, R, gamma, compute_dtype, backend)

    orig_norm = global_norm(pytree, compute_dtype=compute_dtype)
    R_tensor = backend.scalar(R, dtype=orig_norm.dtype, like=orig_norm)
    R_tensor = backend.clamp(R_tensor, lo=0.0)
    gamma_tensor = backend.scalar(gamma, dtype=orig_norm.dtype, like=orig_norm)

    scale = R_tensor / (orig_norm + gamma_tensor)
    scale = backend.where(backend.isfinite(scale), scale, backend.zeros_like(scale))

    def scale_leaf(t):
        if not backend.is_array(t):
            return t
        return backend.astype(scale, t.dtype) * t

    scaled = backend.tree_map(
        lambda t: scale_leaf(t) if backend.is_array(t) else t, pytree
    )

    return scaled, ClipPytreeAux(norm=orig_norm, group_norms=None)


def _clip_pytree_per_group(
    pytree: Any,
    pg: PerGroup,
    return_zero: bool,
    compute_dtype: torch.dtype | None,
    backend: Backend,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group clipping keyed by optree :data:`~opaque.pytree.ParamPath`."""
    paths, leaves, treedef = _tensor_path_leaves(pytree, backend)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype, backend)

    group_sq_norms = _accumulate_group_sq_norms(paths, leaves, pg, acc_dtype, backend)

    group_scales: dict[str, torch.Tensor] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = backend.sqrt(sq_norm)
        cn = backend.scalar(pg.values[group_name], dtype=norm.dtype, like=norm)
        cn = backend.clamp(cn, lo=0.0)
        one = backend.scalar(1.0, like=norm)
        zero = backend.scalar(0.0, like=norm)
        scale = backend.minimum(one, cn / norm)
        scale = backend.where(backend.isfinite(scale), scale, zero)
        group_scales[group_name] = scale

    clipped = _scale_leaves_by_group(paths, leaves, pg, group_scales, treedef, backend)

    if return_zero:
        clipped = backend.tree_map(
            lambda t: backend.zeros_like(t) if backend.is_array(t) else t,
            clipped,
        )

    orig_norm = global_norm(pytree, compute_dtype=compute_dtype)
    group_norms = {name: backend.sqrt(sq) for name, sq in group_sq_norms.items()}
    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=group_norms)


def clip_pytree(
    pytree: Any,
    clipping_norm: float | PerGroup,
    return_zero: bool = False,
    *,
    compute_dtype: torch.dtype | None = None,
) -> tuple[Any, ClipPytreeAux]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    NaN and Inf values in the input are replaced with zeros before clipping.
    This is vmap-compatible and DP-safe (the clipped output has norm <= clipping_norm).

    Args:
        pytree: Tensor pytree to clip (flat or nested).
        clipping_norm: Maximum L2 norm (non-negative, or inf for no clipping).
            When ``PerGroup``, each group is clipped independently; leaf paths
            must match ``clipping_norm.groups`` exactly.
        return_zero: If True, the output PyTree is guaranteed to be zero no matter
            what the inputs are. Does not influence the formal guarantees but useful
            for privacy amplification via padding (see https://arxiv.org/pdf/2411.04205).
        compute_dtype: Internal accumulation dtype for the L2-norm
            reduction.  ``None`` (default) auto-promotes bf16/fp16 inputs to
            float32.  This keeps the sensitivity bound numerically honest
            under low-precision compute — small per-element contributions
            don't get rounded away during the sum-of-squares.

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
    backend = active_backend()

    pytree = backend.tree_map(
        lambda t: backend.nan_to_num(t) if backend.is_array(t) else t,
        pytree,
    )

    if isinstance(clipping_norm, PerGroup):
        return _clip_pytree_per_group(
            pytree, clipping_norm, return_zero, compute_dtype, backend
        )

    orig_norm = global_norm(pytree, compute_dtype=compute_dtype)

    clipping_norm_tensor = backend.scalar(
        clipping_norm, dtype=orig_norm.dtype, like=orig_norm
    )
    clipping_norm_tensor = backend.clamp(clipping_norm_tensor, lo=0.0)

    scale = backend.minimum(backend.scalar(1.0), clipping_norm_tensor / orig_norm)
    scale = backend.where(backend.isfinite(scale), scale, backend.scalar(0.0))

    def scale_leaf(t):
        if not backend.is_array(t):
            return t
        return backend.astype(scale, t.dtype) * t

    clipped = backend.tree_map(
        lambda t: scale_leaf(t) if backend.is_array(t) else t, pytree
    )

    if return_zero:
        clipped = backend.tree_map(
            lambda t: backend.zeros_like(t) if backend.is_array(t) else t, clipped
        )

    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=None)


__all__ = ["ClipPytreeAux", "auto_scale_pytree", "clip_pytree"]
