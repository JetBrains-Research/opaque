"""Core clipping operations for PyTrees."""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.pytree import (
    ParamPath,
    _resolve_reduction_dtype,
    _squared_l2_norm_roundoff,
    _squared_l2_norms,
    param_path_display,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from opaque.api.engine.types import PerGroup

ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm", "group_norms"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
    group_norms: Per-group L2 norms before clipping (dict[str, Tensor]),
        or None when global clipping is used.
"""


def _tensor_path_leaves(
    pytree: Any,
) -> tuple[list[ParamPath], list[Any], Any]:
    """Flatten tensor leaves with :data:`~opaque.pytree.types.ParamPath` keys."""
    paths, leaves, treedef = tree_flatten_with_paths(pytree)
    tensor_leaves: list[Any] = []
    for path, leaf in zip(paths, leaves, strict=True):
        if not ops.is_array(leaf):
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


def _guard_scale(
    ratio: Any,
    storage_dtype: Any,
    norm_roundoff: float,
) -> Any:
    """Shrink a clipping ratio so that storing it cannot break the norm bound.

    Three roundings sit between ``C / ||x||`` and the stored output: the norm
    reduction (bounded by ``norm_roundoff``), the downcast of the scale into
    ``storage_dtype``, and the per-element multiply.  Each can round up, so the
    ratio gives up ``2 (u_store + norm_roundoff)``.  Non-floating leaves carry
    no rounding error and are returned unchanged.

    The absolute term covers subnormals, where round-to-nearest error is not
    bounded by ``u * |x|``; it is sized from ``storage_dtype`` alone.
    """
    if not (ops.is_floating(storage_dtype) or ops.is_complex(storage_dtype)):
        return ratio
    real = ops.real_dtype(storage_dtype)
    eps = ops.finfo_eps(real)
    relative = 1.0 - 2.0 * (eps / 2.0 + norm_roundoff)
    absolute = ops.finfo_smallest_normal(real) * eps
    return ops.subtract(ops.multiply(ratio, relative), absolute)


def _finalize_scale(
    ratio: Any,
    storage_dtype: Any,
    norm_roundoff: float,
    *,
    clamp_to_one: bool,
) -> Any:
    """Guard a ratio for one leaf and reduce it to a usable scale."""
    scale = _guard_scale(ratio, storage_dtype, norm_roundoff)
    if clamp_to_one:
        scale = ops.minimum(ops.ones_like(scale), scale)
    scale = ops.where(ops.isfinite(scale), scale, ops.zeros_like(scale))
    return ops.clamp(scale, lo=0.0)


def _accumulate_group_sq_norms(
    paths: list[ParamPath],
    leaves: list[Any],
    pg: PerGroup,
    acc_dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    if not leaves:
        return ops.scalar(0.0, dtype=acc_dtype), {}
    group_names = [pg.groups[path] for path in paths]
    return _squared_l2_norms(leaves, group_names, dtype=acc_dtype)


def _scale_leaves_by_group(
    paths: list[ParamPath],
    leaves: list[Any],
    pg: PerGroup,
    group_ratios: dict[str, Any],
    treedef: Any,
    norm_roundoff: float,
    *,
    clamp_to_one: bool,
) -> Any:
    scaled = [
        ops.multiply(
            ops.astype(
                _finalize_scale(
                    group_ratios[pg.groups[path]],
                    ops.dtype(leaf),
                    norm_roundoff,
                    clamp_to_one=clamp_to_one,
                ),
                ops.dtype(leaf),
            ),
            leaf,
        )
        for path, leaf in zip(paths, leaves, strict=True)
    ]
    return tree_unflatten(treedef, scaled)


def _auto_scale_per_group(
    pytree: Any,
    pg: PerGroup,
    gamma: float,
    compute_dtype: Any | None,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group AUTO-S scaling: each group is scaled to sensitivity R_k."""
    paths, leaves, treedef = _tensor_path_leaves(pytree)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_reduction_dtype(leaves, compute_dtype)
    roundoff = _squared_l2_norm_roundoff(leaves, dtype=acc_dtype)

    total_sq_norm, group_sq_norms = _accumulate_group_sq_norms(
        paths, leaves, pg, acc_dtype
    )

    group_ratios: dict[str, Any] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = ops.sqrt(sq_norm)
        R = ops.clamp(
            ops.scalar(pg.values[group_name], dtype=ops.dtype(norm), like=norm), lo=0.0
        )
        gamma_tensor = ops.scalar(gamma, dtype=ops.dtype(norm), like=norm)
        group_ratios[group_name] = ops.divide(R, ops.add(norm, gamma_tensor))

    scaled = _scale_leaves_by_group(
        paths, leaves, pg, group_ratios, treedef, roundoff, clamp_to_one=False
    )

    orig_norm = ops.astype(ops.sqrt(total_sq_norm), acc_dtype)
    group_norms = {
        name: ops.astype(ops.sqrt(sq), acc_dtype) for name, sq in group_sq_norms.items()
    }
    return scaled, ClipPytreeAux(norm=orig_norm, group_norms=group_norms)


def auto_scale_pytree(
    pytree: Any,
    R: float | PerGroup = 1.0,
    gamma: float = 0.01,
    *,
    compute_dtype: Any | None = None,
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
        or :math:`\sqrt{\sum_k R_k^2}` (per-group case).  The bound holds on
        the values as *stored*: each leaf's scale is shrunk by a few ULPs of
        that leaf's own dtype, so neither downcasting it nor the per-element
        multiply can round the result back above ``R``.  Unlike
        :func:`clip_pytree` there is no ``min(1, ...)``, so the shrink applies
        to every input — ~0.8% of the output magnitude at bfloat16, ~0.1% at
        float16 and ~1.2e-7 at float32.

    NaN/Inf values are sanitized to zero before scaling, matching the
    behavior of :func:`clip_pytree`.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    pytree = tree_map(
        lambda t: ops.nan_to_num(t) if ops.is_array(t) else t,
        pytree,
    )

    if isinstance(R, PerGroup):
        return _auto_scale_per_group(pytree, R, gamma, compute_dtype)

    leaves = [leaf for leaf in tree_leaves(pytree) if ops.is_array(leaf)]
    acc_dtype = _resolve_reduction_dtype(leaves, compute_dtype)
    if not leaves:
        orig_norm = ops.scalar(0.0, dtype=acc_dtype)
        return pytree, ClipPytreeAux(norm=orig_norm, group_norms=None)
    roundoff = _squared_l2_norm_roundoff(leaves, dtype=acc_dtype)

    total_sq_norm, _ = _squared_l2_norms(leaves, None, dtype=acc_dtype)
    norm = ops.sqrt(total_sq_norm)
    orig_norm = ops.astype(norm, acc_dtype)

    R_tensor = ops.clamp(ops.scalar(R, dtype=ops.dtype(norm), like=norm), lo=0.0)
    gamma_tensor = ops.scalar(gamma, dtype=ops.dtype(norm), like=norm)
    ratio = ops.divide(R_tensor, ops.add(norm, gamma_tensor))

    def scale_leaf(t):
        if not ops.is_array(t):
            return t
        scale = _finalize_scale(ratio, ops.dtype(t), roundoff, clamp_to_one=False)
        return ops.multiply(ops.astype(scale, ops.dtype(t)), t)

    scaled = tree_map(scale_leaf, pytree)

    return scaled, ClipPytreeAux(norm=orig_norm, group_norms=None)


def _clip_pytree_per_group(
    pytree: Any,
    pg: PerGroup,
    return_zero: bool,
    compute_dtype: Any | None,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group clipping keyed by optree :data:`~opaque.pytree.types.ParamPath`."""
    paths, leaves, treedef = _tensor_path_leaves(pytree)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_reduction_dtype(leaves, compute_dtype)
    roundoff = _squared_l2_norm_roundoff(leaves, dtype=acc_dtype)

    total_sq_norm, group_sq_norms = _accumulate_group_sq_norms(
        paths, leaves, pg, acc_dtype
    )

    group_ratios: dict[str, Any] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = ops.sqrt(sq_norm)
        cn = ops.clamp(
            ops.scalar(pg.values[group_name], dtype=ops.dtype(norm), like=norm), lo=0.0
        )
        group_ratios[group_name] = ops.divide(cn, norm)

    clipped = _scale_leaves_by_group(
        paths, leaves, pg, group_ratios, treedef, roundoff, clamp_to_one=True
    )

    if return_zero:
        clipped = tree_map(
            lambda t: ops.zeros_like(t) if ops.is_array(t) else t, clipped
        )

    orig_norm = ops.astype(ops.sqrt(total_sq_norm), acc_dtype)
    group_norms = {
        name: ops.astype(ops.sqrt(sq), acc_dtype) for name, sq in group_sq_norms.items()
    }
    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=group_norms)


def clip_pytree(
    pytree: Any,
    clipping_norm: float | PerGroup,
    return_zero: bool = False,
    *,
    compute_dtype: Any | None = None,
) -> tuple[Any, ClipPytreeAux]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    NaN and Inf values in the input are replaced with zeros before clipping.
    This is vmap-compatible and DP-safe: ``norm(output) <= clipping_norm`` holds
    on the values as *stored*, not just in exact arithmetic.  The norm reduction,
    the downcast of the scale into a leaf's dtype, and the per-element multiply
    all round to nearest and so can round *up*; each leaf's scale is shrunk by a
    few ULPs of its own dtype to absorb that.  The shrink is applied before the
    ``min(1, ...)``, so an input well inside the threshold is returned unchanged;
    one whose norm sits within a few ULPs of ``clipping_norm`` is scaled by that
    much.  For clipped inputs the cost is ~0.8% of the output magnitude at
    bfloat16, ~0.1% at float16 and ~1.2e-7 at float32.

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
    pytree = tree_map(
        lambda t: ops.nan_to_num(t) if ops.is_array(t) else t,
        pytree,
    )

    if isinstance(clipping_norm, PerGroup):
        return _clip_pytree_per_group(pytree, clipping_norm, return_zero, compute_dtype)

    leaves = [leaf for leaf in tree_leaves(pytree) if ops.is_array(leaf)]
    acc_dtype = _resolve_reduction_dtype(leaves, compute_dtype)
    if not leaves:
        orig_norm = ops.scalar(0.0, dtype=acc_dtype)
        return pytree, ClipPytreeAux(norm=orig_norm, group_norms=None)
    roundoff = _squared_l2_norm_roundoff(leaves, dtype=acc_dtype)

    total_sq_norm, _ = _squared_l2_norms(leaves, None, dtype=acc_dtype)
    norm = ops.sqrt(total_sq_norm)
    orig_norm = ops.astype(norm, acc_dtype)

    clipping_norm_tensor = ops.clamp(
        ops.scalar(clipping_norm, dtype=ops.dtype(norm), like=norm), lo=0.0
    )
    ratio = ops.divide(clipping_norm_tensor, norm)

    def scale_leaf(t):
        if not ops.is_array(t):
            return t
        scale = _finalize_scale(ratio, ops.dtype(t), roundoff, clamp_to_one=True)
        return ops.multiply(ops.astype(scale, ops.dtype(t)), t)

    clipped = tree_map(scale_leaf, pytree)

    if return_zero:
        clipped = tree_map(
            lambda t: ops.zeros_like(t) if ops.is_array(t) else t, clipped
        )

    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=None)


__all__ = ["ClipPytreeAux", "auto_scale_pytree", "clip_pytree"]
