"""Core clipping operations for PyTrees."""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Any

import torch

from opaque.api.engine.pytree import (
    ParamPath,
    param_path_display,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from opaque.api.engine.types import PerGroup

_SQ_NORM_ACCUM_DTYPE = torch.float64
"""Dtype the cross-leaf squared-norm accumulator runs in.

The per-leaf squaring still happens in the caller's ``compute_dtype``.
"""

ClipPytreeAux = namedtuple("ClipPytreeAux", ["norm", "group_norms"])
"""Auxiliary outputs from clip_pytree.

Fields:
    norm: The L2 norm of the original (unclipped) pytree.
    group_norms: Per-group L2 norms before clipping (dict[str, Tensor]),
        or None when global clipping is used.
"""


def _tensor_path_leaves(
    pytree: Any,
) -> tuple[list[ParamPath], list[torch.Tensor], Any]:
    """Flatten tensor leaves with :data:`~opaque.pytree.ParamPath` keys."""
    paths, leaves, treedef = tree_flatten_with_paths(pytree)
    tensor_leaves: list[torch.Tensor] = []
    for path, leaf in zip(paths, leaves, strict=True):
        if not isinstance(leaf, torch.Tensor):
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


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """The real dtype a complex leaf's magnitudes accumulate in."""
    if not dtype.is_complex:
        return dtype
    return torch.tensor((), dtype=dtype).real.dtype


def _resolve_compute_dtype_for_reduction(
    leaves: list[torch.Tensor],
    compute_dtype: torch.dtype | None,
) -> torch.dtype:
    """Pick the dtype for sum-of-squares reductions over ``leaves``."""
    if compute_dtype is not None:
        if not compute_dtype.is_floating_point:
            raise TypeError(
                f"compute_dtype must be a real floating-point dtype, got "
                f"{compute_dtype!r}.  Integer/bool/complex compute dtypes can "
                f"silently corrupt the L2-norm reduction (the squared sum is "
                f"non-negative real and the final sqrt assumes a real "
                f"accumulator)."
            )
        return compute_dtype
    acc = torch.float32
    for leaf in leaves:
        if not (torch.is_floating_point(leaf) or torch.is_complex(leaf)):
            continue
        real = _real_dtype(leaf.dtype)
        promoted = torch.float32 if real in (torch.float16, torch.bfloat16) else real
        acc = torch.promote_types(acc, promoted)
    return acc


def _guard_scale(
    ratio: torch.Tensor,
    storage_dtype: torch.dtype,
    norm_roundoff: float,
) -> torch.Tensor:
    """Shrink a clipping ratio so that storing it cannot break the norm bound.

    Three roundings sit between ``C / ||x||`` and the stored output: the norm
    reduction (bounded by ``norm_roundoff``), the downcast of the scale into
    ``storage_dtype``, and the per-element multiply.  Each can round up, so the
    ratio gives up ``2 (u_store + norm_roundoff)``.  Non-floating leaves carry
    no rounding error and are returned unchanged.

    The absolute term covers subnormals, where round-to-nearest error is not
    bounded by ``u * |x|``; it is sized from ``storage_dtype`` alone.
    """
    if not (storage_dtype.is_floating_point or storage_dtype.is_complex):
        return ratio
    store = torch.finfo(storage_dtype)
    relative = 1.0 - 2.0 * (store.eps / 2.0 + norm_roundoff)
    absolute = store.smallest_normal * store.eps
    return ratio * relative - absolute


def _finalize_scale(
    ratio: torch.Tensor,
    storage_dtype: torch.dtype,
    norm_roundoff: float,
    *,
    clamp_to_one: bool,
) -> torch.Tensor:
    """Guard a ratio for one leaf and reduce it to a usable scale."""
    scale = _guard_scale(ratio, storage_dtype, norm_roundoff)
    if clamp_to_one:
        scale = torch.minimum(torch.ones_like(scale), scale)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    return torch.clamp(scale, min=0.0)


def _sq_accum_dtype(leaves: list[torch.Tensor]) -> torch.dtype:
    """Dtype for the cross-leaf squared-norm accumulator.

    MPS has no float64, so it accumulates in float32 and the guard widens to
    match; every other backend gets the free float64 scalar.
    """
    if any(leaf.device.type == "mps" for leaf in leaves):
        return torch.float32
    return _SQ_NORM_ACCUM_DTYPE


_BLOCKED_REDUCTION_MIN = 4096
"""Element count above which a leaf's sum of squares is reduced in two stages.

A flat reduction admits ``numel`` sequential roundings; splitting it into
``sqrt(numel)`` blocks admits ``2 sqrt(numel)``, which is what keeps the guard
affordable on the float32 accumulator MPS is limited to.  Below the threshold
the flat term is already negligible and the extra kernel is not worth it.
"""


def _reduction_terms(leaves: list[torch.Tensor], sq_dtype: torch.dtype) -> int:
    """Worst-case sequential additions inside a single leaf's reduction."""
    widest = max((math.prod(leaf.shape) for leaf in leaves), default=0)
    if widest <= 1:
        return 0
    if widest > _BLOCKED_REDUCTION_MIN:
        return 2 * math.isqrt(widest) + 2
    return widest


def _norm_roundoff(
    acc_dtype: torch.dtype, sq_dtype: torch.dtype, n_leaves: int, n_reduction: int
) -> float:
    """Relative error bound on the computed norm.

    Three sources: squaring each element in ``acc_dtype``, the reduction inside
    the widest leaf (``n_reduction`` additions, see :func:`_reduction_terms`),
    and the cross-leaf accumulation (``n_leaves``).  Halved because the error is
    carried through a square root.
    """
    u_acc = torch.finfo(acc_dtype).eps / 2.0 if acc_dtype.is_floating_point else 0.0
    u_sq = torch.finfo(sq_dtype).eps / 2.0 if sq_dtype.is_floating_point else 0.0
    return (u_acc + (max(n_leaves, 1) + n_reduction) * u_sq) / 2.0


def _leaf_sq_sum(
    leaf: torch.Tensor, acc_dtype: torch.dtype, sq_dtype: torch.dtype
) -> torch.Tensor:
    """Sum of squares of one leaf, reduced in ``sq_dtype``.

    Complex leaves contribute ``real^2 + imag^2``; casting them to a real
    accumulator would drop the imaginary part and under-read the norm.
    """
    if torch.is_complex(leaf):
        real = leaf.real.to(acc_dtype)
        imag = leaf.imag.to(acc_dtype)
        sq = real * real + imag * imag
    else:
        x = leaf.to(acc_dtype)
        sq = x * x

    flat = sq.reshape(-1)
    n = flat.shape[0]
    if n <= _BLOCKED_REDUCTION_MIN:
        return flat.sum(dtype=sq_dtype)

    block = math.isqrt(n)
    main = (n // block) * block
    total = (
        flat[:main].reshape(-1, block).sum(dim=-1, dtype=sq_dtype).sum(dtype=sq_dtype)
    )
    if main < n:
        total = total + flat[main:].sum(dtype=sq_dtype)
    return total


def _sq_norm(
    leaves: list[torch.Tensor], acc_dtype: torch.dtype, sq_dtype: torch.dtype
) -> torch.Tensor:
    """Sum of squares over ``leaves``, accumulated in ``sq_dtype``."""
    total: torch.Tensor | None = None
    for leaf in leaves:
        sq = _leaf_sq_sum(leaf, acc_dtype, sq_dtype)
        total = sq if total is None else total + sq
    if total is None:
        return torch.zeros((), dtype=sq_dtype)
    return total


def _accumulate_group_sq_norms(
    paths: list[ParamPath],
    leaves: list[torch.Tensor],
    pg: PerGroup,
    acc_dtype: torch.dtype,
    sq_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    group_sq_norms: dict[str, torch.Tensor] = {}
    for path, tensor in zip(paths, leaves, strict=True):
        group_name = pg.groups[path]
        sq = _leaf_sq_sum(tensor, acc_dtype, sq_dtype)
        if group_name in group_sq_norms:
            group_sq_norms[group_name] = group_sq_norms[group_name] + sq
        else:
            group_sq_norms[group_name] = sq
    return group_sq_norms


def _scale_leaves_by_group(
    paths: list[ParamPath],
    leaves: list[torch.Tensor],
    pg: PerGroup,
    group_ratios: dict[str, torch.Tensor],
    treedef: Any,
    norm_roundoff: float,
    *,
    clamp_to_one: bool,
) -> Any:
    scaled = [
        _finalize_scale(
            group_ratios[pg.groups[path]],
            leaf.dtype,
            norm_roundoff,
            clamp_to_one=clamp_to_one,
        ).to(dtype=leaf.dtype)
        * leaf
        for path, leaf in zip(paths, leaves, strict=True)
    ]
    return tree_unflatten(treedef, scaled)


def _auto_scale_per_group(
    pytree: Any,
    pg: PerGroup,
    gamma: float,
    compute_dtype: torch.dtype | None,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group AUTO-S scaling: each group is scaled to sensitivity R_k."""
    paths, leaves, treedef = _tensor_path_leaves(pytree)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms(leaves, sq_dtype)
    )

    group_sq_norms = _accumulate_group_sq_norms(paths, leaves, pg, acc_dtype, sq_dtype)

    group_ratios: dict[str, torch.Tensor] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = torch.sqrt(sq_norm)
        R = torch.tensor(pg.values[group_name], dtype=norm.dtype, device=norm.device)
        R = torch.clamp(R, min=0.0)
        gamma_tensor = torch.tensor(gamma, dtype=norm.dtype, device=norm.device)
        group_ratios[group_name] = R / (norm + gamma_tensor)

    scaled = _scale_leaves_by_group(
        paths, leaves, pg, group_ratios, treedef, roundoff, clamp_to_one=False
    )

    orig_norm = torch.sqrt(_sq_norm(leaves, acc_dtype, sq_dtype)).to(acc_dtype)
    group_norms = {
        name: torch.sqrt(sq).to(acc_dtype) for name, sq in group_sq_norms.items()
    }
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
        lambda t: (
            torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(t, torch.Tensor)
            else t
        ),
        pytree,
    )

    if isinstance(R, PerGroup):
        return _auto_scale_per_group(pytree, R, gamma, compute_dtype)

    leaves = [leaf for leaf in tree_leaves(pytree) if isinstance(leaf, torch.Tensor)]
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms(leaves, sq_dtype)
    )

    norm = torch.sqrt(_sq_norm(leaves, acc_dtype, sq_dtype))
    orig_norm = norm.to(acc_dtype)

    R_tensor = torch.clamp(
        torch.tensor(R, dtype=norm.dtype, device=norm.device), min=0.0
    )
    gamma_tensor = torch.tensor(gamma, dtype=norm.dtype, device=norm.device)
    ratio = R_tensor / (norm + gamma_tensor)

    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        scale = _finalize_scale(ratio, t.dtype, roundoff, clamp_to_one=False)
        return scale.to(dtype=t.dtype) * t

    scaled = tree_map(
        lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree
    )

    return scaled, ClipPytreeAux(norm=orig_norm, group_norms=None)


def _clip_pytree_per_group(
    pytree: Any,
    pg: PerGroup,
    return_zero: bool,
    compute_dtype: torch.dtype | None,
) -> tuple[Any, ClipPytreeAux]:
    """Per-group clipping keyed by optree :data:`~opaque.pytree.ParamPath`."""
    paths, leaves, treedef = _tensor_path_leaves(pytree)
    _validate_per_group_paths(paths, pg)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms(leaves, sq_dtype)
    )

    group_sq_norms = _accumulate_group_sq_norms(paths, leaves, pg, acc_dtype, sq_dtype)

    group_ratios: dict[str, torch.Tensor] = {}
    for group_name, sq_norm in group_sq_norms.items():
        norm = torch.sqrt(sq_norm)
        cn = torch.tensor(pg.values[group_name], dtype=norm.dtype, device=norm.device)
        cn = torch.clamp(cn, min=0.0)
        group_ratios[group_name] = cn / norm

    clipped = _scale_leaves_by_group(
        paths, leaves, pg, group_ratios, treedef, roundoff, clamp_to_one=True
    )

    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t,
            clipped,
        )

    orig_norm = torch.sqrt(_sq_norm(leaves, acc_dtype, sq_dtype)).to(acc_dtype)
    group_norms = {
        name: torch.sqrt(sq).to(acc_dtype) for name, sq in group_sq_norms.items()
    }
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
        lambda t: (
            torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(t, torch.Tensor)
            else t
        ),
        pytree,
    )

    if isinstance(clipping_norm, PerGroup):
        return _clip_pytree_per_group(pytree, clipping_norm, return_zero, compute_dtype)

    leaves = [leaf for leaf in tree_leaves(pytree) if isinstance(leaf, torch.Tensor)]
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms(leaves, sq_dtype)
    )

    sq_norm = _sq_norm(leaves, acc_dtype, sq_dtype)
    norm = torch.sqrt(sq_norm)
    orig_norm = norm.to(acc_dtype)

    clipping_norm_tensor = torch.tensor(
        clipping_norm, dtype=norm.dtype, device=norm.device
    )
    clipping_norm_tensor = torch.clamp(clipping_norm_tensor, min=0.0)
    ratio = clipping_norm_tensor / norm

    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        scale = _finalize_scale(ratio, t.dtype, roundoff, clamp_to_one=True)
        return scale.to(dtype=t.dtype) * t

    clipped = tree_map(
        lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree
    )

    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t, clipped
        )

    return clipped, ClipPytreeAux(norm=orig_norm, group_norms=None)


__all__ = ["ClipPytreeAux", "auto_scale_pytree", "clip_pytree"]
