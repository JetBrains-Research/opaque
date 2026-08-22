"""Shared backend-neutral DP second-moment correction helpers."""

from __future__ import annotations

from typing import Any

from opaque.api.engine.pytree import ParamPath, tree_flatten_with_paths, tree_unflatten
from opaque.types import PerGroup


def init_per_group_phi(params: Any) -> dict[ParamPath, float]:
    paths, _leaves, _spec = tree_flatten_with_paths(params)
    return dict.fromkeys(paths, 0.0)


def is_per_group(noise_stddev: float | PerGroup) -> bool:
    return isinstance(noise_stddev, PerGroup)


def resolve_noise_variance(
    noise_stddev: float | PerGroup, path: ParamPath | None = None
) -> float:
    if isinstance(noise_stddev, PerGroup):
        if path is None:
            raise ValueError("PerGroup noise_stddev requires a parameter path")
        return float(noise_stddev.for_path(path)) ** 2
    return float(noise_stddev) ** 2


def update_phi_ema(
    phi: float | dict[ParamPath, float],
    variance: float | dict[ParamPath, float],
    beta: float,
) -> float | dict[ParamPath, float]:
    if isinstance(phi, dict):
        if not isinstance(variance, dict):
            raise TypeError("per-group phi requires per-group variance")
        return {path: beta * phi[path] + (1.0 - beta) * variance[path] for path in phi}
    return beta * phi + (1.0 - beta) * float(variance)


def map_leaves_with_path(fn: Any, tree: Any, *others: Any) -> Any:
    paths, leaves, spec = tree_flatten_with_paths(tree)
    other_leaves = [tree_flatten_with_paths(value) for value in others]
    for other_paths, flat, _other_spec in other_leaves:
        if other_paths != paths or len(flat) != len(leaves):
            raise ValueError("ParamPath mismatch between optimizer pytrees")
    return tree_unflatten(
        spec,
        [
            fn(path, leaf, *(flat[index] for _, flat, _ in other_leaves))
            for index, (path, leaf) in enumerate(zip(paths, leaves, strict=True))
        ],
    )


__all__ = [
    "init_per_group_phi",
    "is_per_group",
    "map_leaves_with_path",
    "resolve_noise_variance",
    "update_phi_ema",
]
