"""Explicit lifecycle for the MLX distributed group used by Opaque."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_GROUP: Any | None = None


def _group_attribute(group: Any, name: str) -> int:
    value = getattr(group, name)
    return int(value() if callable(value) else value)


def initialize(
    *,
    strict: bool = False,
    backend: str = "any",
    all_gather_factory: Any = None,
) -> Any:
    """Initialize MLX communication, register its group, and return it.

    This is the only Opaque entry point that calls ``mx.distributed.init``.
    Importing this module, selecting the MLX backend, and runtime capability
    probes are intentionally inert.
    """
    group = mx.distributed.init(
        strict=strict,
        backend=backend,
        all_gather_factory=all_gather_factory,
    )
    register_group(group)
    return group


def register_group(group: Any) -> None:
    """Register an already initialized MLX communication group.

    MLX's group object is process-global, but Opaque only issues collectives
    through a group explicitly registered here. This allows applications that
    own initialization to control precisely when Opaque joins their group.
    """
    try:
        rank = _group_attribute(group, "rank")
        size = _group_attribute(group, "size")
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "MLX distributed group must expose integer 'rank' and 'size' attributes."
        ) from error
    if rank < 0 or size < 1 or rank >= size:
        raise ValueError(
            f"Invalid MLX distributed group view: rank={rank}, size={size}."
        )

    global _GROUP
    _GROUP = group


def clear_group() -> None:
    """Stop Opaque from using the registered MLX group.

    This only clears Opaque's registration. MLX does not expose a matching
    group-destruction API, so any global MLX group remains owned by its caller.
    """
    global _GROUP
    _GROUP = None


def _registered_group() -> Any | None:
    """Return Opaque's registered group without initializing MLX."""
    return _GROUP


def _process_group_view() -> tuple[int, int] | None:
    """Return the registered group view for engine process-group discovery."""
    group = _registered_group()
    if group is None:
        return None
    return _group_attribute(group, "rank"), _group_attribute(group, "size")


def _group_rank(group: Any) -> int:
    """Return a group rank across MLX's callable metadata API."""
    return _group_attribute(group, "rank")


def _group_size(group: Any) -> int:
    """Return a group size across MLX's callable metadata API."""
    return _group_attribute(group, "size")


__all__ = ["clear_group", "initialize", "register_group"]
