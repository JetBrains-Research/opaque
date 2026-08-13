"""Portable, backend-dispatched automatic-differentiation transforms."""

from __future__ import annotations

import threading
from functools import wraps
from typing import TYPE_CHECKING, Any

from opaque.api.engine.backend import ensure_backend
from opaque.api.engine.primitive import Primitive, PrimitiveTier, primitive

if TYPE_CHECKING:
    from collections.abc import Callable


@primitive(tier=PrimitiveTier.CORE, name="opaque.autodiff.grad_and_value")
def _grad_and_value_transform(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Any]:
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE, name="opaque.autodiff.vmap")
def _vmap_transform(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
) -> Callable[..., Any]:
    raise NotImplementedError


def _deferred_transform(
    operation: Primitive,
    fn: Callable[..., Any],
    /,
    *transform_args: Any,
    **transform_kwargs: Any,
) -> Callable[..., Any]:
    transforms: dict[str, Callable[..., Any]] = {}
    lock = threading.Lock()

    @wraps(fn)
    def executable(*args: Any, **kwargs: Any) -> Any:
        backend = ensure_backend(args, kwargs)
        transformed = transforms.get(backend.name)
        if transformed is None:
            with lock:
                transformed = transforms.get(backend.name)
                if transformed is None:
                    factory = operation.resolve(backend)
                    transformed = factory(fn, *transform_args, **transform_kwargs)
                    transforms[backend.name] = transformed
        return transformed(*args, **kwargs)

    return executable


def grad_and_value(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Return an executable that differentiates with its invocation backend."""
    return _deferred_transform(
        _grad_and_value_transform,
        fn,
        argnums=argnums,
        has_aux=has_aux,
    )


def vmap(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
) -> Callable[..., Any]:
    """Return an executable vectorized with its invocation backend."""
    return _deferred_transform(
        _vmap_transform,
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
    )


__all__ = ["grad_and_value", "vmap"]
