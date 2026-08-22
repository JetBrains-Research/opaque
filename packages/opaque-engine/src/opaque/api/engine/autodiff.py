"""Portable, backend-dispatched automatic-differentiation transforms."""

from __future__ import annotations

import threading
from functools import wraps
from typing import TYPE_CHECKING, Any

from opaque.api.engine.backend import _registry, ensure_backend
from opaque.api.engine.primitive import Primitive, PrimitiveTier, primitive

if TYPE_CHECKING:
    from collections.abc import Callable


@primitive(tier=PrimitiveTier.CORE, name="opaque.autodiff.grad_and_value")
def _grad_and_value_transform(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    values_only: bool = False,
) -> Callable[..., Any]:
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE, name="opaque.autodiff.vmap")
def _vmap_transform(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
    randomness: str = "same",
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

    def _transform_for(backend: Any) -> Callable[..., Any]:
        transformed = transforms.get(backend.name)
        if transformed is None:
            with lock:
                transformed = transforms.get(backend.name)
                if transformed is None:
                    factory = operation.resolve(backend)
                    transformed = factory(fn, *transform_args, **transform_kwargs)
                    transforms[backend.name] = transformed
        return transformed

    # When a backend is already active at creation, pin its transform so
    # the per-call hot path is a plain closure call that compiled graphs
    # (torch.compile fullgraph) can trace. In eager execution the pin is
    # used only while the invocation context's active backend is still the
    # one it was built for — otherwise the call re-resolves from its
    # arguments, so a wrapper built under one backend never silently runs
    # a stale transform after ``clear_backend()`` or a context switch.
    # Inside a traced graph, ContextVar reads are untraceable; there the
    # pin is trusted directly, guarded by the same plain globals as the
    # primitive fast path.
    pinned: Callable[..., Any] | None = None
    initial = _registry.active_backend()
    if initial is not None and operation.supports(initial):
        pinned = _transform_for(initial)

    if pinned is not None:
        pinned_transform = pinned
        pinned_name = initial.name

        @wraps(fn)
        def executable(*args: Any, **kwargs: Any) -> Any:
            if _registry._SINGLE_BACKEND and _registry._IS_COMPILING():
                return pinned_transform(*args, **kwargs)
            active = _registry._ACTIVE.get()
            if active is not None and active.name == pinned_name:
                return pinned_transform(*args, **kwargs)
            backend = ensure_backend(args, kwargs)
            return _transform_for(backend)(*args, **kwargs)

        return executable

    @wraps(fn)
    def executable(*args: Any, **kwargs: Any) -> Any:
        backend = ensure_backend(args, kwargs)
        return _transform_for(backend)(*args, **kwargs)

    return executable


def grad_and_value(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    values_only: bool = False,
) -> Callable[..., Any]:
    """Return an executable that differentiates with its invocation backend.

    The executable takes the same arguments as ``fn`` and returns
    ``(grads, value)`` — gradients first. ``grads`` mirrors the structure of
    the differentiated argument, so a pytree of parameters yields a pytree of
    the same shape; with a tuple ``argnums`` it is a tuple of those, in the
    order given. With ``has_aux``, the second element becomes ``(value, aux)``,
    so the whole result is ``(grads, (value, aux))``.

    ``fn`` must return a scalar (or ``(scalar, aux)``), and ``aux`` must be
    arrays or containers of them — providers unwrap it through the transform
    and cannot carry Python scalars out.

    Args:
        fn: The function to differentiate.
        argnums: Which argument(s) to differentiate with respect to. An
            ``int`` gives one gradient; a tuple gives one per entry.
        has_aux: Whether ``fn`` returns ``(value, aux)``.
        values_only: Declares that the caller reads the result as a value
            and will not differentiate it again. Providers that eagerly
            build a differentiable graph may skip that work; providers
            that build the derivative program lazily (trace-based
            autodiff) have nothing to skip and ignore the hint. It is a
            performance hint in both cases, never a change to the returned
            numbers: a provider must still honor an enclosing transform
            that does differentiate the result.

    Returns:
        A callable with ``fn``'s signature returning ``(grads, value)``, or
        ``(grads, (value, aux))`` when ``has_aux`` is set.
    """
    return _deferred_transform(
        _grad_and_value_transform,
        fn,
        argnums=argnums,
        has_aux=has_aux,
        values_only=values_only,
    )


def vmap(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
    randomness: str = "same",
) -> Callable[..., Any]:
    """Return an executable vectorized with its invocation backend.

    ``fn`` sees one slice of each mapped argument and is written as if for a
    single example; the executable maps it over the batch. Arrays created
    inside ``fn`` are per-element, and the mapped axis must have the same
    length on every mapped argument.

    Args:
        fn: The function to vectorize over the mapped axes.
        in_axes: Which axis of each argument to map over. An ``int`` applies
            to every argument; ``None`` broadcasts an argument unmapped; a
            structure matching the arguments (a tuple, or a pytree mirroring
            one) sets them individually.
        out_axes: Where the mapped axis lands in each result. ``0`` (default)
            makes it the leading axis; another ``int`` inserts it at that
            position; a structure matching the outputs sets them individually.
        randomness: How framework-native RNG ops inside ``fn`` behave
            across the mapped batch: ``"same"`` (default) shares one draw
            across batch elements — every element gets *identical* values,
            which is rarely what a mechanism wants — ``"different"`` draws
            independently, and ``"error"`` rejects RNG ops outright.
            Opaque's own randomness is keyed and does not go through an
            ambient generator, so this setting does not reach it: derive a
            key per element instead of relying on ``"different"``. Providers
            whose RNG model is purely key-based may ignore the hint entirely.

    Returns:
        A callable with ``fn``'s signature, accepting batched arguments and
        returning batched results.
    """
    return _deferred_transform(
        _vmap_transform,
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        randomness=randomness,
    )


def vmap_factory() -> Callable[..., Any]:
    """Return the best vmap constructor for the current context.

    With an active backend that implements the transform, this is the
    provider's raw factory — plain code that compiled callers (e.g.
    ``torch.compile(fullgraph=True)``) can trace straight into the
    framework's native transform. Otherwise it is the deferred
    :func:`vmap`, which resolves per invocation.

    Call at transform-construction time (outside any compiled region) and
    close over the result.
    """
    backend = _registry.active_backend()
    if backend is not None and _vmap_transform.supports(backend):
        return _vmap_transform.resolve(backend)
    return vmap


__all__ = ["grad_and_value", "vmap", "vmap_factory"]
