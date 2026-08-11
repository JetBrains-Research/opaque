"""The :class:`Backend` protocol — the five-primitive compute surface.

A backend abstracts the small set of numerical primitives the DP clipping
compute needs, so the algorithms can be expressed once and run against
different array frameworks (PyTorch today, MLX or others later).

The surface is deliberately small and organized into five primitive groups:

1. **autodiff** — :meth:`Backend.value_and_grad`.
2. **vectorization** — :meth:`Backend.vmap`.
3. **pytree** — :meth:`Backend.tree_map`, :meth:`Backend.tree_flatten`,
   :meth:`Backend.tree_flatten_with_paths`, :meth:`Backend.tree_unflatten`,
   :meth:`Backend.tree_leaves`.
4. **array math** — elementwise / reduction / dtype helpers
   (:meth:`Backend.sqrt`, :meth:`Backend.square`, :meth:`Backend.sum`,
   :meth:`Backend.minimum`, :meth:`Backend.maximum`, :meth:`Backend.where`,
   :meth:`Backend.isfinite`, :meth:`Backend.nan_to_num`,
   :meth:`Backend.clamp`, :meth:`Backend.zeros_like`,
   :meth:`Backend.concatenate`, :meth:`Backend.astype`,
   :meth:`Backend.scalar`, :meth:`Backend.is_array`,
   :meth:`Backend.is_floating`, :meth:`Backend.promote_dtype`).
5. **rng** — :meth:`Backend.generator`, :meth:`Backend.normal`.

The protocol is structural: any object exposing these members (and a
``name`` attribute) satisfies it, no explicit subclassing required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


@runtime_checkable
class Backend(Protocol):
    """Structural protocol declaring the five primitive groups.

    Implementations delegate to a concrete array framework.  Method bodies
    must stay thin, side-effect-free pass-throughs so the array-math
    primitives remain traceable under differentiation and vectorization.
    """

    name: str

    # --- autodiff ---
    def value_and_grad(
        self,
        fn: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """Transform ``fn`` into a function returning its gradient and value."""
        ...

    # --- vectorization ---
    def vmap(
        self,
        fn: Callable[..., Any],
        in_axes: Any = 0,
        out_axes: Any = 0,
        randomness: str = "error",
    ) -> Callable[..., Any]:
        """Vectorize ``fn`` over the leading axes given by ``in_axes``."""
        ...

    # --- pytree ---
    def tree_map(self, fn: Callable[..., Any], *trees: Any) -> Any:
        """Apply ``fn`` to corresponding leaves of one or more pytrees."""
        ...

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        """Flatten ``tree`` into ``(leaves, treedef)``."""
        ...

    def tree_flatten_with_paths(self, tree: Any) -> tuple[list[Any], list[Any], Any]:
        """Flatten ``tree`` into ``(paths, leaves, treedef)``."""
        ...

    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        """Rebuild a pytree from ``treedef`` and ``leaves``."""
        ...

    def tree_leaves(self, tree: Any) -> list[Any]:
        """Return the array leaves of ``tree``."""
        ...

    # --- array math (elementwise + reduction + dtype helpers) ---
    def is_array(self, x: Any) -> bool:
        """Return ``True`` if ``x`` is a backend array."""
        ...

    def is_floating(self, x: Any) -> bool:
        """Return ``True`` if ``x`` (an array or dtype) is floating point."""
        ...

    def sqrt(self, x: Any) -> Any:
        """Elementwise square root."""
        ...

    def square(self, x: Any) -> Any:
        """Elementwise square."""
        ...

    def sum(self, x: Any, axis: Any = None, dtype: Any = None) -> Any:
        """Reduce-sum over ``axis`` (all elements when ``axis`` is ``None``)."""
        ...

    def minimum(self, a: Any, b: Any) -> Any:
        """Elementwise minimum."""
        ...

    def maximum(self, a: Any, b: Any) -> Any:
        """Elementwise maximum."""
        ...

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        """Elementwise select ``a`` where ``cond`` else ``b``."""
        ...

    def isfinite(self, x: Any) -> Any:
        """Elementwise finiteness mask."""
        ...

    def nan_to_num(self, x: Any) -> Any:
        """Replace NaN / +Inf / -Inf with zeros (DP-safe sanitization)."""
        ...

    def clamp(self, x: Any, lo: Any = None, hi: Any = None) -> Any:
        """Clamp ``x`` into ``[lo, hi]`` (open on ``None`` bounds)."""
        ...

    def zeros_like(self, x: Any) -> Any:
        """Zeros with the same shape / dtype / device as ``x``."""
        ...

    def concatenate(self, xs: Any, axis: int = 0) -> Any:
        """Concatenate a sequence of arrays along ``axis``."""
        ...

    def astype(self, x: Any, dtype: Any) -> Any:
        """Cast ``x`` to ``dtype``."""
        ...

    def scalar(self, value: Any, *, dtype: Any = None, like: Any = None) -> Any:
        """Materialize ``value`` as a scalar array.

        ``dtype`` and ``like`` control the dtype / device: ``like`` (an
        array) supplies the device, and ``dtype`` overrides the dtype.
        """
        ...

    def promote_dtype(self, a: Any, b: Any) -> Any:
        """Return the dtype that ``a`` and ``b`` promote to."""
        ...

    # --- rng (surface complete; consumer rewrite deferred) ---
    def generator(self, key: RngKey) -> Any:
        """Build a deterministic RNG generator from an immutable key."""
        ...

    def normal(self, shape: Any, *, dtype: Any, generator: Any) -> Any:
        """Sample standard-normal noise of ``shape`` using ``generator``."""
        ...


__all__ = ["Backend"]
