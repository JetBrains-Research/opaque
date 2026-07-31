"""State synchronization, gathering, and type-dispatched sync for DP training.

This module exposes:

- scalar/tensor helpers: ``reduce_scalar``, ``gather_tensors``, ``gather_pytree``
- sanity checks: ``assert_scalar_equal``, ``assert_pytree_equal``
- dataclass field sync: ``sync_object``
- type registry + dispatcher: ``register_sync_type``, ``sync``

The dispatcher is how user code synchronizes algorithm-specific states
(clipping, profiler, etc.) without having to know about each subsystem's
sync function: each subsystem registers its types on import.
"""

from __future__ import annotations

import contextlib
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from opaque.api.engine.pytree import tree_map

from .collectives import all_reduce_, get_world_size, is_distributed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def assert_scalar_equal(
    value: float | int,
    *,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    device: torch.device | None = None,
) -> None:
    """Raise if ``value`` is not equal across ranks within tolerance."""
    if not is_distributed():
        return

    min_value = reduce_scalar(float(value), op="min", device=device)
    max_value = reduce_scalar(float(value), op="max", device=device)
    if not torch.isclose(
        torch.tensor(min_value),
        torch.tensor(max_value),
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError(
            f"{name} mismatch across ranks: min={min_value}, max={max_value}."
        )


def assert_pytree_equal(
    pytree: Any,
    *,
    name: str = "pytree",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Fingerprint-check a pytree for equality across ranks.

    Uses a cheap scalar (sum of elements) rather than transferring the whole
    tree. Useful for debugging divergence of params/grads.
    """
    if not is_distributed():
        return

    total = 0.0

    def _accumulate(leaf: Any) -> Any:
        nonlocal total
        if isinstance(leaf, torch.Tensor):
            total += leaf.detach().double().sum().item()
        return leaf

    tree_map(_accumulate, pytree)
    assert_scalar_equal(total, name=name, atol=atol, rtol=rtol)


def reduce_scalar(
    value: float,
    op: str = "mean",
    device: torch.device | None = None,
) -> float:
    """All-reduce a Python scalar and return the reduced float.

    For NCCL backends, a CUDA device is required. If ``device`` is None a
    sensible one is inferred (current CUDA device for NCCL, else CPU).
    """
    if not is_distributed():
        return value

    if device is None:
        backend = (
            dist.get_backend()
            if dist.is_available() and dist.is_initialized()
            else None
        )
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Distributed backend is 'nccl' but CUDA is not available; "
                    "provide an explicit `device` to `reduce_scalar` or initialize "
                    "with a CUDA-capable process."
                )
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    all_reduce_(tensor, op=op)
    return tensor.item()


def gather_tensors(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather variable-size tensors across ranks and concatenate along ``dim``."""
    if not is_distributed():
        return tensor

    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, tensor.cpu())

    gathered_tensors: list[torch.Tensor] = [
        t.to(tensor.device) for t in gathered if t is not None
    ]
    return torch.cat(gathered_tensors, dim=dim)


def gather_pytree(pytree: Any) -> Any:
    """Gather each tensor leaf across ranks; preserves ``None`` leaves."""
    if not is_distributed():
        return pytree

    def _gather(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return gather_tensors(leaf, dim=0)
        raise TypeError(
            f"Distributed aux gathering supports tensor leaves only; got {type(leaf)}."
        )

    return tree_map(_gather, pytree)


def sync_object(
    state: Any,
    field_ops: Mapping[str, str | Callable[..., float | int | None]] | None = None,
    device: torch.device | None = None,
) -> Any:
    """All-reduce scalar fields of a dataclass, returning a new instance.

    ``field_ops`` maps field name to a reduction op string
    (``"sum" | "mean" | "max" | "min" | "product" | "assert_equal"``) or to a
    callable ``fn(value[, device]) -> float | int | None``.

    **Callable semantics:** invoked on the raw field value (any type).  If the
    return value is a real ``float`` or ``int`` (not ``bool``), it replaces the
    field in the returned dataclass (same as the legacy numeric-reduction path).
    If the return value is ``None``, the callable is treated as **assertion
    only** — no field update.

    Defaults to averaging all numeric fields when ``field_ops`` is None.
    """
    if not is_distributed():
        return state

    if not is_dataclass(state):
        raise TypeError(f"state must be a dataclass, got {type(state)}")

    state_fields = {f.name for f in fields(state)}

    if field_ops is None:
        field_ops = {}
        for f in fields(state):
            val = getattr(state, f.name)
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                field_ops[f.name] = "mean"
    else:
        invalid_fields = set(field_ops) - state_fields
        if invalid_fields:
            raise ValueError(
                f"field_ops contains non-existent fields: {invalid_fields}. "
                f"Available fields: {state_fields}"
            )

    updates: dict[str, Any] = {}
    for field_name, field_op in field_ops.items():
        value = getattr(state, field_name)
        if callable(field_op):
            try:
                result = field_op(value, device)
            except TypeError:
                result = field_op(value)
            if isinstance(result, bool) or not isinstance(result, (float, int)):
                continue
            synced = int(result) if isinstance(value, int) else float(result)
            updates[field_name] = synced
            continue
        if not isinstance(value, (float, int)):
            continue
        if field_op == "assert_equal":
            assert_scalar_equal(
                float(value), name=f"{type(state).__name__}.{field_name}"
            )
            continue
        if isinstance(field_op, str):
            synced = reduce_scalar(value, op=field_op, device=device)
        else:
            raise TypeError(
                f"field_ops[{field_name}] must be str or callable, got {type(field_op)}"
            )
        if isinstance(value, int):
            synced = int(synced)
        updates[field_name] = synced

    if updates:
        state = type(state)(
            **{**{f.name: getattr(state, f.name) for f in fields(state)}, **updates}
        )
    return state


# ---------------------------------------------------------------------------
# Type-based sync dispatcher
# ---------------------------------------------------------------------------

_SYNC_REGISTRY: dict[type, Callable[[Any], Any]] = {}

# Memoizes the ``__mro__`` walk in :func:`_resolve_sync_fn`; cleared on
# every registration so a late registration is never shadowed.
_SYNC_RESOLVED: dict[type, Callable[[Any], Any] | None] = {}


def register_sync_type(state_type: type, sync_fn: Callable[[Any], Any]) -> None:
    """Register a sync function for ``state_type`` and its subclasses.

    Subsystems (clipping, profiling, noise) call this on import to make their
    state types discoverable by :func:`sync`. A subclass resolves to the
    nearest registered base class, so a variant that adds no fields (e.g.
    the AUTO-S clipping aux) is covered by its base registration; a subclass
    that needs different handling registers its own function.
    """
    _SYNC_REGISTRY[state_type] = sync_fn
    _SYNC_RESOLVED.clear()


def _resolve_sync_fn(state_type: type) -> Callable[[Any], Any] | None:
    """Return the sync function for ``state_type`` or its nearest base."""
    cached = _SYNC_RESOLVED.get(state_type, ...)
    if cached is not ...:
        return cached  # type: ignore[return-value]

    resolved: Callable[[Any], Any] | None = None
    for base in state_type.__mro__:
        candidate = _SYNC_REGISTRY.get(base)
        if candidate is not None:
            resolved = candidate
            break

    _SYNC_RESOLVED[state_type] = resolved
    return resolved


def _ensure_builtin_sync_types_loaded() -> None:
    """Import internal registrations the first time a dispatch misses.

    Clipping registers itself synchronously; the performance profiler is
    a soft dependency — missing it must not break ``sync()``.  Profiling
    registers ``PerfState`` at the impl path; importing through the
    façade would require an extra side-effect import there, so we target
    the impl module directly.
    """
    import opaque.api.engine.clipping._distributed

    with contextlib.suppress(ImportError):
        import opaque.api.engine.profiling._distributed  # noqa: F401


def sync(*states: Any) -> Any:
    """Synchronize one or more registered state/aux objects across ranks.

    Dispatches on object type — exact type first, then the nearest base class
    registered via :func:`register_sync_type` — and raises for a type that
    resolves to nothing. Skipping an unrecognized state would leave each rank
    training on its own shard with no indication that the collective never
    happened.

    Returns:
        A single synchronized object if one argument was passed, otherwise a
        tuple matching the input order.
    """

    def _sync_one(single: Any) -> Any:
        state_type = type(single)
        sync_fn = _resolve_sync_fn(state_type)
        if sync_fn is None:
            _ensure_builtin_sync_types_loaded()
            sync_fn = _resolve_sync_fn(state_type)
        if sync_fn is not None:
            return sync_fn(single)
        raise TypeError(
            f"No sync function registered for {state_type.__name__} or any of "
            f"its base classes. Registered types: "
            f"{[t.__name__ for t in _SYNC_REGISTRY]}"
        )

    if len(states) == 1:
        return _sync_one(states[0])
    if len(states) > 1:
        return tuple(_sync_one(s) for s in states)
    return ()


__all__ = [
    "assert_pytree_equal",
    "assert_scalar_equal",
    "gather_pytree",
    "gather_tensors",
    "reduce_scalar",
    "register_sync_type",
    "sync",
    "sync_object",
]
