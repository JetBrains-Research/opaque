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
import math
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from opaque.api.engine import ops, runtime
from opaque.api.engine.backend import ensure_backend
from opaque.api.engine.pytree import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)

from .collectives import is_distributed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def assert_scalar_equal(
    value: float | int,
    *,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    """Raise if ``value`` is not equal across ranks within tolerance.

    Integer values are compared exactly through the integer reduction path.
    Floating-point values use the selected provider's scalar reduction dtype.
    """
    if not is_distributed():
        return

    min_value = reduce_scalar(value, op="min")
    max_value = reduce_scalar(value, op="max")
    if isinstance(value, int) and not isinstance(value, bool):
        equal = min_value == max_value
    else:
        equal = math.isclose(min_value, max_value, abs_tol=atol, rel_tol=rtol)
    if not equal:
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
    ensure_backend(pytree)
    if not is_distributed():
        return

    total = 0.0

    def _accumulate(leaf: Any) -> Any:
        nonlocal total
        if ops.is_array(leaf):
            total += float(ops.scalar_item(ops.sum(ops.detach(leaf))))
        return leaf

    tree_map(_accumulate, pytree)
    assert_scalar_equal(
        total,
        name=name,
        atol=atol,
        rtol=rtol,
    )


def reduce_scalar(
    value: float | int,
    op: str = "mean",
) -> float | int:
    """Return a Python scalar reduced across workers."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"value must be a float or int, got {type(value)}")
    reduced = runtime.distributed_all_reduce(value, op=runtime.ReduceOp(op))
    if isinstance(reduced, bool) or not isinstance(reduced, (float, int)):
        raise TypeError(
            "The provider returned a non-scalar result for scalar reduction: "
            f"{type(reduced).__name__}."
        )
    return reduced


def _validate_gathered_tensor_column(
    tensors: list[Any],
    *,
    ranks: list[int],
    leaf_index: int,
    dim: int,
) -> int:
    reference = tensors[0]
    reference_shape = ops.shape(reference)
    if not reference_shape:
        raise ValueError(
            "Distributed tensor gathering cannot concatenate scalar tensor "
            f"leaf {leaf_index}; provide at least one dimension."
        )
    normalized_dim = dim if dim >= 0 else len(reference_shape) + dim
    if normalized_dim < 0 or normalized_dim >= len(reference_shape):
        raise ValueError(
            f"Gather dimension {dim} is out of range for tensor leaf {leaf_index} "
            f"with {len(reference_shape)} dimensions."
        )

    reference_rank = ranks[0]
    for rank, tensor in zip(ranks[1:], tensors[1:], strict=True):
        tensor_shape = ops.shape(tensor)
        if ops.dtype(tensor) != ops.dtype(reference):
            raise TypeError(
                "Distributed tensor gathering requires matching dtypes; "
                f"leaf {leaf_index} has {ops.dtype(reference)} on rank "
                f"{reference_rank} and "
                f"{ops.dtype(tensor)} on rank {rank}."
            )
        if len(tensor_shape) != len(reference_shape):
            raise ValueError(
                "Distributed tensor gathering requires matching tensor ranks; "
                f"leaf {leaf_index} has {len(reference_shape)} dimensions on rank "
                f"{reference_rank} "
                f"and {len(tensor_shape)} on rank {rank}."
            )
        for axis, (expected, actual) in enumerate(
            zip(reference_shape, tensor_shape, strict=True)
        ):
            if axis != normalized_dim and actual != expected:
                raise ValueError(
                    "Distributed tensor gathering requires matching "
                    "non-concatenated dimensions; "
                    f"leaf {leaf_index}, axis {axis} has size {expected} on "
                    f"rank {reference_rank} and {actual} on rank {rank}."
                )
    return normalized_dim


def _merge_gathered_pytrees(
    values: list[Any],
    *,
    dim: int,
) -> Any:
    """Merge rank-ordered optional tensor pytrees after one object collective."""
    present = [(rank, value) for rank, value in enumerate(values) if value is not None]
    if not present:
        return None

    first_rank, first = present[0]
    treedef = tree_structure(first)
    for rank, payload in present[1:]:
        other = tree_structure(payload)
        if other != treedef:
            raise TypeError(
                "Distributed tensor gathering requires matching pytree "
                "structures across non-empty ranks; "
                f"rank {first_rank} has {treedef} and rank {rank} has {other}."
            )

    leaf_lists = [tree_flatten(payload)[0] for _, payload in present]
    if not leaf_lists[0]:
        return first

    merged_leaves: list[Any] = []
    for leaf_index, column in enumerate(zip(*leaf_lists, strict=True)):
        if not all(ops.is_array(leaf) for leaf in column):
            raise TypeError(
                "Distributed tensor gathering supports tensor leaves only; "
                f"leaf {leaf_index} has types "
                f"{[type(leaf).__name__ for leaf in column]}."
            )
        tensors = list(column)
        normalized_dim = _validate_gathered_tensor_column(
            tensors,
            ranks=[rank for rank, _ in present],
            leaf_index=leaf_index,
            dim=dim,
        )
        merged_leaves.append(
            ops.concatenate(
                tensors,
                axis=normalized_dim,
            )
        )
    return tree_unflatten(treedef, merged_leaves)


def gather_tensors(tensor: Any, dim: int = 0) -> Any:
    """Gather compatible variable-size tensors and concatenate along ``dim``."""
    ensure_backend(tensor)
    if not is_distributed():
        return tensor
    return runtime.distributed_all_gather(tensor, axis=dim)


def gather_pytree(pytree: Any, dim: int = 0) -> Any:
    """Gather an optional tensor pytree through one symmetric collective.

    Non-``None`` ranks must provide matching pytree structures and compatible
    tensor leaves. Rank-local ``None`` payloads contribute no rows while still
    participating in the collective.
    """
    ensure_backend(pytree)
    if not is_distributed():
        return pytree

    local_leaves = tree_leaves(pytree)
    local_schema = None
    if pytree is not None:
        leaf_schemas = []
        for leaf in local_leaves:
            if not ops.is_array(leaf):
                leaf_schemas.append(type(leaf))
                continue
            shape = ops.shape(leaf)
            if not shape:
                if dim not in (0, -1):
                    raise IndexError("dim is out of bounds for a scalar array")
                leaf_schemas.append((ops.dtype(leaf), 1, ()))
                continue
            if not -len(shape) <= dim < len(shape):
                raise IndexError(
                    f"dim {dim} is out of bounds for an array with "
                    f"{len(shape)} dimensions"
                )
            normalized_dim = dim % len(shape)
            leaf_schemas.append(
                (
                    ops.dtype(leaf),
                    len(shape),
                    tuple(
                        size
                        for axis, size in enumerate(shape)
                        if axis != normalized_dim
                    ),
                )
            )
        local_schema = (
            tree_structure(pytree),
            tuple(leaf_schemas),
        )
    schemas = runtime.distributed_all_gather_object(local_schema)
    present_schemas = [schema for schema in schemas if schema is not None]
    if not present_schemas:
        return None
    if any(schema != present_schemas[0] for schema in present_schemas[1:]):
        raise TypeError(
            "Distributed tensor gathering requires matching pytree structures "
            "and compatible leaves across non-empty ranks."
        )
    if len(present_schemas) != len(schemas):
        gathered = runtime.distributed_all_gather_object(pytree)
        return _merge_gathered_pytrees(gathered, dim=dim)

    treedef = tree_structure(pytree)
    gathered_leaves = []
    for leaf_index, leaf in enumerate(local_leaves):
        if not ops.is_array(leaf):
            raise TypeError(
                "Distributed tensor gathering supports array leaves only; "
                f"leaf {leaf_index} has type {type(leaf).__name__}."
            )
        gathered_leaves.append(runtime.distributed_all_gather(leaf, axis=dim))
    return tree_unflatten(treedef, gathered_leaves)


def sync_object(
    state: Any,
    field_ops: Mapping[str, str | Callable[..., float | int | None]],
) -> Any:
    """All-reduce scalar fields of a dataclass, returning a new instance.

    ``field_ops`` maps field name to a reduction op string
    (``"sum" | "mean" | "max" | "min" | "product" | "assert_equal" |
    "assert_optional_equal" | "local"``) or to a callable
    ``fn(value[, device]) -> float | int | None``.
    Every dataclass field must be present in this mapping. ``"local"`` marks a
    field whose rank-local value is intentionally carried through unchanged.

    **Callable semantics:** invoked on the raw field value (any type).  If the
    return value is a real ``float`` or ``int`` (not ``bool``), it replaces the
    field in the returned dataclass (same as the legacy numeric-reduction path).
    If the return value is ``None``, the callable is treated as **assertion
    only** — no field update.

    The complete schema is validated before any distributed collective begins,
    and operations execute in dataclass declaration order rather than mapping
    insertion order. This prevents rank-local values from changing the
    collective schedule.
    """
    if not is_dataclass(state):
        raise TypeError(f"state must be a dataclass, got {type(state)}")

    dataclass_fields = fields(state)
    state_fields = {field.name for field in dataclass_fields}
    field_op_names = set(field_ops)
    unknown_fields = field_op_names - state_fields
    missing_fields = state_fields - field_op_names
    if unknown_fields or missing_fields:
        details = []
        if unknown_fields:
            details.append(f"unknown fields: {sorted(unknown_fields)}")
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        raise ValueError(
            "field_ops must define every dataclass field; " + "; ".join(details) + "."
        )

    valid_ops = {
        "sum",
        "mean",
        "max",
        "min",
        "product",
        "assert_equal",
        "assert_optional_equal",
        "local",
    }
    for field in dataclass_fields:
        field_op = field_ops[field.name]
        if isinstance(field_op, str) and field_op not in valid_ops:
            raise ValueError(
                f"field_ops[{field.name!r}] has unsupported operation {field_op!r}. "
                f"Expected one of {sorted(valid_ops)} or a callable."
            )

    ensure_backend(state)
    if not is_distributed():
        return state

    updates: dict[str, Any] = {}
    for field in dataclass_fields:
        field_name = field.name
        field_op = field_ops[field_name]
        value = getattr(state, field_name)
        if field_op == "local":
            continue
        if callable(field_op):
            try:
                result = field_op(value)
            except TypeError:
                result = field_op(value, None)
            if isinstance(result, bool) or not isinstance(result, (float, int)):
                continue
            if isinstance(value, bool):
                raise TypeError(
                    f"Callable field_ops[{field_name!r}] cannot update a bool field; "
                    "use 'local' or return None for assertion-only behavior."
                )
            synced = int(result) if isinstance(value, int) else float(result)
            updates[field_name] = synced
            continue
        if field_op == "assert_equal":
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError(
                    f"field_ops[{field_name!r}]='assert_equal' requires a float or int, "
                    f"got {type(value).__name__}."
                )
            assert_scalar_equal(
                value,
                name=f"{type(state).__name__}.{field_name}",
                atol=0.0,
                rtol=0.0,
            )
            continue
        if field_op == "assert_optional_equal":
            is_present = int(value is not None)
            present_min = reduce_scalar(is_present, op="min")
            present_max = reduce_scalar(is_present, op="max")
            if present_min != present_max:
                raise RuntimeError(
                    f"{type(state).__name__}.{field_name} presence mismatch across ranks."
                )
            if not is_present:
                continue
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError(
                    f"field_ops[{field_name!r}]='assert_optional_equal' requires "
                    f"a float, int, or None, got {type(value).__name__}."
                )
            assert_scalar_equal(
                value,
                name=f"{type(state).__name__}.{field_name}",
                atol=0.0,
                rtol=0.0,
            )
            continue
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(
                f"field_ops[{field_name!r}]={field_op!r} requires a float or int, "
                f"got {type(value).__name__}."
            )
        synced = reduce_scalar(value, op=field_op)
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


def register_sync_type(state_type: type, sync_fn: Callable[[Any], Any]) -> None:
    """Register a sync function for ``state_type``.

    Subsystems (clipping, profiling, noise) call this on import to make their
    state types discoverable by :func:`sync`.  Subclasses inherit the
    registration through :func:`sync`'s MRO walk, so registering a base type
    covers its specialisations unless one of them registers its own handler.
    """
    _SYNC_REGISTRY[state_type] = sync_fn


def _resolve_sync_fn(state_type: type) -> Callable[[Any], Any] | None:
    """Return the handler for ``state_type`` or its nearest registered base.

    Exact type first, then ``__mro__`` order, matching the ``isinstance``
    semantics the underlying sync helpers already implement (for example
    ``sync_clipped_grad_aux`` rebuilds via ``type(aux)`` and so handles any
    ``ClippedGradAux`` subclass).
    """
    fn = _SYNC_REGISTRY.get(state_type)
    if fn is not None:
        return fn
    for base in state_type.__mro__[1:]:
        fn = _SYNC_REGISTRY.get(base)
        if fn is not None:
            return fn
    return None


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

    Dispatches on object type — exact match first, then ``__mro__`` — to
    whichever function was registered via :func:`register_sync_type`.  An
    unregistered type raises ``TypeError``; nothing is passed through
    unsynchronized.

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
            f"No sync function registered for {state_type.__name__}. "
            f"Registered types: {[t.__name__ for t in _SYNC_REGISTRY]}"
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
