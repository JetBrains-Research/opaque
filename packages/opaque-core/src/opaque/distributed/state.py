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

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist

from opaque.pytree import tree_map

from .collectives import all_reduce_, get_world_size, is_distributed


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
    field_ops: Mapping[str, str | Callable[..., float]] | None = None,
    device: torch.device | None = None,
) -> Any:
    """All-reduce scalar fields of a dataclass, returning a new instance.

    ``field_ops`` maps field name to a reduction op string
    (``"sum" | "mean" | "max" | "min" | "product" | "assert_equal"``) or to a
    callable ``fn(value[, device]) -> float``. Defaults to averaging all
    numeric fields when ``field_ops`` is None.
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
        if not isinstance(value, (float, int)):
            continue
        if field_op == "assert_equal":
            assert_scalar_equal(
                float(value), name=f"{type(state).__name__}.{field_name}"
            )
            continue
        if isinstance(field_op, str):
            synced = reduce_scalar(value, op=field_op, device=device)
        elif callable(field_op):
            numeric_value = float(value)
            try:
                synced = field_op(numeric_value, device)
            except TypeError:
                synced = field_op(numeric_value)
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


def register_sync_type(state_type: type, sync_fn: Callable[[Any], Any]) -> None:
    """Register a sync function for ``state_type``.

    Subsystems (clipping, profiling, noise) call this on import to make their
    state types discoverable by :func:`sync`.
    """
    _SYNC_REGISTRY[state_type] = sync_fn


def _ensure_builtin_sync_types_loaded() -> None:
    """Import internal registrations the first time a dispatch misses.

    Clipping registers itself synchronously; the performance profiler is
    a soft dependency — missing it must not break ``sync()``.
    """
    import opaque.clipping.distributed  # noqa: F401

    try:
        import opaque.profiling.distributed  # noqa: F401
    except ImportError:
        pass


def sync(*states: Any) -> Any:
    """Synchronize one or more registered state/aux objects across ranks.

    Dispatches based on object type to whichever function was registered via
    :func:`register_sync_type`.

    Returns:
        A single synchronized object if one argument was passed, otherwise a
        tuple matching the input order.
    """

    def _sync_one(single: Any) -> Any:
        state_type = type(single)
        if state_type not in _SYNC_REGISTRY:
            _ensure_builtin_sync_types_loaded()
        if state_type in _SYNC_REGISTRY:
            return _SYNC_REGISTRY[state_type](single)
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
    "sync_object",
    "register_sync_type",
    "sync",
]
