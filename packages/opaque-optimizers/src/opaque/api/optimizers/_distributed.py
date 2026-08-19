"""Cross-rank audit of optimizer state.

Functional optimizer state stays bit-identical across ranks by
construction: ``init_fn(params)`` is deterministic from the parameters,
the gradient is reduced before ``update_fn`` runs, and ``update_fn`` is a
pure function of ``(grad, state)``. A registered handler therefore
*audits* that invariant instead of reducing.

The audit fingerprints each array with ``(sum, sumsq, min, max, numel)``
rather than the single cross-leaf sum :func:`assert_pytree_equal`
computes: that sum lets drift cancel arithmetically — ``[1, 2, 3]``
against ``[0, 3, 3]`` sums alike but differs in every other statistic —
and detecting drift is the whole point here. All five statistics are
permutation-invariant, so a reordering within one array still passes;
element order is not part of what this audit can see. Structure —
dataclass type, dict key set, sequence type and length — is asserted
before recursing, and a leaf of unknown type (``optree.PyTreeSpec`` on
``AdafactorState.treespec``) is compared by ``repr`` so structural drift
still surfaces.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    assert_string_equal,
    register_sync_type,
)
from opaque.api.engine.distributed.collectives import is_distributed
from opaque.api.optimizers import types


def _assert_int_equal(value: int, *, name: str) -> None:
    """Exact integer equality across ranks."""
    assert_scalar_equal(int(value), name=name, atol=0.0, rtol=0.0)


def _assert_array_fingerprint_equal(array: Any, *, name: str) -> None:
    """Compare ``(sum, sumsq, min, max, numel)`` across ranks.

    Four statistics plus the element count cancel only under contrived
    adversarial inputs, unlike a lone sum.
    """
    dimensions = ops.shape(array)
    count = math.prod(dimensions)
    _assert_int_equal(count, name=f"{name}.numel")
    if count == 0:
        # No statistics to compare; pin the rank so an empty array still
        # fails the audit if its shape drifted.
        _assert_int_equal(len(dimensions), name=f"{name}.ndim")
        return
    # Widen before reducing so a low-precision state cannot hide drift below
    # its own resolution.
    widened = ops.astype(ops.detach(array), ops.accumulator_dtype(array))
    assert_scalar_equal(ops.scalar_item(ops.sum(widened)), name=f"{name}.sum")
    assert_scalar_equal(
        ops.scalar_item(ops.sum(ops.multiply(widened, widened))), name=f"{name}.sumsq"
    )
    assert_scalar_equal(ops.scalar_item(ops.amin(widened)), name=f"{name}.min")
    assert_scalar_equal(ops.scalar_item(ops.amax(widened)), name=f"{name}.max")


def _audit_value(value: Any, *, name: str) -> None:
    """Audit one optimizer-state field's structure and leaves across ranks."""
    if value is None:
        return
    if isinstance(value, bool):
        # ``bool`` subclasses ``int``; check it first.
        _assert_int_equal(int(value), name=name)
        return
    if isinstance(value, int):
        _assert_int_equal(value, name=name)
        return
    if isinstance(value, float):
        assert_scalar_equal(value, name=name)
        return
    if isinstance(value, str):
        assert_string_equal(value, name=name)
        return
    if ops.is_array(value):
        _assert_array_fingerprint_equal(value, name=name)
        return
    if isinstance(value, dict):
        # Compare the key set — sorted, so ordering drift is covered too —
        # before comparing any leaf.
        assert_string_equal(
            "\x1f".join(sorted(repr(key) for key in value)), name=f"{name}.<keys>"
        )
        for key, child in value.items():
            _audit_value(child, name=f"{name}[{key!r}]")
        return
    if isinstance(value, (tuple, list)):
        _assert_int_equal(len(value), name=f"{name}.<len>")
        assert_string_equal(type(value).__name__, name=f"{name}.<container>")
        for index, child in enumerate(value):
            _audit_value(child, name=f"{name}[{index}]")
        return
    if dataclasses.is_dataclass(value):
        assert_string_equal(type(value).__name__, name=f"{name}.<dataclass>")
        for field in dataclasses.fields(value):
            _audit_value(getattr(value, field.name), name=f"{name}.{field.name}")
        return
    assert_string_equal(repr(value), name=f"{name}.<repr>")


def sync_optimizer_state(state: Any) -> Any:
    """Assert that an optimizer state is equal across ranks, preserving it.

    Returns ``state`` unchanged: the handler audits rather than reduces,
    and is a no-op in single-process mode.
    """
    if not is_distributed():
        return state
    if isinstance(state, tuple):
        return tuple(sync_optimizer_state(inner) for inner in state)
    if isinstance(state, list):
        return [sync_optimizer_state(inner) for inner in state]
    if not dataclasses.is_dataclass(state):
        return state
    state_name = type(state).__name__
    assert_string_equal(state_name, name=f"{state_name}.<type>")
    for field in dataclasses.fields(state):
        _audit_value(getattr(state, field.name), name=f"{state_name}.{field.name}")
    return state


for _state_type in (
    types.AdamState,
    types.SGDState,
    types.LionState,
    types.RAdamState,
    types.RMSpropState,
    types.AdagradState,
    types.AdadeltaState,
    types.AdafactorState,
    types.AdEMAMixState,
    types.ScheduleFreeState,
):
    register_sync_type(_state_type, sync_optimizer_state)


__all__ = ["sync_optimizer_state"]
