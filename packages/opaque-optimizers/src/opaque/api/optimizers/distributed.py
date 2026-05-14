"""Distributed synchronization helpers for functional optimizer state.

Functional optimizer state stays bit-identical across ranks by construction:
``init_fn(params)`` is deterministic from params, the gradient is
``sum_gradients_``-reduced before ``update_fn`` runs, and ``update_fn`` is a
pure function of ``(grad, state)``. So a registered sync handler does not need
to all-reduce — it only needs to *audit* that this invariant still holds.

We register defensive ``assert``-style handlers for every Opaque optimizer
state. They walk the dataclass fields and validate tensor / scalar leaves are
equal across ranks via :func:`assert_pytree_equal` /
:func:`assert_scalar_equal`. This mirrors the
:class:`~opaque.dpsgd.noise.GaussianNoiseState` precedent which asserts that
``seed`` and ``step`` match across ranks rather than reducing.

The handlers are registered at import time. The dispatcher in
:func:`opaque.distributed.sync` discovers them via
``_ensure_builtin_sync_types_loaded`` (see
:mod:`opaque.distributed.state`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from opaque.api.engine.distributed import is_distributed, register_sync_type
from opaque.api.engine.distributed._state import assert_pytree_equal, assert_scalar_equal

from ._adafactor import AdafactorState
from ._adagrad import AdagradState
from ._adam import AdamState
from ._ademamix import AdEMAMixState
from ._lion import LionState
from ._rmsprop import RMSpropState
from ._schedule_free import ScheduleFreeState

__all__ = [
    "sync_optimizer_state",
]


def _check_value(value: Any, *, name: str) -> None:
    """Defensive cross-rank equality check on one optimizer-state field.

    Recurses into containers (dict / tuple / list / nested dataclasses) so a
    field like ``AdafactorState.v_flat`` (tuple of tuple of tensors) or
    ``ScheduleFreeState.inner`` (torchopt chain state — usually a tuple of
    inner-state dataclasses) is fully covered.
    """
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        assert_scalar_equal(float(value), name=name)
        return
    if isinstance(value, torch.Tensor):
        assert_pytree_equal(value, name=name)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _check_value(v, name=f"{name}[{k!r}]")
        return
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _check_value(v, name=f"{name}[{i}]")
        return
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _check_value(getattr(value, f.name), name=f"{name}.{f.name}")
        return
    # Unknown type (e.g. optree treespec): conservatively skip.  The structural
    # equality of treespecs is implied by the surrounding tensor leaves.


def sync_optimizer_state(state: Any) -> Any:
    """Validate optimizer state matches across ranks; return ``state``.

    Pure-functional optimizers cannot drift after ``sum_gradients_``, so the
    handler is an audit, not a reduction. In single-process mode it's a no-op.

    Accepts both bare dataclass states (``AdamState``, ``LionState``, …) and
    torchopt chain states (a ``tuple`` of inner per-transform states, as
    returned by :func:`opaque.optimizers.adamw` / :func:`lion` / etc.). For
    chain states, recurses into each element so every leaf state is audited.
    """
    if not is_distributed():
        return state
    if isinstance(state, tuple):
        return tuple(sync_optimizer_state(s) for s in state)
    if isinstance(state, list):
        return [sync_optimizer_state(s) for s in state]
    if not dataclasses.is_dataclass(state):
        # ``EmptyState`` from torchopt is an instance of a non-dataclass class
        # that holds nothing; nothing to audit.
        return state
    state_name = type(state).__name__
    for f in dataclasses.fields(state):
        _check_value(getattr(state, f.name), name=f"{state_name}.{f.name}")
    return state


for _state_type in (
    AdamState,
    LionState,
    RMSpropState,
    AdagradState,
    AdEMAMixState,
    AdafactorState,
    ScheduleFreeState,
):
    register_sync_type(_state_type, sync_optimizer_state)
