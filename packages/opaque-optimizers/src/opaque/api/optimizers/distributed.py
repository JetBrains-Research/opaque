"""Distributed synchronization helpers for functional optimizer state.

Functional optimizer state stays bit-identical across ranks by construction:
``init_fn(params)`` is deterministic from params, the gradient is
``sum_gradients_``-reduced before ``update_fn`` runs, and ``update_fn`` is a
pure function of ``(grad, state)``. So a registered sync handler does not need
to all-reduce — it only needs to *audit* that this invariant still holds.

We register defensive ``assert``-style handlers for every Opaque optimizer
state dataclass.  They walk the dataclass fields and validate tensor /
scalar / structural leaves are equal across ranks via
:func:`assert_pytree_equal` / :func:`assert_scalar_equal` plus the
container-structure helpers in this module.  This mirrors the
:class:`~opaque.dpsgd.noise.GaussianNoiseState` precedent which asserts that
``seed`` and ``step`` match across ranks rather than reducing.

Registration happens at import time of this module.  It is triggered as a
side effect of importing :mod:`opaque.api.optimizers` (the package's
``__init__.py`` imports this module), so anything that imports any optimizer
factory picks up the registrations.  The engine dispatcher's
``_ensure_builtin_sync_types_loaded`` fallback only loads engine clipping and
profiling registrations — it does *not* know about the optimizer package — so
the side-effect import is what makes these handlers discoverable.

Dispatch contract: :func:`opaque.distributed.sync` looks up the handler by the
*exact* type of its argument.  Pass a bare dataclass state
(``AdamState``, ``LionState``, …) and the dispatcher routes here.  For
torchopt chain states (a ``tuple`` of inner per-transform states, as returned
by :func:`opaque.optimizers.adamw` / :func:`lion` / etc.), call
:func:`sync_optimizer_state` directly — ``tuple`` is not a registered type and
cannot be globally registered without intercepting unrelated tuples.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import torch

from opaque.api.engine.distributed import is_distributed, register_sync_type
from opaque.api.engine.distributed._state import (
    assert_pytree_equal,
    assert_scalar_equal,
)

from ._adadelta import AdadeltaState
from ._adafactor import AdafactorState
from ._adagrad import AdagradState
from ._adam import AdamState
from ._ademamix import AdEMAMixState
from ._lion import LionState
from ._radam import RAdamState
from ._rmsprop import RMSpropState
from ._schedule_free import ScheduleFreeState

__all__ = [
    "sync_optimizer_state",
]


# ---------------------------------------------------------------------------
# Cross-rank equality primitives — minimal local helpers
# ---------------------------------------------------------------------------


def _assert_int_equal(value: int, *, name: str) -> None:
    """Exact integer equality across ranks (no float tolerance window).

    ``assert_scalar_equal`` uses ``torch.isclose`` with ``rtol=1e-5``, so at
    large step counts adjacent integers fall within tolerance and the audit
    silently misses single-step drift.  Compare integers exactly instead by
    pinning ``atol=0`` / ``rtol=0``.
    """
    assert_scalar_equal(float(int(value)), name=name, atol=0.0, rtol=0.0)


def _assert_str_equal(value: str, *, name: str) -> None:
    """Cross-rank string equality via 64-bit blake2b fingerprint.

    Python's built-in :func:`hash` is salted per process, so we hash the
    UTF-8 bytes with a stable hasher.  We then collapse the 128-bit digest to
    a ``float`` (two 32-bit halves combined exactly into a 53-bit-mantissa
    float — see :func:`int.from_bytes`) and compare with ``atol=0`` /
    ``rtol=0``.  Two distinct strings have astronomical collision odds at
    this width.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=6).digest()
    fp = float(int.from_bytes(digest, "big"))
    assert_scalar_equal(fp, name=name, atol=0.0, rtol=0.0)


def _assert_tensor_fingerprint_equal(tensor: torch.Tensor, *, name: str) -> None:
    """Stronger-than-sum fingerprint: ``(sum, sum-of-squares, min, max)``.

    :func:`assert_pytree_equal` aggregates a single ``sum`` across all leaves,
    so per-tensor drift can cancel arithmetically (``[1, 2, 3]`` vs
    ``[2, 1, 3]``) and the audit passes.  For optimizer-state audits — where
    drift detection is the entire point — fingerprint each tensor with four
    statistics that cancel only under contrived adversarial inputs.
    """
    if tensor.numel() == 0:
        # Empty tensor has no statistics; assert shape via dim count.
        assert_scalar_equal(float(tensor.ndim), name=f"{name}.ndim", atol=0.0, rtol=0.0)
        return
    t = tensor.detach().double()
    assert_scalar_equal(float(t.sum().item()), name=f"{name}.sum")
    assert_scalar_equal(float((t * t).sum().item()), name=f"{name}.sumsq")
    assert_scalar_equal(float(t.amin().item()), name=f"{name}.min")
    assert_scalar_equal(float(t.amax().item()), name=f"{name}.max")
    assert_scalar_equal(float(tensor.numel()), name=f"{name}.numel", atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Structural / leaf walker
# ---------------------------------------------------------------------------


def _audit_value(value: Any, *, name: str) -> None:
    """Cross-rank audit on one optimizer-state field.

    Asserts both the *structure* (container shape, dataclass type, dict keys,
    sequence length) and the *leaf values* (tensor fingerprint, scalar
    equality, string content) match across ranks before recursing.
    """
    if value is None:
        return
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; check it first.
        _assert_int_equal(int(value), name=name)
        return
    if isinstance(value, int):
        _assert_int_equal(value, name=name)
        return
    if isinstance(value, float):
        assert_scalar_equal(value, name=name)
        return
    if isinstance(value, str):
        _assert_str_equal(value, name=name)
        return
    if isinstance(value, torch.Tensor):
        _assert_tensor_fingerprint_equal(value, name=name)
        return
    if isinstance(value, dict):
        # Assert the key set matches before any leaf comparison.  Using a
        # stable, sort-normalised join also covers ordering drift.
        keys_fp = "\x1f".join(sorted(repr(k) for k in value))
        _assert_str_equal(keys_fp, name=f"{name}.<keys>")
        for k, v in value.items():
            _audit_value(v, name=f"{name}[{k!r}]")
        return
    if isinstance(value, (tuple, list)):
        _assert_int_equal(len(value), name=f"{name}.<len>")
        _assert_str_equal(type(value).__name__, name=f"{name}.<container>")
        for i, v in enumerate(value):
            _audit_value(v, name=f"{name}[{i}]")
        return
    if dataclasses.is_dataclass(value):
        _assert_str_equal(type(value).__name__, name=f"{name}.<dataclass>")
        for f in dataclasses.fields(value):
            _audit_value(getattr(value, f.name), name=f"{name}.{f.name}")
        return
    # Unknown leaf types (e.g. ``optree.PyTreeSpec`` on
    # ``AdafactorState.treespec``) — fingerprint via ``repr`` so the audit
    # still notices structural drift between ranks.
    _assert_str_equal(repr(value), name=f"{name}.<repr>")


# ---------------------------------------------------------------------------
# Public handler + registration
# ---------------------------------------------------------------------------


def sync_optimizer_state(state: Any) -> Any:
    """Validate optimizer state matches across ranks; return ``state``.

    Pure-functional optimizers cannot drift after ``sum_gradients_``, so the
    handler is an audit, not a reduction. In single-process mode it's a no-op
    and the input object is returned unchanged.

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
        # ``EmptyState`` from torchopt is a non-dataclass class that carries
        # no fields; nothing to audit.
        return state
    state_name = type(state).__name__
    _assert_str_equal(state_name, name=f"{state_name}.<type>")
    for f in dataclasses.fields(state):
        _audit_value(getattr(state, f.name), name=f"{state_name}.{f.name}")
    return state


# Re-export for back-compat with callers that imported ``assert_pytree_equal`` /
# ``assert_scalar_equal`` from this module.
__all__ += ["assert_pytree_equal", "assert_scalar_equal"]


for _state_type in (
    AdamState,
    AdadeltaState,
    AdEMAMixState,
    AdafactorState,
    AdagradState,
    LionState,
    RAdamState,
    RMSpropState,
    ScheduleFreeState,
):
    register_sync_type(_state_type, sync_optimizer_state)
